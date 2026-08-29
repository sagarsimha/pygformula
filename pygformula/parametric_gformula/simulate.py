import numpy as np
import pandas as pd
import re
import math
import types
from functools import reduce
import operator
from scipy.stats import truncnorm
from .histories import update_precoded_history, update_custom_history
from ..utils.helper import categorical_func
from ..interventions import intervention_func
from ..interventions import natural
from ..interventions import static

import string

def binorm_sample(simprob, simul_rng):
    return simul_rng.binomial(n=1, p=simprob, size=1)[0]

def norm_sample(mean, rmse, simul_rng):
    return simul_rng.normal(loc=mean, scale=rmse, size=1)[0]

def truc_sample(mean, rmse, a, b):
    return truncnorm.rvs((a - mean) / rmse, (b - mean) / rmse, loc=mean, scale=rmse)

def poisson_sample(lam, simul_rng):
    return simul_rng.poisson(lam)

def apply_bounds(prediction, cov, bounds, integer=False):
    """
    Enforce empirical bounds for a simulated covariate. Winsorization is applied.

    Parameters
    ----------
    prediction : array-like or pd.Series
        Simulated values.
    cov : str
        Covariate name.
    bounds : dict
        Dictionary of bounds from fit step.
    integer : bool
        If True, round to nearest integer after clipping.

    Returns
    -------
    np.ndarray or pd.Series
    """
    if cov in bounds:
        lower, upper = bounds[cov]
        prediction = np.where(prediction < lower, lower, prediction)
        prediction = np.where(prediction > upper, upper, prediction)

    if integer:
        prediction = np.round(prediction).astype(int)

    return prediction

def simulate_postdischarge_variable_hazard(
    pool_with_A1_t_t: pd.DataFrame,
    zmodel,
    zmodel_predict_custom,
    z_outcome_fit,
    simul_rng,
    *,
    id_col: str = "admission_id",
    tD_col: str = "tD",     # "t0" is the discharge index tD since the row at discharge is fed.
    t_max: int = 13,        # t in [0, 13], 14 bins total.
    return_t_death: bool = False,
) -> pd.DataFrame:
    """
    Simulate post-discharge death by K under a variable (time-since-discharge)
    discrete-time hazard model.

    Schema:
    - Grid t in [0, t_max] with t_max = 13 (13-day horizon, 12h bins).
    - W^Y_t = [grid_end(t), grid_end(t+1)) = [12(t+1), 12(t+2))h since intime.
    - K = grid_end(t_max + 1) = 12(t_max + 2)h since intime = 156h for t_max=13.

    Risk-set semantics:
    - For each discharged row at tD, expand into (t_max + 1 - tD) post-discharge
      bins with tsd = 0, 1, ..., t_max - tD. The hazard model predicts a per-bin
      probability p_{i,tsd} that may depend on (frozen z_covs, tsd, tD).
    - Per-bin Bernoulli draws z_{i, tsd} are taken independently.
    - death_by_K_i = 1 iff any z_{i, tsd} == 1.
    - If return_t_death and death_by_K_i == 1: t_death_i = tD + (first tsd at
      which z_{i, tsd} == 1).

    Differences from the constant-hazards version:
    - The cumulative q_i is computed by per-bin Bernoulli sampling rather than
      the closed form 1 - (1 - p)^n. This allows the model to express time-varying
      hazard (e.g. front-loaded mortality immediately post-discharge), and removes
      the systematic over-estimation of cumulative q under front-loaded true
      hazard (Jensen's inequality on log(1-p) vs log(1-bar p)).
    - The output index is preserved to match the call site's
      pool_with_A1_t_t.index, so `result['death_by_K']` can be assigned back into
      the simulator's pool DataFrame without realignment.

    Return:
    - DataFrame indexed like pool_with_A1_t_t with columns:
        admission_id, tD, n_intervals, mean_p_per_bin, death_by_K
        (+ t_death if return_t_death=True)
    """

    df = pool_with_A1_t_t.copy()

    if df.shape[0] == 0:
        return pd.DataFrame(index=df.index, columns=[id_col, "tD", "death_by_K"])

    if tD_col not in df.columns:
        raise KeyError(f"'{tD_col}' not found. Set tD_col to your discharge index column (e.g. 't0').")

    df[tD_col] = df[tD_col].astype(int)

    tD_arr = df[tD_col].to_numpy(dtype=int)
    n = (t_max + 1 - tD_arr).astype(int)
    n = np.maximum(n, 0)

    n_disch = len(df)

    # ---- Expand each discharge row into n_i post-discharge bins ----
    # Skip discharges with n == 0 (would mean tD > t_max — should not occur).
    keep = n > 0
    if not keep.all():
        n_dropped = int((~keep).sum())
        print(f"simulate_postdischarge_variable_hazard: {n_dropped} discharge rows "
              f"have tD > t_max and contribute zero post-discharge risk.")

    df_keep = df.loc[keep].copy()
    tD_keep = tD_arr[keep]
    n_keep = n[keep]

    if df_keep.shape[0] == 0:
        # All discharges out of horizon — return zero deaths aligned to input.
        out = pd.DataFrame({
            id_col: df[id_col].values if id_col in df.columns else np.arange(n_disch),
            "tD": tD_arr,
            "n_intervals": n,
            "mean_p_per_bin": np.zeros(n_disch),
            "death_by_K": np.zeros(n_disch, dtype=int),
        }, index=df.index)
        if return_t_death:
            out["t_death"] = np.nan
        return out

    # Repeat each discharge row n_i times. rep_idx selects from df_keep positions.
    rep_idx = np.repeat(np.arange(len(df_keep)), n_keep)
    post = df_keep.iloc[rep_idx].copy()

    # Per-bin offsets within each block: tsd = 0, 1, ..., n_i - 1
    ends = np.cumsum(n_keep)
    starts = ends - n_keep
    total = int(ends[-1])
    tsd = np.empty(total, dtype=int)
    for i in range(len(n_keep)):
        tsd[starts[i]:ends[i]] = np.arange(int(n_keep[i]), dtype=int)

    post["tsd"] = tsd
    post["tD"] = np.repeat(tD_keep, n_keep)
    # 't' (grid index for this bin) is informative but not required by predict.
    post["t"] = post["tD"].to_numpy(dtype=int) + post["tsd"].to_numpy(dtype=int)

    # ---- Predict per-bin hazard ----
    if zmodel_predict_custom is not None:
        p = zmodel_predict_custom(zmodel=zmodel, new_df=post, fit=z_outcome_fit)
    else:
        # statsmodels will use the formula design-info inside z_outcome_fit
        p = z_outcome_fit.predict(post).astype(float)

    p = np.asarray(p, dtype=float)
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    # ---- Per-bin Bernoulli draws ----
    u = simul_rng.uniform(size=p.shape[0])
    z = (u < p).astype(np.int8)

    # ---- Aggregate per discharge row ----
    # death_by_K = max over bins; t_death = tD + first tsd with z==1 (if any).
    death_by_K_keep = np.zeros(len(df_keep), dtype=int)
    if return_t_death:
        t_death_keep = np.full(len(df_keep), np.nan)
    mean_p_keep = np.zeros(len(df_keep))

    for i in range(len(df_keep)):
        s, e = int(starts[i]), int(ends[i])
        block_z = z[s:e]
        block_p = p[s:e]
        mean_p_keep[i] = float(block_p.mean()) if e > s else 0.0
        if block_z.any():
            death_by_K_keep[i] = 1
            if return_t_death:
                first_pos = int(np.argmax(block_z))  # first True/1
                t_death_keep[i] = float(tD_keep[i] + first_pos)

    # ---- Map keep-only results back to the full input ----
    death_by_K = np.zeros(n_disch, dtype=int)
    mean_p = np.zeros(n_disch)
    death_by_K[keep] = death_by_K_keep
    mean_p[keep] = mean_p_keep

    out = pd.DataFrame({
        id_col: df[id_col].values if id_col in df.columns else np.arange(n_disch),
        "tD": tD_arr,
        "n_intervals": n,
        "mean_p_per_bin": mean_p,
        "death_by_K": death_by_K,
    }, index=df.index)

    if return_t_death:
        t_death = np.full(n_disch, np.nan)
        t_death[keep] = t_death_keep
        out["t_death"] = t_death

    return out


def simulate(simul_rng, time_points, time_name, id, obs_data, basecovs,
             outcome_type, rmses, bounds, intervention, intervention_function,
             custom_histvars, custom_histories, covpredict_custom, 
             zmodel, 
             z_outcome_fit, 
             zmodel_predict_custom,  
             #ymodel,
             #outcome_fit, 
             #ymodel_predict_custom, 
             outcome_name,
             competing, compevent_name, compevent_model, compevent_fit, compevent_cens, trunc_params,
             visit_names, visit_covs, ts_visit_names, max_visits, time_thresholds, baselags, below_zero_indicator,
             restrictions, yrestrictions, compevent_restrictions, covnames, covtypes, covmodels,
             covariate_fits, cov_hist, sim_trunc, I_fit, I_name,
             I_model=None, I_model_predict_custom=None):

    """
    This is an internal function to perform Monte Carlo simulation of the parametric g-formula.

    Parameters
    ----------
    seed: Int
        An integer indicating the starting seed for simulations and bootstrapping.

    time_points: Int
        An integer indicating the number of time points to simulate. It is set equal to the maximum number of records
        that obs_data contains for any individual plus 1, if not specified by users.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    id: Str
        A string specifying the name of the id variable in obs_data.

    covtypes: List
        A list of strings specifying the “type” of each time-varying covariate included in covnames.
        The supported types: "binary", "normal", "categorical", "bounded normal", "zero-inflated normal",
        "truncated normal", "absorbing", "categorical time" "square time" and "custom". The list must be the same length
        as covnames and in the same order.

    obs_data: DataFrame
        A data frame containing the observed data.

    covnames: List
        A list of strings specifying the names of the time-varying covariates in obs_data.

    basecovs: List
        A list of strings specifying the names of baseline covariates in obs_data. These covariates should not be
        included in covnames.

    covmodels: List
        A list of strings, where each string is the model statement of the time-varying covariate. The list
        must be the same length as covnames and in the same order. If a model is not required for a certain covariate,
        it should be set to 'NA' at that index.

    cov_hist: Dict
        A dictionary whose keys are covariate names and values are sub-dictionaries with historical information for
        covariates. Each sub-dictionaty contains keys 'lagged', 'cumavg' and 'lagavg', the corresponding value for the
        key 'lagged' is a two-element list where the first element is a list with all lagged terms, the second element
        is a list with the corresponding lagged numbers. Same for the key 'lagavg'. The corresponding value for the key
        'cumavg' is a list with all cumavg terms.

    covariate_fits: List
         A list that contains the fitted model for all time-varying covariates.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    rmses: List
        A list that contains the root mean square errors (rmses) of all the fitted models.

    bounds: List
        A list that contains the bound for all time-varying covariates in the obs_data.

    intervention: List
        List of lists. The k-th list contains the intervention list on k-th treatment name in the intervention.
        The intervention list contains a function implementing a particular intervention on the treatment variable,
        required values for the intervention function and a list of time points in which the intervention
        is applied.

    custom_histvars: List
        A list of strings, each of which specifies the names of the time-varying covariates with user-specified custom histories.

    custom_histories: List
        A list of functions, each function is the user-specified custom history functions for covariates. The list
        should be the same length as custom_histvars and in the same order.

    covpredict_custom: List
        A list, each element could be 'NA' or a user-specified predict function. The non-NA value is set
        for the covariates with custom type. The 'NA' value is set for other covariates. The list must be the
        same length as covnames and in the same order.

    ymodel_predict_custom: Function
        A user-specified predict function for the outcome variable.

    ymodel: Str
        A string specifying the model statement for the outcome variable.

    outcome_fit: Class
        A class object of the fitted model for outcome.

    outcome_name: Str
        A string specifying the name of the outcome variable in obs_data.

    competing: Bool
        A boolean value indicating if there is a competing event in obs_data.

    compevent_name: Str
        A string specifying the name of the competing event variable in obs_data. Only applicable for survival outcomes.

    compevent_model: Str
        A string specifying the model statement for the competing event variable. Only applicable for survival outcomes.

    compevent_fit: Class
        A class object of the fitted model for the competing event.

    compevent_cens: Bool
        A boolean value indicating whether to treat competing events as censoring events.

    trunc_params: List
        A list, each element could be 'NA' or a two-element list. If not 'NA', the first element specifies the truncated
        value and the second element specifies the truncated direction (‘left’ or ‘right’). The non-NA value is set
        for the truncated normal covariates. The 'NA' value is set for other covariates. The list should be the same
        length as covnames and in the same order.

    visit_names: List
        A list, each of which is a string specifying the covariate name of a visit process.

    visit_covs: List
        A list of strings, each of which specifying the name of a covariate whose modeling depends on the visit process.

    ts_visit_names: List
        A list of strings, each of which indicates the number of consecutive missed visits for one covariate before an
        individual is censored.

    max_visits: List
        A list of integers, each integer indicates the maximum number of consecutive missed visits for one covariate that
        has a visit process.

    time_thresholds: List
        A list of integers that splits the time points into different intervals. It is used to create the variable
        "categorical time".

    baselags: Bool
        A boolean value specifying the convention used for lagi and lag_cumavgi terms in the model statements when
        pre-baseline times are not included in obs_data and when the current time index, t, is such that t < i. If this
        argument is set to False, the value of all lagi and lag_cumavgi terms in this context are set to 0 (for
        non-categorical covariates) or the reference level (for categorical covariates). If this argument is set to
        True, the value of lagi and lag_cumavgi terms are set to their values at time 0. The default is False.

    below_zero_indicator: Bool
        A boolean value indicating if the obs_data contains pre-baseline times.

    restrictions: List
        List of lists. Each inner list contains its first entry the covariate name of that its deterministic knowledge
        is known; its second entry is a dictionary whose key is the conditions which should be True when the covariate
        is modeled, the third entry is the value that is set to the covariate during simulation when the conditions
        in the second entry are not True.

    yrestrictions: List
        List of lists. For each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the outcome is modeled, the second entry is the value that is set to the outcome during
        simulation when the conditions in the first entry are not True.

    compevent_restrictions: List, default is None
        List of lists. For each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the competing event is modeled, the second entry is the value that is set to the competing
        event during simulation when the conditions in the first entry are not True. Only applicable for survival outcomes.

    sim_trunc: Bool
        A boolean value indicating if the simulated values of normal covariates are truncated by the observed ranges.

    Returns
    -------
    g_result: List
        A list contains the parametric estimate of risks at all the time points for a particular intervention.

    pool: DataFrame
        A simulated data frame under a particular intervention.

    """

    if basecovs:
        column_names = [id] + [time_name] + covnames + basecovs if covnames is not None else [id] + [time_name] + basecovs
    else:
        column_names = [id] + [time_name] + covnames if covnames is not None else [id] + [time_name]
    if ts_visit_names:
        column_names.extend(ts_visit_names)
    pool = obs_data.loc[:, column_names]

    # Initialise the bin-level I-event indicator on pool, so that it propagates
    # through every slicing operation later. Without this, `pool_with_I1_t = pool[...]`
    # would produce a DataFrame missing the I_event_t column (or with NaN where
    # the bin-level sample value should be), and the per-t I-hazard plot would
    # show zero deaths.
    pool = pool.copy()
    pool["I_event_t"] = np.nan


    #Changes for NC, SR, DR
    final_df_list = [] # Collect dataframes of ids which has A=1 at t=0, t=1, t=2 and so on.
    ###########################################################################################

    for t in range(0, time_points):
        print('Time point', t)
        if t == 0:
            pool = pool[pool[time_name] <= t].copy() # pool all data until t=0
            new_df = pool[pool[time_name] == t] # pick only t=0

            # Intervening happens here. new_df is updated within the function.
            intervention_func(new_df=new_df, pool=pool, intervention=intervention, time_name=time_name, t=t) # does intervention and new_df is updated within

            # Compulsory discharge at "time_points" for static.
            if (intervention_function == static) and (t == time_points - 1):
                new_df.loc[new_df[time_name] == t, 'A'] = 1

            pool.loc[pool[time_name] == t] = new_df
            if covnames is not None:
                update_precoded_history(pool, covnames, cov_hist, covtypes, time_name, id, below_zero_indicator,
                                      baselags, ts_visit_names)
                if custom_histvars is not None:
                    update_custom_history(pool, custom_histvars, custom_histories, time_name, t, id)
            new_df = pool[pool[time_name] == t].copy()

            '''if competing and not compevent_cens:
                prob_D = compevent_fit.predict(new_df)
                new_df['prob_D'] = prob_D

                if compevent_restrictions is not None:
                    for restriction in compevent_restrictions:
                        conditions = restriction[0]
                        masks = []
                        for cond_var, condition in conditions.items():
                            mask = new_df[cond_var].apply(condition)
                            masks.append(mask)
                        comp_restrict_mask = reduce(operator.and_, masks)
                        new_df.loc[~comp_restrict_mask, 'prob_D'] = restriction[1]

                new_df[compevent_name] = new_df['prob_D'].apply(binorm_sample)'''


            '''if ymodel_predict_custom is not None:
                pre_y = ymodel_predict_custom(ymodel=ymodel, new_df=new_df, fit=outcome_fit)
            else:
                pre_y = outcome_fit.predict(new_df)'''

            '''if outcome_type == 'survival':
                new_df['prob1'] = pre_y

                if yrestrictions is not None:
                    for restriction in yrestrictions:
                        conditions = restriction[0]
                        masks = []
                        for cond_var, condition in conditions.items():
                            mask = new_df[cond_var].apply(condition)
                            masks.append(mask)
                        restrict_mask = reduce(operator.and_, masks)
                        new_df.loc[~restrict_mask, 'prob1'] = restriction[1]

                new_df['prob0'] = 1 - new_df['prob1']
                new_df[outcome_name] = new_df['prob1'].apply(binorm_sample)'''

            #if intervention != natural: # Changes for NC
            '''if intervention_function == static:
                if outcome_type == 'binary_eof':
                    new_df['Py'] = 'NA' if t < time_points - 1 else pre_y
                if outcome_type == 'continuous_eof':
                    new_df['Ey'] = 'NA' if t < time_points - 1 else pre_y'''

            '''if competing and not compevent_cens:
                new_df.loc[new_df[compevent_name] == 1, outcome_name] = pd.NA'''

            pool = pd.concat([pool[pool[time_name] < t], new_df])
            pool.sort_values([id, time_name], ascending=[True, True], inplace=True)


            # If A=1,
                # Stop simulation for that ID.
                # Predict post-discharge risk P(Z) until time point K, given history until discharge P(Z | H_tD) 
                    ## which is a sum of risks from discharge until time point K (end of follow-up).
                # If binorm(P(Z))=1, then set all-cause mortality Y=1 at that time point. 
                # Kick that ID out which stops simulation.
            
            # ids for which A=1 at t=0. Works on pool dataframe which has entire trajectory so far. 
            if True:
                ids_with_A1_t0 = pool.loc[pool['A'] == 1, id].unique()    # ids with A=1 at t=0

                pool_with_A1_t0 = pool[pool[id].isin(ids_with_A1_t0)]     # pool only with A=1 at t=0
                pool = pool[~pool[id].isin(ids_with_A1_t0)]  # Remove ids with A=1, t=0 from pool

                # Predict Z only on the discharge rows at the current time t
                pool_with_A1_t0_t = pool_with_A1_t0.loc[pool_with_A1_t0[time_name] == t].copy()

                # Compute P(post-discharge hazard), i.e., P(Z).
                    #pre_z = outcome_fit.predict(pool_with_A1_t0_t)
                    #Z_A1_t0 = pre_z.apply(binorm_sample).to_numpy()
                #Z_A1_t0 = simulate_post_discharge_Z_from_discharge_rows(pool_with_A1_t0_t, z_outcome_fit, zmodel, zmodel_predict_custom, simul_rng)
                pool_with_A1_t0_t_tD = pool_with_A1_t0_t.rename(columns={time_name: "tD"})
                Z_A1_t0 = simulate_postdischarge_variable_hazard(pool_with_A1_t0_t_tD, 
                                                                 zmodel,
                                                                 zmodel_predict_custom, 
                                                                 z_outcome_fit, 
                                                                 simul_rng)
                death_by_K = Z_A1_t0['death_by_K']

                if outcome_type == 'binary_eof':
                    pool_with_A1_t0.loc[pool_with_A1_t0[time_name] == t, 'Z_hat'] = death_by_K # Outcome Z 
                    pool_with_A1_t0["I_hat"] = np.nan #In-ICU death is NA for those discharged.
                    pool_with_A1_t0["I_event_t"] = np.nan  # No I-bin event applies to discharged stays
                    pool['Z_hat'] = np.nan # For the rest of the pool

                '''if outcome_type == 'continuous_eof':
                    pool_with_A1_t0.loc[pool_with_A1_t0[time_name] == t, 'Ey'] = pre_y
                    pool['Ey'] = np.nan'''

                final_df_list.append(pool_with_A1_t0)  # store them in global list for concatenation at the end
                
                # If A=0,
                # Predict risk of in-icu death with P(I)
                # If binorm(P(I))=1, then set Y=1 for that ID. 
                # Kick that ID out which stop simulation for that ID.
                # Else if binorm(P(I))=0, then continue simulation for that ID (nothing to do here).
                
                ids_with_A0_t0 = pool.loc[pool['A'] == 0, id].unique()    # ids with A=0 at t=0
                pool_with_A0_t0 = pool[pool[id].isin(ids_with_A0_t0)]     # pool only with A=0 at t=0
                
                pool_with_A0_t0.sort_values([id, time_name], ascending=[True, True], inplace=True)
                pool_with_A0_t0_t = pool_with_A0_t0[pool_with_A0_t0[time_name] == t].copy() # Redundant for t0 since only 1 time point.

                # Compute P(in-icu mortality).
                # If a user-supplied I_model_predict_custom is provided (e.g., LightGBM),
                # use it to obtain probabilities. Otherwise fall back to the GLM
                # `.predict(...)` API, which returns probabilities directly for a
                # binomial GLM.
                if I_model_predict_custom is not None:
                    pre_i = I_model_predict_custom(
                        I_model=I_model, new_df=pool_with_A0_t0_t, fit=I_fit
                    )
                else:
                    pre_i = I_fit.predict(pool_with_A0_t0_t)
                #pre_i = pd.to_numeric(pre_i, errors="coerce").clip(1e-12, 1-1e-12).fillna(0.0)

                I_t0 = pre_i.apply(binorm_sample, simul_rng=simul_rng)

                # Persist the bin-level event into `pool` BEFORE slicing
                # `pool_with_I1_t0` from it. Without this, the bin-level
                # I_event_t lives only on the local `pool_with_A0_t0_t` copy
                # and is lost when that DataFrame goes out of scope.
                #
                # We cannot use `pool.loc[pool_with_A0_t0_t.index, ...]` because
                # `pool`'s integer index can have duplicate labels after the
                # `pd.concat(pool, new_df)` operations earlier in the simulator
                # do not pass `ignore_index=True`. Duplicate-label `.loc`
                # selection expands the LHS to more rows than the RHS and pandas
                # raises "Must have equal len keys and value".
                # Instead, write back via a stay-id-keyed map combined with a
                # boolean mask on (id, time_name) — both are unambiguous and
                # robust to index duplication.
                _id_to_I_t0 = pd.Series(
                    I_t0.values,
                    index=pool_with_A0_t0_t[id].values,
                )
                _mask_t0 = (pool[time_name] == t) & (pool[id].isin(ids_with_A0_t0))
                assert _mask_t0.sum() == len(I_t0), (
                    f"t=0 I-write-back mask size {_mask_t0.sum()} != len(I_t0) {len(I_t0)}"
                )
                _mapped_t0 = pool.loc[_mask_t0, id].map(_id_to_I_t0).values
                pool.loc[_mask_t0, 'I_event_t'] = _mapped_t0
                pool.loc[_mask_t0, 'I_hat']     = _mapped_t0

                # Keep the local copy in sync for the immediate ids_with_I1_t0 lookup.
                pool_with_A0_t0_t['I_event_t'] = I_t0  # bin-level event indicator
                pool_with_A0_t0_t['I_hat'] = I_t0      # bin-level marker, will be overwritten if stay dies

                ids_with_I1_t0 = pool_with_A0_t0_t.loc[pool_with_A0_t0_t['I_hat'] == 1, id].unique()    # ids with I=1 at t=0 (A=0 as well)

                pool_with_I1_t0 = pool[pool[id].isin(ids_with_I1_t0)]     # pool only with I=1, at t=0 (A=0 as well)
                pool = pool[~pool[id].isin(ids_with_I1_t0)]  # Remove ids with I=1, t=0 (A=0) from pool

                # Store I=1 as Y=1 for IDs in pool_with_I1_t0. These are IDs with in-icu death.
                # Stay-level death indicator: I_hat = 1 on every row of the stay.
                # I_event_t stays as it was set during sampling — only 1 at the bin where
                # death was drawn, 0 (or NaN at unobserved bins) elsewhere.
                pool_with_I1_t0['I_hat'] = 1
                pool_with_I1_t0['Z_hat'] = np.nan # For this cohort which suffered in-icu death, Z is NA.
                pool['Z_hat'] = np.nan # For the rest of the pool

                final_df_list.append(pool_with_I1_t0)  # store them in global list for concatenation at the end

        else:
            new_df = pool[pool[time_name] == t-1].copy()
            new_df[time_name] = t

            '''if covtypes is not None:
                if 'categorical time' in covtypes:
                    new_df.loc[new_df[time_name] == t, time_name + '_f'] = new_df[time_name].apply(categorical_func, time_thresholds=time_thresholds)
                if 'square time' in covtypes:
                    new_df.loc[new_df[time_name] == t, 'square_' + time_name] = new_df[time_name] * new_df[time_name]'''
            pool = pd.concat([pool, new_df])
            pool.sort_values([id, time_name], ascending=[True, True], inplace=True)

            if covnames is not None:
                update_precoded_history(pool, covnames, cov_hist, covtypes, time_name, id, below_zero_indicator,
                                      baselags, ts_visit_names)
                if custom_histvars is not None:
                    update_custom_history(pool, custom_histvars, custom_histories, time_name, t, id)
                new_df = pool[pool[time_name] == t].copy()

                for k, cov in enumerate(covnames):
                    if covmodels[k] != 'NA':
                        if visit_names and cov in visit_names: ### assign values for visit indicator
                            estimated_mean = covariate_fits[cov].predict(new_df)
                            prediction = estimated_mean.apply(binorm_sample, simul_rng=simul_rng)
                            max_visit = max_visits[visit_names.index(cov)]
                            ts_visit_name = ts_visit_names[visit_names.index(cov)]
                            new_df[cov] = np.where(new_df['lag1_{0}'.format(ts_visit_name)] < max_visit, prediction, 1)
                            new_df[ts_visit_name] = np.where(new_df[cov] == 0, new_df[ts_visit_name] + 1, 0)

                        elif covtypes[k] == 'binary':
                            estimated_mean = covariate_fits[cov].predict(new_df)
                            prediction = estimated_mean.apply(binorm_sample, simul_rng=simul_rng) # N values are generated
                            new_df[cov] = prediction

                        elif covtypes[k] == 'normal':
                            #estimated_mean = covariate_fits[cov].predict(new_df)
                            #prediction = estimated_mean.apply(norm_sample, rmse=rmses[cov], simul_rng=simul_rng)
                            #if sim_trunc:
                            #    prediction = np.where(prediction < bounds[cov][0], bounds[cov][0], prediction)
                            #    prediction = np.where(prediction > bounds[cov][1], bounds[cov][1], prediction)
                            #new_df[cov] = prediction
                            estimated_mean = covariate_fits[cov].predict(new_df)
                            prediction = estimated_mean.apply(norm_sample, rmse=rmses[cov], simul_rng=simul_rng)
                            prediction = apply_bounds(prediction, cov, bounds, integer=False)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'categorical':
                            predict_probs = covariate_fits[cov].predict(new_df)
                            predict_probs = np.asarray(predict_probs)
                            categories = pd.Categorical(obs_data[cov]).categories
                            predict_index = [simul_rng.choice(len(probs), p=probs) for probs in predict_probs]
                            prediction = [categories[i] for i in predict_index]
                            new_df[cov] = prediction

                        elif covtypes[k] == 'bounded normal':
                            estimated_mean = covariate_fits[cov].predict(new_df)
                            prediction = estimated_mean.apply(norm_sample, rmse=rmses[cov], simul_rng=simul_rng)
                            prediction = prediction.apply(lambda x: x * (bounds[cov][1] - bounds[cov][0]) + bounds[cov][0])
                            #prediction = np.where(prediction < bounds[cov][0], bounds[cov][0], prediction)
                            #prediction = np.where(prediction > bounds[cov][1], bounds[cov][1], prediction)
                            prediction = apply_bounds(prediction, cov, bounds, integer=False)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'zero-inflated normal':
                            estimated_indicator_mean = covariate_fits[cov][0].predict(new_df)
                            indicator = estimated_indicator_mean.apply(binorm_sample, simul_rng=simul_rng)
                            estimated_mean = covariate_fits[cov][1].predict(new_df)
                            prediction = estimated_mean.apply(norm_sample, rmse=rmses[cov], simul_rng=simul_rng)
                            nonzero_predict = prediction.apply(lambda x: math.exp(x))
                            prediction = indicator * nonzero_predict
                            prediction = np.where((prediction < bounds[cov][0]) & (indicator == 1), bounds[cov][0], prediction)
                            prediction = np.where((prediction > bounds[cov][1]) & (indicator == 1), bounds[cov][1], prediction)
                            new_df[cov] = prediction

                            # Only clip positive simulated values; preserve zeros exactly
                            #if cov in bounds:
                            #    lower, upper = bounds[cov]
                            #    prediction = np.where((prediction < lower) & (indicator == 1), lower, prediction)
                            #    prediction = np.where((prediction > upper) & (indicator == 1), upper, prediction)
                            #new_df[cov] = prediction

                        elif covtypes[k] == 'truncated normal':
                            fit_coefficients = covariate_fits[cov]
                            _, covmodel = re.split('~', covmodels[k].replace(' ', ''))
                            var_names = re.split('\+', covmodel)
                            new_data = np.concatenate((np.ones((new_df.shape[0], 1)), new_df[var_names].to_numpy()), axis=1)
                            estimated_mean = np.dot(new_data, fit_coefficients['x'][:-1])

                            if trunc_params[k][1] == 'left':
                                trunc_bounds = [trunc_params[k][0], float('inf')]
                            else:
                                trunc_bounds = [-float('inf'), trunc_params[k][0]]

                            prediction = pd.Series(estimated_mean).apply(truc_sample, rmse=rmses[cov], a=trunc_bounds[0],
                                                                     b=trunc_bounds[1])
                            prediction = apply_bounds(prediction, cov, bounds, integer=False)
                            #prediction = np.where(prediction < bounds[cov][0], bounds[cov][0], prediction)
                            #prediction = np.where(prediction > bounds[cov][1], bounds[cov][1], prediction)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'absorbing':
                            predict_prob = covariate_fits[cov].predict(new_df)
                            prediction = predict_prob.apply(binorm_sample, simul_rng=simul_rng)
                            prediction = np.where(pool.loc[pool[time_name] == t - 1, cov] == 0, prediction, 1)
                            new_df[cov] = prediction
                        
                        elif covtypes[k] == 'poisson':
                            estimated_mean = covariate_fits[cov].predict(new_df)
                            prediction = estimated_mean.apply(poisson_sample, simul_rng=simul_rng)
                            prediction = apply_bounds(prediction, cov, bounds, integer=True)
                            new_df[cov] = prediction

                        elif covtypes[k] == 'custom':
                            pred_func = covpredict_custom[k]
                            prediction = pred_func(covmodel=covmodels[k], new_df=new_df, fit=covariate_fits[cov])
                            # Apply bounds if this custom covariate has stored bounds
                            if cov in bounds:
                                is_integer = pd.api.types.is_integer_dtype(obs_data[cov].dropna())
                                prediction = apply_bounds(prediction, cov, bounds, integer=is_integer)
                            new_df[cov] = prediction

                        if visit_covs and cov in visit_covs: ### assign visited covariate the model output value or its lagged value based on visit indicator
                            visit_name = visit_names[visit_covs.index(cov)]
                            new_df[cov] = np.where(new_df[visit_name] == 0, new_df['lag1_{0}'.format(cov)], new_df[cov])

                    if restrictions is not None:
                        restrictcovs = [restrictions[i][0] for i in range(len(restrictions))]
                        if cov in restrictcovs:
                            index = restrictcovs.index(cov)
                            restriction = restrictions[index]
                            conditions = restriction[1]
                            masks = []
                            for cond_var, condition in conditions.items():
                                mask = new_df[cond_var].apply(condition)
                                masks.append(mask)
                            restrict_mask = reduce(operator.and_, masks)
                            if isinstance(restriction[2], types.FunctionType):
                                assigned_values = restriction[2](new_df=new_df, pool=pool, time_name=time_name, t=t)
                                new_df.loc[~restrict_mask, cov] = assigned_values
                            else:
                                new_df.loc[~restrict_mask, cov] = restriction[2]

                    pool.loc[pool[time_name] == t] = new_df

                    if len(cov_hist[cov]['cumavg']) > 0:
                        update_precoded_history(pool, covnames, cov_hist, covtypes, time_name, id, below_zero_indicator, baselags, ts_visit_names)
                    if custom_histvars is not None and cov in custom_histvars:
                        update_custom_history(pool, custom_histvars, custom_histories, time_name, t, id)
                    new_df = pool[pool[time_name] == t].copy()

            intervention_func(new_df=new_df, pool=pool, intervention=intervention, time_name=time_name, t=t)

            
            # Compulsory discharge at "time_points" for static.
            if (intervention_function == static) and (t == time_points - 1):
                new_df.loc[new_df[time_name] == t, 'A'] = 1
            
            pool.loc[pool[time_name] == t] = new_df
            if covnames is not None:
                update_precoded_history(pool, covnames, cov_hist, covtypes, time_name, id, below_zero_indicator,
                                      baselags, ts_visit_names)
                if custom_histvars is not None:
                    update_custom_history(pool, custom_histvars, custom_histories, time_name, t, id)
            new_df = pool[pool[time_name] == t].copy()

            '''if competing and not compevent_cens:
                params_D = re.split('[~|\+]', compevent_model.replace(' ', ''))
                prob_D = compevent_fit.predict(new_df[params_D])
                new_df['prob_D'] = prob_D

                if compevent_restrictions is not None:
                    for restriction in compevent_restrictions:
                        conditions = restriction[0]
                        masks = []
                        for cond_var, condition in conditions.items():
                            mask = new_df[cond_var].apply(condition)
                            masks.append(mask)
                        comp_restrict_mask = reduce(operator.and_, masks)
                        new_df.loc[~comp_restrict_mask, 'prob_D'] = restriction[1]

                new_df[compevent_name] = new_df['prob_D'].apply(binorm_sample)'''

            '''if ymodel_predict_custom is not None:
                pre_y = ymodel_predict_custom(ymodel=ymodel, new_df=new_df, fit=outcome_fit)
            else:
                pre_y = outcome_fit.predict(new_df)'''

            '''if outcome_type == 'survival':
                new_df['prob1'] = pre_y

                if yrestrictions is not None:
                    for restriction in yrestrictions:
                        conditions = restriction[0]
                        masks = []
                        for cond_var, condition in conditions.items():
                            mask = new_df[cond_var].apply(condition)
                            masks.append(mask)
                        restrict_mask = reduce(operator.and_, masks)
                        new_df.loc[~restrict_mask, 'prob1'] = restriction[1]

                new_df['prob0'] = 1 - new_df['prob1']
                new_df[outcome_name] = new_df['prob1'].apply(binorm_sample)'''

            # Changes for NC
            '''if intervention_function == static:
                if outcome_type == 'binary_eof':
                    new_df['Py'] = 'NA' if t < time_points - 1 else pre_y
                if outcome_type == 'continuous_eof':
                    new_df['Ey'] = 'NA' if t < time_points - 1 else pre_y'''

            '''if competing and not compevent_cens:
                new_df.loc[new_df[compevent_name] == 1, outcome_name] = pd.NA'''

            pool.loc[pool[time_name] == t] = new_df

            '''# Changes for NC to stop simulation once A=1 at any t
            if intervention_function != static:
                ids_with_A1_t = pool.loc[pool['A'] == 1, id].unique()  # ids with A=1 at t

                pool_with_A1_t = pool[pool[id].isin(ids_with_A1_t)]  # pool only with A=1 at t
                pool = pool[~pool[id].isin(ids_with_A1_t)]  # Remove ids with A=1, t from pool

                if outcome_type == 'binary_eof':
                    pool_with_A1_t.loc[pool_with_A1_t[time_name] < t, 'Py'] = np.nan
                    pool_with_A1_t.loc[pool_with_A1_t[time_name] == t, 'Py'] = pre_y
                    pool['Py'] = np.nan
                if outcome_type == 'continuous_eof':
                    pool_with_A1_t.loc[pool_with_A1_t[time_name] < t, 'Ey'] = np.nan
                    pool_with_A1_t.loc[pool_with_A1_t[time_name] == t, 'Ey'] = pre_y
                    pool['Ey'] = np.nan'''
            
            # If A=1,
                # Stop simulation for that ID.
                # Predict post-discharge risk P(Z) until time point K, given history until discharge P(Z | H_tD) 
                    ## which is a sum of risks from discharge until time point K (end of follow-up).
                # If binorm(P(Z))=1, then set all-cause mortality Y=1 at that time point. 
                # Kick that ID out which stops simulation.
            
            # ids for which A=1 at t=0. Works on pool dataframe which has entire trajectory so far. 
            if True:
                ids_with_A1_t = pool.loc[pool['A'] == 1, id].unique()    # ids with A=1 at t

                pool_with_A1_t = pool[pool[id].isin(ids_with_A1_t)]     # pool only with A=1 at t
                pool = pool[~pool[id].isin(ids_with_A1_t)]  # Remove ids with A=1, t from pool

                # Predict Z only on the discharge rows at the current time t
                pool_with_A1_t_t = pool_with_A1_t.loc[pool_with_A1_t[time_name] == t].copy()
                
                # Compute P(post-discharge mortality), i.e., P(Z). For now only computing outcome at discharge. Expand for hazard until K later.
                #pre_z = outcome_fit.predict(pool_with_A1_t_t)
                #Z_A1_t = pre_z.apply(binorm_sample).to_numpy()
                #Z_A1_t = simulate_post_discharge_Z_from_discharge_rows(pool_with_A1_t_t, z_outcome_fit, zmodel, zmodel_predict_custom, simul_rng)
                pool_with_A1_t_t_tD = pool_with_A1_t_t.rename(columns={time_name: "tD"})
                Z_A1_t = simulate_postdischarge_variable_hazard(pool_with_A1_t_t_tD, 
                                                                zmodel, 
                                                                zmodel_predict_custom, 
                                                                z_outcome_fit, 
                                                                simul_rng)
                death_by_K = Z_A1_t['death_by_K']

                if outcome_type == 'binary_eof':
                    pool_with_A1_t.loc[pool_with_A1_t[time_name] == t, 'Z_hat'] = death_by_K # Outcome Z. t is the time of discharge.
                    pool_with_A1_t["I_hat"] = np.nan #In-ICU death is NA for those discharged.
                    pool_with_A1_t["I_event_t"] = np.nan  # No I-bin event applies to discharged stays
                    pool['Z_hat'] = np.nan # For the rest of the pool

                '''if outcome_type == 'continuous_eof':
                    pool_with_A1_t0.loc[pool_with_A1_t0[time_name] == t, 'Ey'] = pre_y
                    pool['Ey'] = np.nan'''

                final_df_list.append(pool_with_A1_t)  # store them in global list for concatenation at the end
                
                # If A=0,
                # Predict risk of in-icu death with P(I)
                # If binorm(P(I))=1, then set Y=1 for that ID. 
                # Kick that ID out which stop simulation for that ID.
                # Else if binorm(P(I))=0, then continue simulation for that ID (nothing to do here).
                
                ids_with_A0_t = pool.loc[pool['A'] == 0, id].unique()    # ids with A=0 at t
                pool_with_A0_t = pool[pool[id].isin(ids_with_A0_t)]     # pool only with A=0 at t
                
                pool_with_A0_t.sort_values([id, time_name], ascending=[True, True], inplace=True) # Sort so that we pick lates t where A=0 at next step.
                pool_with_A0_t_t = pool_with_A0_t[pool_with_A0_t[time_name] == t].copy() # Pick current time point where A=0.

                # Compute P(in-icu mortality).
                # See note at the t=0 block: use I_model_predict_custom if provided.
                if I_model_predict_custom is not None:
                    pre_i = I_model_predict_custom(
                        I_model=I_model, new_df=pool_with_A0_t_t, fit=I_fit
                    )
                else:
                    pre_i = I_fit.predict(pool_with_A0_t_t)
                #pre_i = pd.to_numeric(pre_i, errors="coerce").clip(1e-12, 1-1e-12).fillna(0.0)

                I_t = pre_i.apply(binorm_sample, simul_rng=simul_rng)

                # Persist the bin-level event into `pool` BEFORE slicing
                # `pool_with_I1_t` from it. See t=0 block for the same fix
                # and rationale (pool's integer index has duplicate labels due
                # to earlier `pd.concat` without `ignore_index=True`).
                _id_to_I_t = pd.Series(
                    I_t.values,
                    index=pool_with_A0_t_t[id].values,
                )
                _mask_t = (pool[time_name] == t) & (pool[id].isin(ids_with_A0_t))
                assert _mask_t.sum() == len(I_t), (
                    f"t={t} I-write-back mask size {_mask_t.sum()} != len(I_t) {len(I_t)}"
                )
                _mapped_t = pool.loc[_mask_t, id].map(_id_to_I_t).values
                pool.loc[_mask_t, 'I_event_t'] = _mapped_t
                pool.loc[_mask_t, 'I_hat']     = _mapped_t

                # Keep the local copy in sync for the immediate ids_with_I1_t lookup.
                pool_with_A0_t_t['I_event_t'] = I_t  # bin-level event indicator
                pool_with_A0_t_t['I_hat'] = I_t      # bin-level marker, will be overwritten if stay dies

                ids_with_I1_t = pool_with_A0_t_t.loc[pool_with_A0_t_t['I_hat'] == 1, id].unique()    # ids with I=1 at t (A=0 as well)

                pool_with_I1_t = pool[pool[id].isin(ids_with_I1_t)]     # pool only with I=1, at t (A=0 as well)
                pool = pool[~pool[id].isin(ids_with_I1_t)]  # Remove ids with I=1, t (A=0) from pool

                # Store I=1 as Y=1 for IDs in pool_with_I1_t0. These are IDs with in-icu death.
                # Stay-level death indicator: I_hat = 1 on every row of the stay.
                # I_event_t stays as it was set during sampling — only 1 at the bin where
                # death was drawn, 0 (or NaN at unobserved bins) elsewhere.
                pool_with_I1_t['I_hat'] = 1
                pool_with_I1_t['Z_hat'] = np.nan # For this cohort which suffered in-icu death, Z is NA.
                pool['Z_hat'] = np.nan # For the rest of the pool

                final_df_list.append(pool_with_I1_t)  # store them in global list for concatenation at the end


    # Changes for NC and dynamic
    # Concatenate all dataframes at different t into a single DataFrame pool

    # pool is leftover stays at end-of-follow-up: enforce Case 4 terminal outcomes
    #print(pool[id].nunique(), 'unique ids in pool after simulation')
    
    # 1) Set I_hat = 0 for all rows
    pool["I_hat"] = 0      # administratively-censored survivors: stay-level I=0
    pool["I_event_t"] = 0  # bin-level event=0 on all rows of these stays

    # 2) Set Z_hat = NaN for all rows
    pool["Z_hat"] = np.nan

    # 3) Set Y_hat = NaN for all rows, then set last row per stay_id to 0. The below gets overwritten with nan. change later! for now. okay!
    pool["Y_hat"] = np.nan
    last_idx = pool.sort_values([id, time_name]).groupby(id, sort=False).tail(1).index
    pool.loc[last_idx, "Y_hat"] = 0

    # Concatenate all final frames.
    # Note: each frame in `final_df_list` may have different presence/NaN-ness
    # for I_hat / Z_hat / Y_hat (discharged stays have Z_hat populated, in-ICU
    # death stays have I_hat=1, residual stays have I_hat=0). Pandas 2.x emits
    # a FutureWarning that pandas 3.x will change concat dtype inference for
    # empty / all-NA columns. We pre-harmonize the dtypes of the outcome
    # bookkeeping columns to avoid both the warning and any version-dependent
    # dtype drift in downstream consumers.
    final_df_list.append(pool)
    for _df in final_df_list:
        for _col, _dtype in (("I_hat", "float64"), ("Z_hat", "float64"), ("Y_hat", "float64"),
                             ("I_event_t", "float64")):
            if _col in _df.columns:
                _df[_col] = _df[_col].astype(_dtype)
            else:
                _df[_col] = pd.Series(np.nan, index=_df.index, dtype=_dtype)
    pool = pd.concat(final_df_list, ignore_index=True)
    pool.sort_values([id, time_name], ascending=[True, True], inplace=True)
    pool.reset_index(drop=True, inplace=True)
    #############################################################

    pool = pool[pool[time_name] >= 0]

    '''if outcome_type == 'survival':
        if competing and not compevent_cens:
            pool['cumprob0'] = pool.groupby([id])['prob0'].cumprod()
            pool['prob_D0'] = 1 - pool['prob_D']
            pool['cumprob_D0'] = pool.groupby([id])['prob_D0'].cumprod()
            pool['prodp1'] = np.where(pool[time_name] > 0,pool.groupby([id])['cumprob0'].shift(1) * pool['cumprob_D0'].shift(1)
                                      * pool['prob1'] * (1 - pool['prob_D']),
                                      pool['prob1'] * (1 - pool['prob_D']))
            pool['risk'] = pool.groupby([id])['prodp1'].cumsum()
            pool['survival'] = 1 - pool['risk']
            g_result = pool.groupby(time_name, group_keys=False)['risk'].mean().tolist()
        else:
            pool['cumprod0'] = pool.groupby([id])['prob0'].cumprod()
            pool['prodp1'] = np.where(pool[time_name] > 0, pool.groupby([id])['cumprod0'].shift(1) * pool['prob1'], pool['prob1'])
            pool['risk'] = pool.groupby([id])['prodp1'].cumsum()
            pool['survival'] = 1 - pool['risk']
            g_result = pool.groupby(time_name, group_keys=False)['risk'].mean().tolist()

    if outcome_type == 'continuous_eof':
        #g_result = pool.loc[pool[time_name] == time_points - 1]['Ey'].mean()
        g_result = pool.groupby(id).tail(1)['Ey'].mean()'''

    if outcome_type == 'binary_eof':
        #g_result = pool.loc[pool[time_name] == time_points - 1]['Py'].mean()
        #final_result_stay = pool.groupby(id).agg(D=("I_hat","max"), A=("A","max"), Z=("Z_hat","max"))
        #Y = ((final_result_stay["D"] == 1) | ((final_result_stay["A"] == 1) & (final_result_stay["Z"] == 1))).astype(int)
        #g_result = Y.mean()

        # Ensure sorted
        pool = pool.sort_values([id, time_name])

        # Initialize Y_hat as NaN everywhere
        pool["Y_hat"] = np.nan

        # Identify last row per stay
        last_idx = pool.groupby(id, sort=False).tail(1).index

        # Extract needed columns on last rows
        A_last = pool.loc[last_idx, "A"]
        I_last = pool.loc[last_idx, "I_hat"]
        Z_last = pool.loc[last_idx, "Z_hat"]

        # Compute Y_hat according to your rule (without filling NaNs globally)
        Y_last = (
            (I_last == 1) |
            ((A_last == 1) & (Z_last == 1))
        ).astype(int)

        # Assign only to terminal rows
        pool.loc[last_idx, "Y_hat"] = Y_last
        g_result = pool.loc[last_idx, "Y_hat"].mean()

    return {'g_result': g_result, 'pool': pool}

