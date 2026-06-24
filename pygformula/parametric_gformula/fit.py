import sys
import numpy as np
import pandas as pd
import math
import re
import statsmodels.api as sm
import statsmodels.formula.api as smf
from pytruncreg import truncreg


def fit_covariate_model(covmodels, covnames, covtypes, covfits_custom, time_name, obs_data, return_fits,
                        trunc_params=None, visit_names=None, max_visits=None, ts_visit_names=None, visit_covs=None,
                        restrictions=None):
    """
    This is a function to fit parametric models for all time-varying covariates.

    Parameters
    ----------
    covmodels: List
        A list of strings, where each string is the model statement of the time-varying covariate. The list
        must be the same length as covnames and in the same order. If a model is not required for a certain covariate,
        it should be set to 'NA' at that index.

    covnames: List
        A list of strings specifying the names of the time-varying covariates in obs_data.

    covtypes: List
        A list of strings specifying the “type” of each time-varying covariate included in covnames.
        The supported types: "binary", "normal", "categorical", "bounded normal", "zero-inflated normal",
        "truncated normal", "absorbing", "categorical time", "square time" and "custom". The list must be the same length
        as covnames and in the same order.

    covfits_custom: List
        A list, each element could be 'NA' or a user-specified fit function. The non-NA value is set
        for the covariates with custom type. The 'NA' value is set for other covariates. The list must be the
        same length as covnames and in the same order.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    obs_data: DataFrame
        Observed data or resampled data used to estimate the parameters of the covariate models.

    return_fits: Bool
        A boolean value indicating whether to get the coefficients, standard errors, variance-covariance matrices of the
        fitted model.

    trunc_params:  List
        A list, each element could be 'NA' or a two-element list. If not 'NA', the first element specifies the truncated
        value and the second element specifies the truncated direction (‘left’ or ‘right’). The non-NA value is set
        for the truncated normal covariates. The 'NA' value is set for other covariates. The list should be the same
        length as covnames and in the same order.

    visit_names: List
        A list, each of which is a string specifying the covariate name of a visit process.

    max_visits: List
        A list of integers, each integer indicates the maximum number of consecutive missed visits for one covariate that
        has a visit process.

    ts_visit_names: List
        A list of strings, each of which indicates the number of consecutive missed visits for one covariate before an
        individual is censored.

    visit_covs: List
        A list of strings, each of which specifying the name of a covariate whose modeling depends on the visit process.

    restrictions: List
        List of lists. Each inner list contains its first entry the covariate name of that its deterministic knowledge
        is known; its second entry is a dictionary whose key is the conditions which should be True when the covariate
        is modeled, the third entry is the value that is set to the covariate during simulation when the conditions
        in the second entry are not True.

    Returns
    -------
    covariate_fits: List
         A list that contains the fitted model for all time-varying covariates.

    bounds: List
        A list that contains the bound for all time-varying covariates in the obs_data.

    rmses: List
        A list that contains the root mean square errors (rmses) of all the fitted models.

    model_coeffs: List
        A list that contains the parameter estimates of all the fitted models.

    model_stderrs: List
        A list that contains the standard errors of the parameter estimates of all the fitted models.

    model_vcovs: List
        A list that contains the variance-covariance matrices of the parameter estimates of all the fitted models.

    model_fits_summary: List
        A list that contains the summary information of all the fitted models.

    """

    covariate_fits = {}
    bounds = {}
    rmses = {}
    model_coeffs = {}
    model_stderrs = {}
    model_vcovs = {}
    model_fits_summary = {}

    sub_data = obs_data[obs_data[time_name] > 0]

    for k, cov in enumerate(covnames):
        print('$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$', cov)
        if covmodels[k] != 'NA':
            if visit_names and cov in visit_names:
                max_visit = max_visits[visit_names.index(cov)]
                ts_visit_name = ts_visit_names[visit_names.index(cov)]
                fit_data = sub_data[sub_data['lag1_{0}'.format(ts_visit_name)] < max_visit].copy()

            elif visit_covs and cov in visit_covs:
                visit_cov_name = visit_names[visit_covs.index(cov)]
                fit_data = sub_data[sub_data[visit_cov_name] == 1].copy()

            else:
                fit_data = sub_data.copy()

            if restrictions is not None:
               restrictcovs = [restrictions[0] for i in range(len(restrictions))]
               if cov in restrictcovs:
                   index = restrictcovs.index(cov)
                   conditions = restrictions[index][1]
                   for cond_var, condition in conditions.items():
                       mask = fit_data[cond_var].apply(condition)
                       fit_data = fit_data[mask]

            # PATCH: LightGBM (custom covtype) handles NaN natively and learns from
            # the missingness pattern. The original formula-wide dropna eliminates
            # rows with NaN in ANY predictor, which for a 22-predictor formula on
            # MIMIC-IV ICU data leaves <0.1% of rows — a small, severely biased
            # subset of the most heavily-monitored (sickest) patients. This biased
            # the A-model into "low discharge probability after t=0" and caused
            # the per-t A-hazard collapse seen in the natural-course diagnostics.
            #
            # For GLM-family covtypes (which statsmodels cannot fit with NaN
            # predictors), keep the original behaviour. For custom covtypes, drop
            # only rows with NaN in the *outcome* — predictor NaN is fine because
            # LightGBM handles it natively as a learnable signal.
            if covtypes[k] == 'custom':
                fit_data = fit_data.dropna(subset=[cov])
            else:
                # exclude rows that contains NA values of the predictors in fit_data
                predictors = covmodels[k].split('~')[1].strip().split(' + ')
                all_vars = predictors + [cov]
                all_vars = [item[2:-1] if item.startswith('C(') and item.endswith(')') else item for item in all_vars]
                fit_data = fit_data[all_vars].dropna()

            if covtypes[k] == 'binary':
                #fit_data.to_parquet("fit_data_{0}.parquet".format(cov))
                bounds[cov] = [0, 1]
                fit = smf.glm(covmodels[k], data=fit_data, family=sm.families.Binomial()).fit()
                rmse = np.sqrt(np.mean((fit.predict() - fit_data[cov]) ** 2))
                covariate_fits[cov] = fit
                rmses[cov] = rmse
                if return_fits:
                    model_coeffs[cov] = fit.params
                    model_stderrs[cov] = fit.bse
                    model_vcovs[cov] = fit.cov_params()
                    model_fits_summary[cov] = fit.summary()

            elif covtypes[k] == 'normal':
                min_cov = fit_data[cov].min()
                max_cov = fit_data[cov].max()
                bound = [min_cov, max_cov]
                #fit_data.to_parquet("fit_data_{0}.parquet".format(cov))
                fit = smf.glm(covmodels[k], data=fit_data, family=sm.families.Gaussian()).fit()
                #print(fit_data)
                #print(fit.predict())
                #print(len(fit.predict()), len(fit_data[cov]))
                #exit(0)
                rmse = np.sqrt(np.mean((fit.predict() - fit_data[cov]) ** 2))
                bounds[cov] = bound
                covariate_fits[cov] = fit
                rmses[cov] = rmse
                if return_fits:
                    model_coeffs[cov] = fit.params
                    model_stderrs[cov] = fit.bse
                    model_vcovs[cov] = fit.cov_params()
                    model_fits_summary[cov] = fit.summary()

            elif covtypes[k] == 'categorical':
                #fit_data.to_parquet("fit_data_{0}.parquet".format(cov))
                fit_data[cov] = pd.Categorical(fit_data[cov]).codes
                fit = smf.mnlogit(covmodels[k], data=fit_data).fit()
                covariate_fits[cov] = fit
                if return_fits:
                    model_coeffs[cov] = fit.params
                    model_stderrs[cov] = fit.bse
                    model_vcovs[cov] = fit.cov_params()
                    model_fits_summary[cov] = fit.summary()

            elif covtypes[k] == 'bounded normal':
                min_cov = fit_data[cov].min()
                max_cov = fit_data[cov].max()
                bound = [min_cov, max_cov]
                fit_data[cov] = fit_data[cov].apply(lambda x: (x - min_cov) / (max_cov - min_cov))
                
                #fit_data.to_parquet("fit_data_{0}.parquet".format(cov))
                
                fit = smf.glm(covmodels[k], data=fit_data, family=sm.families.Gaussian()).fit()
                rmse = np.sqrt(np.mean((fit.predict() - fit_data[cov]) ** 2))
                bounds[cov] = bound
                covariate_fits[cov] = fit
                rmses[cov] = rmse
                if return_fits:
                    model_coeffs[cov] = fit.params
                    model_stderrs[cov] = fit.bse
                    model_vcovs[cov] = fit.cov_params()
                    model_fits_summary[cov] = fit.summary()

            elif covtypes[k] == 'zero-inflated normal':
                min_cov = fit_data[fit_data[cov] != 0][cov].min()
                max_cov = fit_data[fit_data[cov] != 0][cov].max()
                bound = [min_cov, max_cov]
                fit_data['I_{0}'.format(cov)] = fit_data[cov].apply(lambda x: 1 if x != 0 else 0)
                fit_data['log_{0}'.format(cov)] = fit_data[cov].apply(lambda x: math.log(x) if x != 0 else 0)
                _, fit_model_name = re.split('~', covmodels[k].replace(' ', ''))
                indicator_model = "~".join(['I_{0}'.format(cov), fit_model_name])
                indicator_fit = smf.glm(indicator_model, data=fit_data, family=sm.families.Binomial()).fit()
                log_nonzero_model = "~".join(['log_{0}'.format(cov), fit_model_name])
                
                #fit_data.to_parquet("fit_data_{0}.parquet".format(cov))

                non_zero_fit = smf.glm(log_nonzero_model, data=fit_data[fit_data[cov] != 0],
                                       family=sm.families.Gaussian()).fit()
                rmse = np.sqrt(np.mean((non_zero_fit.predict() - fit_data[fit_data[cov] != 0]['log_{0}'.format(cov)]) ** 2))
                bounds[cov] = bound
                covariate_fits[cov] = [indicator_fit, non_zero_fit]
                rmses[cov] = rmse
                if return_fits:
                    model_coeffs[cov] = [indicator_fit.params, non_zero_fit.params]
                    model_stderrs[cov] = [indicator_fit.bse, non_zero_fit.bse]
                    model_vcovs[cov] = [indicator_fit.cov_params(), non_zero_fit.cov_params()]
                    model_fits_summary[cov] = [indicator_fit.summary(), non_zero_fit.summary()]

            elif covtypes[k] == 'truncated normal':
                truncation_value = trunc_params[k][0]
                truncation_direction = trunc_params[k][1]
                
                #fit_data.to_parquet("fit_data_{0}.parquet".format(cov))
                
                fit_results = truncreg(formula=covmodels[k], data=fit_data, point=truncation_value, direction=truncation_direction)
                covariate_fits[cov] = fit_results['result']

                _, covmodel = re.split('~', covmodels[k].replace(' ', ''))
                var_names = re.split('\+', covmodel)
                new_data = np.concatenate((np.ones((fit_data.shape[0], 1)), fit_data[var_names].to_numpy()), axis=1)
                fitted_values = np.dot(new_data, fit_results['result']['x'][:-1])
                rmse = np.sqrt(np.mean((fitted_values - fit_data[cov]) ** 2))
                rmses[cov] = rmse
                bounds[cov] = [fit_data[cov].min(), fit_data[cov].max()]
                if return_fits:
                    model_coeffs[cov] = fit_results['result']['x']
                    model_stderrs[cov] = fit_results['SE']
                    model_vcovs[cov] = fit_results['vcov']

            elif covtypes[k] == 'absorbing':
                bounds[cov] = [0, 1]
                fit_data = fit_data[fit_data[time_name] > 0]
                absorb_fit_data = fit_data[fit_data['lag1_{0}'.format(cov)] == 0]
                cov_fit = smf.glm(covmodels[k], absorb_fit_data, family=sm.families.Binomial()).fit()
                rmse = np.sqrt(np.mean((cov_fit.predict() - absorb_fit_data[cov]) ** 2))
                covariate_fits[cov] = cov_fit
                rmses[cov] = rmse
                if return_fits:
                    model_coeffs[cov] = cov_fit.params
                    model_stderrs[cov] = cov_fit.bse
                    model_vcovs[cov] = cov_fit.cov_params()
                    model_fits_summary[cov] = cov_fit.summary()
            
            elif covtypes[k] == 'poisson':
                # empirical bounds, typical for count covariates
                min_cov = fit_data[cov].min()
                max_cov = fit_data[cov].max()
                bounds[cov] = [min_cov, max_cov]

                # Fit Poisson GLM with canonical log link
                cov_fit = smf.glm(
                    covmodels[k],
                    data=fit_data,
                    family=sm.families.Poisson()
                ).fit()

                # RMSE on the mean prediction scale
                rmse = np.sqrt(np.mean((cov_fit.predict() - fit_data[cov]) ** 2))

                covariate_fits[cov] = cov_fit
                rmses[cov] = rmse

                if return_fits:
                    model_coeffs[cov] = cov_fit.params
                    model_stderrs[cov] = cov_fit.bse
                    model_vcovs[cov] = cov_fit.cov_params()
                    model_fits_summary[cov] = cov_fit.summary()

            elif covtypes[k] == 'custom':
                # FIX (NumPy 2.x compat): `np.issubdtype(dtype, np.number)` raises
                # TypeError on pandas extension dtypes (CategoricalDtype, StringDtype, ...).
                # `pd.api.types.is_numeric_dtype` returns False for those cleanly.
                # Also semantically: a Categorical whose codes happen to be int-typed
                # should not record min/max bounds, which `is_numeric_dtype` also
                # correctly handles (returns False).
                if pd.api.types.is_numeric_dtype(fit_data[cov].dtype):
                    bounds[cov] = [fit_data[cov].min(), fit_data[cov].max()]
                fit_func = covfits_custom[k]
                cov_fit = fit_func(covmodel=covmodels[k], covname=covnames[k], fit_data=fit_data)
                covariate_fits[cov] = cov_fit

    #print('check covariate_fits structure')
    #print(type(covariate_fits))
    #print(covariate_fits)
    return covariate_fits, bounds, rmses, model_coeffs, model_stderrs, model_vcovs, model_fits_summary


'''def fit_ymodel(ymodel, outcome_type, outcome_name, ymodel_fit_custom, time_name, obs_data,
               competing, compevent_name, return_fits, yrestrictions):
    """
    This is a function to fit parametric model for the outcome.

    Parameters
    ----------
    ymodel: Str
        A string specifying the model statement for the outcome variable.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    outcome_name: Str
        A string specifying the name of the outcome variable in obs_data.

    ymodel_fit_custom: Function
        A user-specified fit function for the outcome variable.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    obs_data: DataFrame
        Observed data or resampled data used to estimate the parameters of the outcome model.

    competing: Bool
        A boolean value indicating if there is a competing event in obs_data.

    compevent_name: Str
        A string specifying the name of the competing event variable in obs_data. Only applicable for survival outcomes.\

    return_fits: Bool
        A boolean value indicating whether to get the coefficients, standard errors, variance-covariance matrices of the
        fitted outcome model.

    yrestrictions: List
        List of lists. For each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the outcome is modeled, the second entry is the value that is set to the outcome during
        simulation when the conditions in the first entry are not True.

    Returns
    -------
    outcome_fit: Class
        A class object of the fitted model for outcome.

    model_coeffs: Dict
        A dictionary where the key is the outcome name and the value is the parameter estimates of the fitted outcome model.

    model_stderrs: Dict
        A dictionary where the key is the outcome name and the value is the standard errors of the parameter estimates
        of the fitted outcome model.

    model_vcovs: Dict
        A dictionary where the key is the outcome name and the value is the variance-covariance matrices of the parameter
        estimates of the fitted outcome model.

    model_fits_summary: Dict
        A class object that contains the summary information of the fitted outcome model.

    """
    model_coeffs = {}
    model_stderrs = {}
    model_vcovs = {}
    model_fits_summary = {}

    sub_data = obs_data[obs_data[time_name] >= 0]

    fit_data = sub_data
    if yrestrictions is not None:
        for restriction in yrestrictions:
            conditions = restriction[0]
            for cond_var, condition in conditions.items():
                mask = fit_data[cond_var].apply(condition)
                fit_data = fit_data[mask]

    if competing:
        fit_data = fit_data[(fit_data[outcome_name].notna()) & (fit_data[compevent_name] == 0)]
    else:
        fit_data = fit_data[fit_data[outcome_name].notna()]

    if outcome_type == 'survival' or outcome_type == 'binary_eof':
        if ymodel_fit_custom is not None:
            outcome_fit = ymodel_fit_custom(ymodel, fit_data)
        else:
            outcome_fit = smf.glm(ymodel, fit_data, family=sm.families.Binomial()).fit()
    elif outcome_type == 'continuous_eof':
        if ymodel_fit_custom is not None:
            outcome_fit = ymodel_fit_custom(ymodel, fit_data)
        else:
            outcome_fit = smf.glm(ymodel, data=fit_data, family=sm.families.Gaussian()).fit()

    if return_fits and not ymodel_fit_custom:
        model_coeffs[outcome_name] = outcome_fit.params
        model_stderrs[outcome_name] = outcome_fit.bse
        model_vcovs[outcome_name] = outcome_fit.cov_params()
        model_fits_summary[outcome_name] = outcome_fit.summary()

    return outcome_fit, model_coeffs, model_stderrs, model_vcovs, model_fits_summary'''


def fit_compevent_model(compevent_model, compevent_name, time_name, obs_data, return_fits, compevent_restrictions):
    """
    This is a function to fit parametric model for the competing event.

    Parameters
    ----------
    compevent_model: Str
        A string specifying the model statement for the competing event variable.

    compevent_name: Str
        A string specifying the name of the competing event variable in obs_data.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    obs_data: DataFrame
        Observed data or resampled data used to estimate the parameters of the compevent model.

    return_fits: Bool
        A boolean value indicating whether to get the coefficients, standard errors, variance-covariance matrices of the
        fitted compevent model.

    compevent_restrictions: List
        List of lists. For each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the competing event is modeled, the second entry is the value that is set to the competing
        event during simulation when the conditions in the first entry are not True. Only applicable for survival outcomes.

    Returns
    -------
    compevent_fit: Class
        A class object of the fitted model for the competing event.

    model_coeffs: Dict
        A dictionary where the key is the name of competing event and the value is the parameter estimates of the
        fitted compevent model.

    model_stderrs: Dict
        A dictionary where the key is the name of competing event and the value is the standard errors of the parameter
        estimates of the fitted compevent model.

    model_vcovs: Dict
        A dictionary where the key is the name of competing event and the value is the variance-covariance matrices of
        the parameter estimates of the fitted compevent model.

    model_fits_summary: Dict
        A class object that contains the summary information of the fitted compevent model.

    """

    model_coeffs = {}
    model_stderrs = {}
    model_vcovs = {}
    model_fits_summary = {}

    fit_data = obs_data[obs_data[time_name] >= 0]
    if compevent_restrictions is not None:
        for restriction in compevent_restrictions:
            conditions = restriction[0]
            for cond_var, condition in conditions.items():
                mask = fit_data[cond_var].apply(condition)
                fit_data = fit_data[mask]

    fit_data = fit_data[fit_data[compevent_name].notna()]
    compevent_fit = smf.glm(compevent_model, fit_data, family=sm.families.Binomial()).fit()
    if return_fits:
        model_coeffs[compevent_name] = compevent_fit.params
        model_stderrs[compevent_name] = compevent_fit.bse
        model_vcovs[compevent_name] = compevent_fit.cov_params()
        model_fits_summary[compevent_name] = compevent_fit.summary()

    return compevent_fit, model_coeffs, model_stderrs, model_vcovs, model_fits_summary


def fit_censor_model(censor_model, censor_name, time_name, obs_data, return_fits):
    """
    This is a function to fit parametric model for the censor event.

    Parameters
    ----------
    censor_model: Str
        A string specifying the model statement for the censoring variable. Only applicable when using inverse
        probability weights to estimate the natural course means / risk from the observed data.

    censor_name: Str, default is None
        A string specifying the name of the censoring variable in obs_data. Only applicable when using inverse
        probability weights to estimate the natural course means / risk from the observed data.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    obs_data: DataFrame
        Observed data or resampled data used to estimate the parameters of the censor model.

    return_fits: Bool
        A boolean value indicating whether to get the coefficients, standard errors, variance-covariance matrices of the
        fitted censor model.

    Returns
    -------
    censor_fit: Class
        A class object of the fitted model for the censoring event.

    model_coeffs: Dict
        A dictionary where the key is the name of censoring event and the value is the parameter estimates of the
        fitted censor model.

    model_stderrs: Dict
        A dictionary where the key is the name of censoring event and the value is the standard errors of the parameter
        estimates of the fitted censor model.

    model_vcovs: Dict
        A dictionary where the key is the name of censoring event and the value is the variance-covariance matrices of
        the parameter estimates of the fitted censor model.

    model_fits_summary: Dict
        A class object that contains the summary information of the fitted censor model.

    """

    model_coeffs = {}
    model_stderrs = {}
    model_vcovs = {}
    model_fits_summary = {}

    fit_data = obs_data[obs_data[time_name] >= 0]

    fit_data = fit_data[fit_data[censor_name].notna()]
    censor_fit = smf.glm(censor_model, fit_data, family=sm.families.Binomial()).fit()
    if return_fits:
        model_coeffs[censor_name] = censor_fit.params
        model_stderrs[censor_name] = censor_fit.bse
        model_vcovs[censor_name] = censor_fit.cov_params()
        model_fits_summary[censor_name] = censor_fit.summary()

    return censor_fit, model_coeffs, model_stderrs, model_vcovs, model_fits_summary


# Fitting a model for in-icu death
def fit_I_model(I_model, I_name, time_name, obs_data, return_fits,
                I_model_fit_custom=None):
    """
    Fit a parametric (GLM) or custom (e.g. LightGBM) model for the in-ICU death event.

    Parameters
    ----------
    I_model: Str
        A string specifying the model statement for the in-icu death variable.
        For GLM this is a patsy-style formula; for a custom fitter the meaning
        is whatever the user-provided `I_model_fit_custom` function expects
        (typically `"D ~ x1 + x2 + ..."`).

    I_name: Str, default is None
        A string specifying the name of the in-icu death variable in obs_data.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    obs_data: DataFrame
        Observed data or resampled data used to estimate the parameters of the model.

    return_fits: Bool
        A boolean value indicating whether to extract GLM coefficients, SEs, vcovs,
        and summaries. Has no effect when I_model_fit_custom is provided.

    I_model_fit_custom: Callable, optional
        User-supplied fitter with signature
            fit = I_model_fit_custom(I_model, fit_data)
        where `fit_data` is the rows used to fit the model (rows with `I_name` not NA).
        If provided, the GLM path is bypassed.

    Returns
    -------
    I_fit, model_coeffs, model_stderrs, model_vcovs, model_fits_summary
        For GLM: as before.
        For custom: model_coeffs / stderrs / vcovs are empty dicts; model_fits_summary
        may contain `I_fit.feature_importances_` if available (otherwise empty).
    """

    model_coeffs = {}
    model_stderrs = {}
    model_vcovs = {}
    model_fits_summary = {}

    fit_data = obs_data[obs_data[time_name] >= 0]
    fit_data = fit_data[fit_data[I_name].notna()]
    #fit_data.to_parquet("fit_data_I.parquet")

    if I_model_fit_custom is not None:
        # Custom path (e.g., LightGBM). The user is responsible for the
        # internals; pygformula treats `I_fit` as an opaque object that will
        # later be consumed by I_model_predict_custom in simulate.
        I_fit = I_model_fit_custom(I_model, fit_data)
        # Light-touch summary if the fitted object exposes feature importances.
        if return_fits and hasattr(I_fit, "feature_importances_"):
            try:
                model_fits_summary[I_name] = {
                    "feature_name": list(getattr(I_fit, "feature_name_", [])),
                    "feature_importances": list(I_fit.feature_importances_),
                }
            except Exception:
                pass
    else:
        I_fit = smf.glm(I_model, fit_data, family=sm.families.Binomial()).fit()
        if return_fits:
            model_coeffs[I_name] = I_fit.params
            model_stderrs[I_name] = I_fit.bse
            model_vcovs[I_name] = I_fit.cov_params()
            model_fits_summary[I_name] = I_fit.summary()

    return I_fit, model_coeffs, model_stderrs, model_vcovs, model_fits_summary


# Fitting a model for post-discharge mortality with variable hazards until K
def fit_zmodel(zmodel, outcome_type, outcome_name, zmodel_fit_custom, time_name, obs_data,
               competing, compevent_name, return_fits, zrestrictions):
    """
    This is a function to fit parametric model for the outcome (post-discharge mortality with hazards until K).

    Parameters
    ----------
    zmodel: Str
        A string specifying the model statement for the outcome variable.

    outcome_type: Str
        A string specifying the "type" of outcome. The possible "types" are: "survival", "continuous_eof", and "binary_eof".

    outcome_name: Str
        A string specifying the name of the outcome variable in obs_data.

    zmodel_fit_custom: Function
        A user-specified fit function for the outcome variable.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    obs_data: DataFrame
        Observed data or resampled data used to estimate the parameters of the outcome model.

    competing: Bool
        A boolean value indicating if there is a competing event in obs_data.

    compevent_name: Str
        A string specifying the name of the competing event variable in obs_data. Only applicable for survival outcomes.\

    return_fits: Bool
        A boolean value indicating whether to get the coefficients, standard errors, variance-covariance matrices of the
        fitted outcome model.

    zrestrictions: List
        List of lists. For each inner list, its first entry is a dictionary whose key is the conditions which
        should be True when the outcome is modeled, the second entry is the value that is set to the outcome during
        simulation when the conditions in the first entry are not True.

    Returns
    -------
    outcome_fit: Class
        A class object of the fitted model for outcome.

    model_coeffs: Dict
        A dictionary where the key is the outcome name and the value is the parameter estimates of the fitted outcome model.

    model_stderrs: Dict
        A dictionary where the key is the outcome name and the value is the standard errors of the parameter estimates
        of the fitted outcome model.

    model_vcovs: Dict
        A dictionary where the key is the outcome name and the value is the variance-covariance matrices of the parameter
        estimates of the fitted outcome model.

    model_fits_summary: Dict
        A class object that contains the summary information of the fitted outcome model.

    """
    model_coeffs = {}
    model_stderrs = {}
    model_vcovs = {}
    model_fits_summary = {}

    sub_data = obs_data[obs_data[time_name] >= 0]

    z_covs = [
        "vent_mode__last__last_12h", 
        "fio2__last__last_12h", 
        "glasgow_coma_scale_total__last__last_12h", 
        "cumavg_vent_mode__hours_since_last__last_12h", 
        "cumavg_pco2_arterial__mean__last_12h", 
        "cumavg_po2_arterial__mean__last_12h", 
        "cumavg_o2_saturation__mean__last_12h", 
        "cumavg_respiratory_rate_measured__mean__last_12h",
        "cumavg_lactate__last__last_12h", 
        "cumavg_fluid_out_urine__mean__last_12h",
        "cumavg_ureum__last__last_12h", 
        "cumavg_creatinine__last__last_12h",
        "cumavg_arterial_blood_pressure_mean__mean__last_12h",
        "cumavg_heart_rate__mean__last_12h", 
        "cumavg_hemoglobin__last__last_12h",
        "cumavg_temperature__mean__last_12h",
        "cumavg_activated_partial_thromboplastin_time__last__last_12h",
        "cumavg_bicarbonate_arterial__last__last_12h"
    ]

    # ----- SCHEMA NOTE -----
    # Variable-hazards Z model. Build a long-format dataset with one row per
    # (discharged stay, post-discharge bin) for each at-risk bin from tD through
    # min(t_death, t_max) inclusive.
    # Decision grid t in [0, T_MAX] with T_MAX = 59 (30-day horizon, 12h bins).
    # grid_end(t) = (t+1)*12 hours since intime; final follow-up K = intime + 732h.
    # Per-bin hazard model: Z_t ~ z_covs(frozen at discharge) + tsd + tD,
    # where tsd = t - tD (time since discharge).
    # No frequency weights — the long format already encodes at-risk bin counts.
    # -----------------------
    #fit_data_Z = build_fit_data_Z_long(
    #    df=sub_data,
    #    stay_col="admission_id",
    #    t_col="t0",
    #    A_col="A",
    #    t_max=59,
    #    z_covs=z_covs,
    #    death_abs_col="death_abs_time",
    #    death_td_col="death_time_from_intime",
    #)

    # Constant hazards model: one row per discharged stay, with a single Bernoulli outcome
    # indicating whether death occurred within the follow-up window.
    fit_data_Z = build_postdischarge_weighted_rows(
        df=sub_data,
        stay_col="admission_id",
        t_col="t0",
        ref_time_col="ref_time",   # ignored if death_time_from_intime is present (see fallback)
        A_col="A",
        t_max=59,  # FIX: was 60; correct is T_MAX=59 (decision grid t in [0,59])
        z_covs=z_covs,
        death_abs_col="death_abs_time",
        death_td_col="death_time_from_intime",
        check_weights=True,
    )

    #fit_data_Z.to_parquet("fit_data_Z.parquet")

    if zmodel_fit_custom is not None:
        # Fit custom model for Z. The notebook's zmodel_fit_custom is responsible
        # for adding 'tsd' and 'tD' to the feature list (mirroring the GLM path).
        z_outcome_fit = zmodel_fit_custom(zmodel, fit_data_Z)
    else:
        # GLM model for Z. Variable-hazards formulation: per-bin hazard depends
        # on frozen discharge covariates, time-since-discharge (tsd), and
        # discharge time (tD). One Bernoulli outcome per row, no freq_weights.
        z_outcome_fit = smf.glm(
                        zmodel + " + tsd + tD",
                        data=fit_data_Z,                   # already only A==1 risk set, long-format
                        family=sm.families.Binomial()).fit()

    '''if zrestrictions is not None:
        for restriction in zrestrictions:
            conditions = restriction[0]
            for cond_var, condition in conditions.items():
                mask = fit_data[cond_var].apply(condition)
                fit_data = fit_data[mask]'''

    '''if competing:
        fit_data = fit_data[(fit_data[outcome_name].notna()) & (fit_data[compevent_name] == 0)]
    else:
        fit_data = fit_data[fit_data[outcome_name].notna()]'''

    '''if outcome_type == 'survival' or outcome_type == 'binary_eof':
        if zmodel_fit_custom is not None:
            outcome_fit = zmodel_fit_custom(zmodel, fit_data)
        else:
            outcome_fit = smf.glm(zmodel, fit_data, family=sm.families.Binomial()).fit()
    elif outcome_type == 'continuous_eof':
        if zmodel_fit_custom is not None:
            outcome_fit = zmodel_fit_custom(zmodel, fit_data)
        else:
            outcome_fit = smf.glm(zmodel, data=fit_data, family=sm.families.Gaussian()).fit()'''

    if return_fits and not zmodel_fit_custom:
        model_coeffs[outcome_name] = z_outcome_fit.params
        model_stderrs[outcome_name] = z_outcome_fit.bse
        model_vcovs[outcome_name] = z_outcome_fit.cov_params()
        model_fits_summary[outcome_name] = z_outcome_fit.summary()

    return z_outcome_fit, model_coeffs, model_stderrs, model_vcovs, model_fits_summary

'''
# Dataset for post-discharge variable hazards until K
# (e.g. 30 days in 12h grids -> t_max=59; the grid runs t in [0, t_max] inclusive.)
def build_fit_data_Z_long(
    df: pd.DataFrame,
    *,
    stay_col: str = "admission_id",
    t_col: str = "t0",
    A_col: str = "A",
    t_max: int = 59,
    z_covs=None,
    death_abs_col: str = "death_abs_time",            # datetime64[ns] or None
    death_td_col: str = "death_time_from_intime",     # timedelta64[ns] or None
) -> pd.DataFrame:
    """
    Build a long-format post-discharge dataset for a variable-hazards discrete-time
    Z model. For each discharged stay, emit one row per at-risk post-discharge bin
    from tD through min(t_death, t_max) inclusive.

    Output columns: [stay_col, "tD", "tsd", "t", "Z", *z_covs]
      - tD:  discharge bin (constant per stay)
      - tsd: time since discharge in 12h bins; tsd = 0 is the discharge bin itself
      - t:   grid bin = tD + tsd
      - Z:   per-bin death indicator. 1 in the bin where death occurred, 0 elsewhere.
      - z_covs: frozen at the discharge row values.

    Three cases (handled uniformly by the long-format expansion):
      Case 1: death in the discharge bin itself (t_death == tD)
              -> 1 row, tsd = 0, Z = 1.
      Case 2: death after discharge but within follow-up (tD < t_death <= t_max)
              -> (t_death - tD + 1) rows, Z = 0 for tsd < (t_death - tD),
                 Z = 1 at tsd = (t_death - tD).
      Case 3: no death within follow-up (death missing or t_death > t_max)
              -> (t_max - tD + 1) rows, Z = 0 throughout.

    Notes:
    - We include only stays that have an A==1 row (discharge observed).
    - z_covs are frozen at the discharge row values across all post-discharge bins.
    - t_death is computed from death_time_from_intime if available; otherwise from
      death_abs_time minus intime.
    - Bin mapping uses floor(td/12h) - 1, matching the convention in the rest of
      the pipeline (Y construction, observed-Z derivation in 3_organize).
    - No frequency weights — the long format already encodes at-risk bin counts
      via row repetition.
    """

    if z_covs is None:
        raise ValueError("Please pass z_covs list explicitly.")

    # ---- Minimal column checks (fail fast) ----
    needed = {stay_col, t_col, A_col, *z_covs}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Ensure sorted so "first A==1" is deterministic if duplicates exist
    df = df.sort_values([stay_col, t_col], kind="mergesort")

    # ---- 1) Identify unique discharge row per stay (A==1) ----
    extra_cols = []
    if death_td_col in df.columns:
        extra_cols.append(death_td_col)
    if death_abs_col in df.columns:
        extra_cols.append(death_abs_col)
    if "intime" in df.columns:
        extra_cols.append("intime")

    discharge = (
        df.loc[df[A_col] == 1, [stay_col, t_col, *z_covs, *extra_cols]]
          .drop_duplicates(subset=[stay_col], keep="first")
          .copy()
    )

    out_cols = [stay_col, "tD", "tsd", "t", "Z", *z_covs]
    if discharge.empty:
        return pd.DataFrame(columns=out_cols)

    discharge = discharge.rename(columns={t_col: "tD"})
    discharge["tD"] = discharge["tD"].astype(int)

    # ---- 2) Compute death time-from-intime (timedelta) robustly ----
    death_td = None
    if death_td_col in discharge.columns:
        death_td = discharge[death_td_col]

    if (death_td is None) or death_td.isna().all():
        # Fallback: requires death_abs_col AND intime in the discharge frame.
        if death_abs_col in discharge.columns and "intime" in discharge.columns:
            death_td = discharge[death_abs_col] - discharge["intime"]
        else:
            death_td = pd.Series(pd.NaT, index=discharge.index, dtype="timedelta64[ns]")

    death_td = pd.to_timedelta(death_td, errors="coerce")

    # ---- 3) Map death timedelta to grid index t_death ----
    # Schema: W^Y_t = [grid_end(t), grid_end(t+1)) = [12(t+1), 12(t+2))h since intime.
    # A death at hour tau belongs to W^Y_t iff floor(tau/12) - 1 == t.
    # This matches the convention used by the rest of the pipeline.
    bin_len = pd.Timedelta(hours=12)

    td_valid = death_td.notna() & (death_td >= pd.Timedelta(0))
    td_ratio = (death_td[td_valid] / bin_len).astype(float)

    t_death = pd.Series(pd.NA, index=discharge.index, dtype="Int64")
    t_death.loc[td_valid] = (np.floor(td_ratio).astype(int) - 1)

    # Eligibility: no death before tau_0 should reach here, since 3_organize's
    # cell-39 eligibility filter excludes stays with death_time_from_intime < 12h.
    if (t_death.dropna() < 0).any():
        n_bad = int((t_death.dropna() < 0).sum())
        raise AssertionError(
            f"build_fit_data_Z_long: found {n_bad} discharged stays with "
            f"t_death < 0 (death before tau_0 = intime + 12h). "
            f"This should be impossible given upstream eligibility filtering."
        )

    discharge["t_death"] = t_death

    # ---- 4) Determine per-stay expansion end (t_end) and n_steps ----
    # If death is within follow-up, expand from tD through t_death; otherwise
    # expand through t_max. Pre-discharge deaths and stays without tD are
    # excluded (no discharge -> no row in `discharge`).
    tD_arr = discharge["tD"].to_numpy(dtype=int)
    td_known_within = t_death.notna() & (t_death.astype("Int64") <= t_max) & (t_death.astype("Int64") >= 0)
    # Convert to numpy bool aligned with discharge
    td_known_within_np = td_known_within.reindex(discharge.index).fillna(False).to_numpy()

    # t_end[i] = t_death[i] if known-within else t_max
    t_end = np.where(td_known_within_np,
                     t_death.reindex(discharge.index).astype("Int64").fillna(-1).to_numpy().astype(int),
                     t_max)

    # Drop any stay whose tD > t_end (discharge after horizon — should not happen
    # under upstream eligibility, but guard against silent miscoded rows).
    n_steps = (t_end - tD_arr + 1)
    keep = n_steps > 0

    if not keep.all():
        n_dropped = int((~keep).sum())
        # Soft warning via print to avoid silent loss; reviewers can audit.
        print(f"build_fit_data_Z_long: dropping {n_dropped} stays with tD > t_end "
              f"(degenerate post-discharge windows).")

    disc2 = discharge.loc[keep].copy()
    tD_arr2 = tD_arr[keep]
    t_end2 = t_end[keep]
    n_steps2 = n_steps[keep]
    t_death_arr2 = t_death.reindex(disc2.index).astype("Int64").to_numpy()  # may contain pd.NA

    # ---- 5) Expand: repeat each discharge row n_steps times ----
    rep_idx = np.repeat(np.arange(len(disc2)), n_steps2)
    post = disc2.iloc[rep_idx].copy()

    # offsets within each block: 0, 1, ..., n_steps - 1
    ends = np.cumsum(n_steps2)
    starts = ends - n_steps2
    offsets = np.empty(int(ends[-1]) if len(ends) else 0, dtype=int)
    for i in range(len(n_steps2)):
        offsets[starts[i]:ends[i]] = np.arange(int(n_steps2[i]), dtype=int)

    post["tD"] = np.repeat(tD_arr2, n_steps2)
    post["tsd"] = offsets
    post["t"] = post["tD"].to_numpy(dtype=int) + post["tsd"].to_numpy(dtype=int)

    # ---- 6) Z outcome: 1 in the death bin, 0 elsewhere ----
    # For stays in Case 1/2: Z=1 at t == t_death; Z=0 otherwise.
    # For Case 3 stays: t_death is NA, so Z=0 in all rows.
    t_death_rep = np.repeat(
        np.where(pd.isna(t_death_arr2), -1, t_death_arr2).astype(int),
        n_steps2,
    )
    post["Z"] = (post["t"].to_numpy(dtype=int) == t_death_rep).astype(int)
    # Defensive: where t_death is -1 (case 3 sentinel), all Z must be 0
    post.loc[t_death_rep == -1, "Z"] = 0

    # ---- 7) Final column selection and sort ----
    out = post[[stay_col, "tD", "tsd", "t", "Z", *z_covs]].sort_values(
        [stay_col, "tsd"], kind="mergesort"
    ).reset_index(drop=True)

    return out'''



# Dataset for post-discharge constant hazards until K
# (e.g. 30 days in 12h grids -> t_max=59; the grid runs t in [0, t_max] inclusive.)
def build_postdischarge_weighted_rows(
    df: pd.DataFrame,
    *,
    stay_col: str = "admission_id",
    t_col: str = "t0",
    ref_time_col: str = "ref_time",
    A_col: str = "A",                 # <-- set to your actual A column name
    t_max: int = 59,                 # FIX: was 60; t in [0, 59], 60 bins total
    z_covs=None,
    death_abs_col: str = "death_abs_time",            # datetime64[ns] or None
    death_td_col: str = "death_time_from_intime",     # timedelta64[ns] or None
    check_weights: bool = True,
) -> pd.DataFrame:
    """
    Build a post-discharge dataset with weights for a discrete-time Z model,
    producing at most 2 rows per discharged stay:
      Case 1: death in (tD, tD+1] -> 1 row: Z=1, weight=1
      Case 2: death after tD and within follow-up -> 2 rows:
              row1: Z=0, weight=t_death - tD
              row2: Z=1, weight=1
      Case 3: no death within follow-up -> 1 row: Z=0, weight=t_max+1 - tD

    Notes:
    - We only include stays that have an A==1 row (discharge observed).
    - z_covs are frozen at the discharge row values.
    - t_death is computed from death_time_from_intime if available; otherwise from
      death_abs_time minus an inferred intime based on (ref_time, t).
    """

    if z_covs is None:
        raise ValueError("Please pass z_covs list explicitly.")

    # ---- Minimal column checks (fail fast) ----
    #needed = {stay_col, t_col, ref_time_col, A_col, *z_covs}
    needed = {stay_col, t_col, A_col, *z_covs}
    
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"Missing required columns: {sorted(missing)}")

    # Ensure sorted so "first A==1" is deterministic if duplicates exist
    df = df.sort_values([stay_col, t_col], kind="mergesort")

    # ---- 1) Identify unique discharge row per stay (A==1) ----
    # We also pull `intime` if present, to enable a clean fallback for death timing
    # when `death_time_from_intime` is unavailable.
    extra_cols = []
    if death_td_col in df.columns:
        extra_cols.append(death_td_col)
    if death_abs_col in df.columns:
        extra_cols.append(death_abs_col)
    if "intime" in df.columns:
        extra_cols.append("intime")

    discharge = (
        df.loc[df[A_col] == 1, [stay_col, t_col, *z_covs, *extra_cols]]
          .drop_duplicates(subset=[stay_col], keep="first")
          .copy()
    )

    # If no discharged stays, return empty frame with expected columns
    out_cols = [stay_col, "tD", "t_death", "Z", "weight", *z_covs]
    if discharge.empty:
        return pd.DataFrame(columns=out_cols)

    discharge = discharge.rename(columns={t_col: "tD"})
    discharge["tD"] = discharge["tD"].astype(int)

    # ---- 2) Compute death time-from-intime (timedelta) robustly ----
    # Prefer death_time_from_intime if present & non-null; otherwise fall back to
    # (death_abs_time - intime). The previous version used ref_time, which is a
    # derived quantity not always available; intime is per-stay invariant and
    # almost always present in the source schema (3_organize).
    death_td = None

    if death_td_col in discharge.columns:
        death_td = discharge[death_td_col]

    if (death_td is None) or death_td.isna().all():
        # Fallback: requires death_abs_col AND intime in the discharge frame.
        if death_abs_col in discharge.columns and "intime" in discharge.columns:
            death_td = discharge[death_abs_col] - discharge["intime"]
        else:
            # No way to compute death timing
            death_td = pd.Series(pd.NaT, index=discharge.index, dtype="timedelta64[ns]")

    # Normalize to timedelta64[ns]
    death_td = pd.to_timedelta(death_td, errors="coerce")

    # ---- 3) Map death timedelta to grid index t_death ----
    # Schema: W^Y_t = [grid_end(t), grid_end(t+1)) = [12(t+1), 12(t+2))h since intime.
    # A death at hour tau belongs to W^Y_t iff floor(tau/12) - 1 == t.
    # (The previous code used `ceil(tau/12) - 1`, which only agrees at exact 12h
    # boundaries and is off-by-one for any interior death timestamp.)
    bin_len = pd.Timedelta(hours=12)

    td_valid = death_td.notna() & (death_td >= pd.Timedelta(0))
    td_ratio = (death_td[td_valid] / bin_len).astype(float)

    t_death = pd.Series(pd.NA, index=discharge.index, dtype="Int64")
    t_death.loc[td_valid] = (np.floor(td_ratio).astype(int) - 1)

    # Eligibility: no death before tau_0 should reach here, since 3_organize's
    # cell-39 eligibility filter excludes stays with death_time_from_intime < 12h.
    # We assert it as a safety net rather than silently coercing.
    if (t_death.dropna() < 0).any():
        n_bad = int((t_death.dropna() < 0).sum())
        raise AssertionError(
            f"build_postdischarge_weighted_rows: found {n_bad} discharged stays with "
            f"t_death < 0 (death before tau_0 = intime + 12h). "
            f"This should be impossible given upstream eligibility filtering."
        )

    # If death occurs after follow-up, treat as "no death within K"
    t_death_within = t_death.notna() & (t_death <= t_max)

    discharge["t_death"] = t_death

    # ---- 4) Build rows for the 3 cases (vectorized) ----
    # Case 1: death in [tD, tD+1)  <=> t_death == tD (with your mapping)
    case1 = t_death_within & (discharge["t_death"] == discharge["tD"])

    # Case 2: death after tD but within follow-up
    case2 = t_death_within & (discharge["t_death"] > discharge["tD"])

    # Case 3: no death within follow-up (including missing death time or after t_max)
    case3 = ~t_death_within

    base = discharge[[stay_col, "tD", "t_death", *z_covs]].copy()

    # --- Case 1 output ---
    out1 = base.loc[case1].copy()
    out1["Z"] = 1
    out1["weight"] = 1

    # --- Case 2 output: two rows per stay ---
    b2 = base.loc[case2].copy()

    # Row with Z=0, weight = t_death - tD
    out2a = b2.copy()
    out2a["Z"] = 0
    out2a["weight"] = (out2a["t_death"].astype(int) - out2a["tD"].astype(int)).astype(int)

    # Row with Z=1, weight=1
    out2b = b2.copy()
    out2b["Z"] = 1
    out2b["weight"] = 1

    # --- Case 3 output ---
    out3 = base.loc[case3].copy()
    out3["Z"] = 0
    out3["t_death"] = pd.NA
    out3["weight"] = (t_max + 1 - out3["tD"].astype(int)).astype(int)

    out = pd.concat([out1, out2a, out2b, out3], ignore_index=True)
    out = out[[stay_col, "tD", "t_death", "Z", "weight", *z_covs]].sort_values([stay_col, "Z"], kind="mergesort")

    # ---- 5) Sanity checks on weights ----
    if check_weights:
        # Basic positivity
        bad = out["weight"].isna() | (out["weight"] <= 0)
        if bad.any():
            ex = out.loc[bad, [stay_col, "tD", "t_death", "Z", "weight"]].head(10)
            raise AssertionError(f"Found non-positive/NA weights. Examples:\n{ex}")

        # Per-stay expected total weight:
        # - Case1 total = 1
        # - Case2 total = (t_death - tD) + 1
        # - Case3 total = (t_max + 1 - tD)
        # We recompute from discharge table (one row per stay) and compare to sum of output weights.
        disc_expect = discharge[[stay_col, "tD", "t_death"]].copy()

        exp_total = pd.Series(index=disc_expect.index, dtype="int64")

        exp_total.loc[case1] = 1
        exp_total.loc[case2] = (disc_expect.loc[case2, "t_death"].astype(int) - disc_expect.loc[case2, "tD"].astype(int) + 1)
        exp_total.loc[case3] = (t_max + 1 - disc_expect.loc[case3, "tD"].astype(int))

        got_total = out.groupby(stay_col, sort=False)["weight"].sum()
        exp_total_by_stay = pd.Series(exp_total.values, index=disc_expect[stay_col].values)

        # Align indices and compare
        exp_aligned = exp_total_by_stay.loc[got_total.index]
        mismatch = (got_total.values != exp_aligned.values)

        if mismatch.any():
            bad_stays = got_total.index[mismatch][:10]
            detail = pd.DataFrame({
                stay_col: bad_stays,
                "expected_total_weight": exp_aligned.loc[bad_stays].values,
                "got_total_weight": got_total.loc[bad_stays].values,
            })
            raise AssertionError(f"Weight-sum sanity check failed for some stays. Examples:\n{detail}")

    return out