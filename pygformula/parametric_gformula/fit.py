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

            # Restrict to survive ICU and proceed to discharge (D=0). These are also the crowd who have A=0 and A=1.
            # fit_data = fit_data[fit_data["D"] == 0] # Survivors

            if restrictions is not None:
               restrictcovs = [restrictions[0] for i in range(len(restrictions))]
               if cov in restrictcovs:
                   index = restrictcovs.index(cov)
                   conditions = restrictions[index][1]
                   for cond_var, condition in conditions.items():
                       mask = fit_data[cond_var].apply(condition)
                       fit_data = fit_data[mask]

            # exclude rows that contains NA values of the predictors in fit_data
            predictors = covmodels[k].split('~')[1].strip().split(' + ')
            all_vars = predictors + [cov]
            all_vars = [item[2:-1] if item.startswith('C(') and item.endswith(')') else item for item in all_vars]
            fit_data = fit_data[all_vars].dropna()

            if covtypes[k] == 'binary':
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

            elif covtypes[k] == 'custom':
                fit_func = covfits_custom[k]
                cov_fit = fit_func(covmodel=covmodels[k], covname=covnames[k], fit_data=fit_data)
                covariate_fits[cov] = cov_fit

    #print('check covariate_fits structure')
    #print(type(covariate_fits))
    #print(covariate_fits)
    return covariate_fits, bounds, rmses, model_coeffs, model_stderrs, model_vcovs, model_fits_summary


def fit_ymodel(ymodel, outcome_type, outcome_name, ymodel_fit_custom, time_name, obs_data,
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

    return outcome_fit, model_coeffs, model_stderrs, model_vcovs, model_fits_summary


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
def fit_I_model(I_model, I_name, time_name, obs_data, return_fits):
    """
    This is a function to fit parametric model for the in-icu death event.

    Parameters
    ----------
    I_model: Str
        A string specifying the model statement for the in-icu death variable

    I_name: Str, default is None
        A string specifying the name of the in-icu death variable in obs_data.

    time_name: Str
        A string specifying the name of the time variable in obs_data.

    obs_data: DataFrame
        Observed data or resampled data used to estimate the parameters of the censor model.

    return_fits: Bool
        A boolean value indicating whether to get the coefficients, standard errors, variance-covariance matrices of the
        fitted censor model.

    Returns
    -------
    I_fit: Class
        A class object of the fitted model for the in-icu death event.

    model_coeffs: Dict
        A dictionary where the key is the name of in-icu death event and the value is the parameter estimates of the
        fitted in-icu death model.

    model_stderrs: Dict
        A dictionary where the key is the name of in-icu death event and the value is the standard errors of the parameter
        estimates of the fitted in-icu death model.

    model_vcovs: Dict
        A dictionary where the key is the name of in-icu death event and the value is the variance-covariance matrices of
        the parameter estimates of the fitted in-icu death model.

    model_fits_summary: Dict
        A class object that contains the summary information of the fitted in-icu death model.

    """

    model_coeffs = {}
    model_stderrs = {}
    model_vcovs = {}
    model_fits_summary = {}

    fit_data = obs_data[obs_data[time_name] >= 0]

    fit_data = fit_data[fit_data[I_name].notna()]
    I_fit = smf.glm(I_model, fit_data, family=sm.families.Binomial()).fit()
    if return_fits:
        model_coeffs[I_name] = I_fit.params
        model_stderrs[I_name] = I_fit.bse
        model_vcovs[I_name] = I_fit.cov_params()
        model_fits_summary[I_name] = I_fit.summary()

    return I_fit, model_coeffs, model_stderrs, model_vcovs, model_fits_summary


# Fitting a model for post-discharge mortality with constant hazards until K
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
        "A",
        "vent_mode__last__last_12h",
        "cumavg_vent_mode__hours_since_last__last_12h",
        "cumavg_pco2_arterial__mean__last_12h",
        "cumavg_po2_arterial__mean__last_12h",
        "cumavg_o2_flow__last__last_12h",
        "cumavg_o2_saturation__mean__last_12h",
        "cumavg_respiratory_rate_measured__mean__last_12h",
        "cumavg_glasgow_coma_scale_total__last__last_12h",
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

    # Reweigh rows
    fit_data_Z, weight_check_df, sanity = build_post_discharge_agg(
        df=sub_data,
        stay_col="stay_id",
        t_col="t0",
        A_col="A",
        D_col="D",
        death_td_col="death_time_from_intime",
        death_abs_col="death_abs_time",
        intime_col="intime",
        t_max=179,
        z_covs=z_covs,
    )
    
    fit_data_Z.to_parquet("fit_data_Z.parquet")

    if zmodel_fit_custom is not None:
        # Fit custom lgb model for Z
        z_outcome_fit = zmodel_fit_custom(zmodel, fit_data_Z)
    else:
        # GLM model for Z
        z_outcome_fit = smf.glm(
                        zmodel,
                        data=fit_data_Z,                   # already only A==1 risk set
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


# Dataset for post-discharge constant hazards until K (e.g. 90 days in 12h grids -> t_max=179)
def build_post_discharge_agg(
    df: pd.DataFrame,
    stay_col: str = "stay_id",
    t_col: str = "t",
    A_col: str = "A",
    D_col: str = "D",
    death_td_col: str = "death_time_from_intime",   # preferred (timedelta or hours)
    death_abs_col: str = "death_abs_time",          # optional fallback
    intime_col: str = "intime",                     # needed only if using death_abs_col fallback
    t_max: int = 179,
    z_covs: list | None = None,
):
    """
    Build post-discharge aggregated dataset for fitting a weighted GLM on Z.

    Output rows per discharged stay:
      Case 1 (death in [tD, tD+1)): 1 row -> Z=1, weight=1
      Case 2 (death after tD within follow-up): 2 rows -> (Z=0, w=t_death-tD) and (Z=1, w=1)
      Case 3 (no death within follow-up): 1 row -> Z=0, weight=t_max+1-tD, t_death=NaN

    Notes:
    - "Freeze z_covs at discharge": take z_covs from the discharge row and repeat into output rows.
    - Excludes stays with no discharge (no A==1), including in-ICU deaths where last row has D==1.
    """

    if z_covs is None:
        raise ValueError("Pass z_covs explicitly (your list of discharge-frozen predictors).")

    # --- 1) Identify unique discharge rows (A==1), one per stay_id ---
    dis = df.loc[df[A_col].eq(1), [stay_col, t_col, D_col, death_td_col, death_abs_col, intime_col] + z_covs].copy()

    # If a stay has multiple A==1 rows (should not), keep the first by time.
    # (You said it should be unique; we still guard for robustness.)
    dis.sort_values([stay_col, t_col], inplace=True)
    dis = dis.drop_duplicates(subset=[stay_col], keep="first")

    # Rename discharge time
    dis.rename(columns={t_col: "tD"}, inplace=True)

    # --- 2) Compute t_death on 12h grid (t=0 is [0h,12h), t=1 is [12h,24h), etc.) ---
    # Prefer death_time_from_intime if present & non-null; else fallback to death_abs_time - intime.
    # Supports death_time_from_intime as timedelta-like OR numeric hours.
    death_td = None

    if death_td_col in dis.columns and dis[death_td_col].notna().any():
        death_td = dis[death_td_col]
    elif (death_abs_col in dis.columns) and (intime_col in dis.columns) and dis[death_abs_col].notna().any():
        # fallback: absolute timestamps -> timedelta
        death_td = pd.to_datetime(dis[death_abs_col]) - pd.to_datetime(dis[intime_col])
    else:
        # no death info available
        death_td = pd.Series([pd.NaT] * len(dis), index=dis.index)

    # Convert death_td to hours as float
    if np.issubdtype(death_td.dtype, np.number):
        death_hours = death_td.astype(float)
    else:
        # timedelta-like
        death_hours = death_td.dt.total_seconds() / 3600.0

    # Map to 12h grid window index using half-open intervals:
    # t_death = floor(death_hours / 12).
    # Examples: 0h->0, 11.9h->0, 12h->1, 23.9h->1, 24h->2
    t_death = np.floor(death_hours / 12.0)
    t_death = pd.Series(t_death, index=dis.index)

    # Missing death time => NaN
    t_death = t_death.where(np.isfinite(t_death), np.nan).astype("Float64")
    dis["t_death"] = t_death

    # --- 3) Keep only deaths that occur AFTER/AT discharge and within follow-up (0..t_max) ---
    # If death is before discharge (shouldn’t be post-discharge), treat as "no post-discharge death".
    within_fu = dis["t_death"].notna() & (dis["t_death"] >= dis["tD"]) & (dis["t_death"] <= t_max)

    # Case 1: death in [tD, tD+1)  <=> t_death == tD
    case1 = within_fu & (dis["t_death"] == dis["tD"])

    # Case 2: death after discharge within follow-up  <=> t_death > tD
    case2 = within_fu & (dis["t_death"] > dis["tD"])

    # Case 3: no death within follow-up (includes missing t_death or after t_max or before tD)
    case3 = ~within_fu

    # --- 4) Build output rows (vectorized) ---
    base_cols = [stay_col, "tD", "t_death"] + z_covs

    # Case 1 output: one row Z=1, weight=1
    out1 = dis.loc[case1, base_cols].copy()
    out1["Z"] = 1
    out1["weight"] = 1.0

    # Case 2 output: two rows
    # Row A: Z=0, weight = t_death - tD
    out2a = dis.loc[case2, base_cols].copy()
    out2a["Z"] = 0
    out2a["weight"] = (out2a["t_death"].astype(float) - out2a["tD"].astype(float)).astype(float)

    # Row B: Z=1, weight = 1
    out2b = dis.loc[case2, base_cols].copy()
    out2b["Z"] = 1
    out2b["weight"] = 1.0

    # Case 3 output: one row Z=0, t_death=NaN, weight = t_max+1 - tD
    out3 = dis.loc[case3, base_cols].copy()
    out3["t_death"] = pd.Series([pd.NA] * len(out3), index=out3.index, dtype="Float64")
    out3["Z"] = 0
    out3["weight"] = (t_max + 1 - out3["tD"].astype(float)).astype(float)

    out = pd.concat([out1, out2a, out2b, out3], axis=0, ignore_index=True)

    # --- 5) Sanity checks on weights ---
    # Expected total weight per stay:
    # - if death within follow-up: (t_death - tD + 1)
    # - else: (t_max - tD + 1)
    # Note: your construction yields total weight == number of 12h intervals covered (inclusive of terminal interval).
    totals = out.groupby(stay_col)["weight"].sum()

    # Recompute expectation from discharge frame
    exp = dis[[stay_col, "tD", "t_death"]].copy()
    exp["within_fu"] = within_fu.values
    exp["expected_total_weight"] = np.where(
        exp["within_fu"],
        (exp["t_death"].astype(float) - exp["tD"].astype(float) + 1.0),
        (t_max - exp["tD"].astype(float) + 1.0),
    )
    exp = exp.set_index(stay_col)

    # Align and check
    aligned = exp.join(totals.rename("observed_total_weight"), how="inner")
    aligned["weight_ok"] = np.isclose(
        aligned["observed_total_weight"].astype(float),
        aligned["expected_total_weight"].astype(float),
        rtol=0,
        atol=1e-8,
    )

    sanity = {
        "n_discharged_stays": int(dis[stay_col].nunique()),
        "n_output_rows": int(len(out)),
        "case1_stays": int(case1.sum()),
        "case2_stays": int(case2.sum()),
        "case3_stays": int(case3.sum()),
        "weight_failures": int((~aligned["weight_ok"]).sum()),
    }

    # Extra guardrails
    # - weights must be positive
    # - case2 Z=0 rows must have integer-like weights >=1 (since t_death > tD)
    if (out["weight"] <= 0).any():
        sanity["warning_nonpositive_weights"] = True

    return out, aligned.reset_index(), sanity
