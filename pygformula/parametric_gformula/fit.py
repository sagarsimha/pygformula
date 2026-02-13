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


# Fitting a model for post-discharge mortality with hazards until K
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
    
    # Create a version of obs_data that has exapnded rows after discharge until death/end of follow-up K. Stays with in-icu death remain unchanged.
    expanded_sub_data = expand_post_discharge_Z_fast(sub_data, assume_types_ok=True)
    fit_data = expanded_sub_data

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

    fit_data_Z = fit_data[fit_data["A"] == 1].copy()

    # discharge time per stay (first A==1 row)
    td = (fit_data_Z[fit_data_Z["A"] == 1]
        .sort_values(["admission_id","t0"])
        .groupby("admission_id")["t0"].min())

    fit_data_Z["tD"] = fit_data_Z["admission_id"].map(td)
    fit_data_Z["tsd"] = fit_data_Z["t0"] - fit_data_Z["tD"]  # time since discharge

    zmodel_t = zmodel + " + tsd + tD"

    # Hazard model for post-discharge mortality until K, restricted to rows with A=1 (discharged)
    z_outcome_fit = smf.glm(zmodel_t, data=fit_data_Z, family=sm.families.Binomial()).fit()

    if return_fits and not zmodel_fit_custom:
        model_coeffs[outcome_name] = z_outcome_fit.params
        model_stderrs[outcome_name] = z_outcome_fit.bse
        model_vcovs[outcome_name] = z_outcome_fit.cov_params()
        model_fits_summary[outcome_name] = z_outcome_fit.summary()

    return z_outcome_fit, model_coeffs, model_stderrs, model_vcovs, model_fits_summary


# Expanding dataset for post-discharge hazards until K (e.g. 90 days in 12h grids -> t_max=179)
def expand_post_discharge_Z_fast(
    df: pd.DataFrame,
    id_col: str = "admission_id",
    A_col: str = "A",
    t_col: str = "t0",
    grid_end_col: str = "grid_end",          # Timedelta-like or string "0 days 12:00:00"
    ref_time_col: str = "ref_time",          # datetime-like
    Z_col: str = "Z",
    death_td_col: str = "death_time_from_intime",  # preferred Timedelta since intime
    death_abs_col: str = "death_abs_time",         # fallback absolute datetime
    intime_col: str = "intime",                    # needed if using death_abs_col
    t_max: int = 179,                              # 90 days in 12h grids
    step_hours: int = 12,
    assume_types_ok: bool = False,                 # set True if you've already coerced dtypes earlier
) -> pd.DataFrame:
    """
    Expand discharged stays to fixed follow-up (t_max) using post-discharge discrete-time hazards.

    Rules implemented:
      - Z = NaN for all A==0 rows
      - Only stays with a discharge row (first A==1) are expanded
      - Expansion rows start at tD+1 (retain discharge row)
      - Freeze all columns from discharge row, except t/grid_end/ref_time which advance by 12h
      - Map death time to grid: t_death = ceil(death_td / 12h) - 1
      - If t_death == tD: discharge row Z=1 else discharge row Z=0
      - Expanded rows have Z=0, except Z=1 at t_death (if t_death > tD and within follow-up)
      - Expand stops at min(t_death, t_max) or t_max if no death/after follow-up
    """

    out = df.copy()

    # Ensure Z exists
    if Z_col not in out.columns:
        out[Z_col] = np.nan

    # --- Coerce types (costly; skip if you already did this upstream) ---
    if not assume_types_ok:
        out[t_col] = out[t_col].astype(int)
        out[grid_end_col] = pd.to_timedelta(out[grid_end_col])
        out[ref_time_col] = pd.to_datetime(out[ref_time_col], errors="coerce")

        if death_td_col in out.columns:
            out[death_td_col] = pd.to_timedelta(out[death_td_col], errors="coerce")
        if (death_abs_col in out.columns) and (intime_col in out.columns):
            out[death_abs_col] = pd.to_datetime(out[death_abs_col], errors="coerce")
            out[intime_col] = pd.to_datetime(out[intime_col], errors="coerce")

    step = pd.Timedelta(hours=step_hours)
    step_ns = step.value  # nanoseconds per step

    # ---------------------------
    # 1) Identify discharged stays
    # ---------------------------
    discharged = out[out[A_col] == 1]
    if discharged.empty:
        # No discharges: only enforce pre-discharge Z=NaN and return
        out.loc[out[A_col] == 0, Z_col] = np.nan
        return out.sort_values([id_col, t_col]).reset_index(drop=True)

    # First discharge row per stay (fast): index of min t among A==1 rows
    first_discharge_idx = discharged.groupby(id_col)[t_col].idxmin()
    disc = out.loc[first_discharge_idx].copy()  # one row per discharged stay

    # Discharge time index tD
    tD = disc[t_col].astype(int).to_numpy()

    # -----------------------------------
    # 2) Compute death time since intime
    # -----------------------------------
    # Preferred: death_time_from_intime (Timedelta)
    if death_td_col in disc.columns:
        death_td = disc[death_td_col]
    # Fallback: death_abs_time - intime (Timedelta)
    elif (death_abs_col in disc.columns) and (intime_col in disc.columns):
        death_td = disc[death_abs_col] - disc[intime_col]
    else:
        death_td = pd.Series(pd.NaT, index=disc.index, dtype="timedelta64[ns]")

    # Map death_td -> t_death = ceil(death_td/step) - 1, missing -> -1
    valid_death = death_td.notna().to_numpy()
    t_death = np.full(len(disc), -1, dtype=int)

    if valid_death.any():
        a_ns = death_td.astype("int64").to_numpy()[valid_death]  # nanoseconds since intime
        # integer ceil division: ceil(a/b) = (a + b - 1)//b for a>0
        q = (a_ns + step_ns - 1) // step_ns
        td_tmp = (q - 1).astype(int)
        td_tmp = np.maximum(td_tmp, 0)  # deaths in first interval -> 0
        t_death[valid_death] = td_tmp

    # If death occurs before discharge, ignore it for post-discharge Z (treat as no post-discharge death)
    t_death = np.where((t_death != -1) & (t_death < tD), -1, t_death)

    # ---------------------------------------
    # 3) Set Z on the *discharge row itself*
    # ---------------------------------------
    # Default: discharge row Z=0
    disc_Z = np.zeros(len(disc), dtype=int)
    # If death happens in [tD, tD+1) => t_death == tD => discharge row Z=1
    disc_Z = np.where((t_death != -1) & (t_death == tD) & (t_death <= t_max), 1, disc_Z)

    # Write discharge-row Z back into the original out dataframe
    out.loc[disc.index, Z_col] = disc_Z

    # ------------------------------------------
    # 4) Determine expansion end and row counts
    # ------------------------------------------
    # If no death/after follow-up: expand to t_max
    t_end = np.where(t_death == -1, t_max, np.minimum(t_death, t_max))

    # Expand rows from tD+1 .. t_end inclusive
    n_expand = np.maximum(t_end - tD, 0)

    mask = n_expand > 0
    if mask.any():
        disc2 = disc.loc[mask].copy()
        n_expand2 = n_expand[mask]
        tD2 = tD[mask]
        t_death2 = t_death[mask]

        # Repeat discharge rows by n_expand (vectorized)
        rep_idx = np.repeat(disc2.index.to_numpy(), n_expand2)
        expanded = disc2.loc[rep_idx].copy()

        # Per-stay step k = 1..n_expand (vectorized via cumcount)
        expanded["_k"] = expanded.groupby(id_col).cumcount() + 1

        # Update time columns
        k = expanded["_k"].to_numpy()
        expanded[t_col] = expanded[t_col].to_numpy() + k
        expanded[grid_end_col] = expanded[grid_end_col] + k * step
        expanded[ref_time_col] = expanded[ref_time_col] + k * step

        # Z for expanded rows: default 0
        expanded[Z_col] = 0

        # Set Z=1 on death grid if death occurs after discharge (t_death > tD) and within follow-up
        t_death_rep = np.repeat(t_death2, n_expand2)
        expanded.loc[
            (t_death_rep != -1) & (expanded[t_col].to_numpy() == t_death_rep) & (t_death_rep > np.repeat(tD2, n_expand2)),
            Z_col
        ] = 1

        expanded.drop(columns=["_k"], inplace=True)

        # Append expanded rows once (fast)
        out = pd.concat([out, expanded], ignore_index=True)

    # -----------------------------------------
    # 5) Enforce: while A==0, Z must be NaN
    # -----------------------------------------
    out.loc[out[A_col] == 0, Z_col] = np.nan

    # Return sorted (sorting can cost time; drop sort if not needed downstream)
    return out.sort_values([id_col, t_col]).reset_index(drop=True)



