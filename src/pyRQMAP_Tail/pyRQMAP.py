# -*- coding: utf-8 -*-
"""
RQMAP — Python implementation of quantile mapping (RQUANT method).

This module is a Python port of the ``fitRQMAP`` / ``doRQMAP``
functions from the R package **qmap** (Gudmundsson et al.):

  Gudmundsson, L. (2016). qmap: Statistical transformations for post-processing
  climate model output. R package.
  https://doi.org/10.32614/CRAN.package.qmap

The underlying method is described in:

  Boe, J.; Terray, L.; Habets, F. & Martin, E. (2007). Statistical and
  dynamical downscaling of the Seine basin climate for hydro-meteorological
  studies. International Journal of Climatology, 27, 1643–1655.
  https://doi.org/10.1002/joc.1602

Python implementation: Petr Vohnicky (petr.vohnicky@unipd.it)
"""

import numpy as np
from scipy.stats import mstats
from scipy.interpolate import interp1d, CubicSpline

def fitRQMAP(obs, mod, wet_day=True, qstep=0.01, nlls=10, nboot=10, **kwargs):
    
    """
    Fit quantile mapping with ratio adjustment (RQUANT) for bias correction.

    This function estimates a transfer function that maps the distribution of 
    modeled values (`mod`) to that of observed values (`obs`). It accounts for 
    wet/dry day frequency differences and provides parameters for subsequent 
    correction with :func:`doRQMAP`.

    The method uses local linear regression on empirical quantiles of the input 
    distributions, optionally with bootstrap resampling, to obtain a smooth 
    mapping between model and observed quantiles.

    Parameters
    ----------
    obs : array-like
        Observed time series (1D). NaNs are ignored.
    mod : array-like
        Modeled time series (1D), same length as `obs` (if not, both are 
        resampled to common quantiles). NaNs are ignored.
    wet_day : bool or float, optional
        Defines treatment of dry/wet days:
            - If float, values < `wet_day` are treated as dry.
            - If True (default), values > 0 are wet; the wet-day threshold is 
              set to the smallest nonzero modeled value.
            - If False, no wet-day adjustment is applied.
    qstep : float, optional
        Step size for quantile grid (default 0.01 = 1% increments).
    nlls : int, optional
        Number of local neighbors to include in local linear regression. 
        Default is 10.
    nboot : int, optional
        Number of bootstrap samples for estimating fitted quantiles. 
        If >1, results are averaged over bootstraps. Default is 10.
    **kwargs : dict
        Additional arguments (currently unused).

    Returns
    -------
    fobj : dict
        Fitted quantile mapping object, with structure:
            - ``fobj['par']['modq']`` : ndarray of model quantiles (reference x-axis).
            - ``fobj['par']['fitq']`` : ndarray of fitted quantiles (mapped y-axis).
            - ``fobj['par']['slope']`` : array of shape (2, 1), slopes at lower and 
              upper distribution tails for extrapolation.
            - ``fobj['wet_day']`` : float or None, threshold defining wet days.
            - ``fobj['wet_ratio']`` : float, ratio of wet to dry days in observed data.
            - ``fobj['class']`` : list identifying the object type, e.g. ["fitRQMAP"].

    Notes
    -----
    - If `obs` and `mod` differ in length, both are resampled to a common grid 
      of empirical quantiles using Harrell–Davis quantile estimator 
      (alphap=1/3, betap=1/3).
    - Local linear regression is used to estimate the relationship between 
      model and observed quantiles at each grid point.
    - Bootstrap resampling (if nboot > 1) stabilizes the fitted curve by 
      averaging across resampled fits.
    - Negative fitted values are forced to zero if a wet-day threshold is applied.
    - The returned object `fobj` should be passed to :func:`doRQMAP` to 
      correct new model data.
    """
    
    ys = obs[~np.isnan(obs)]
    xs = mod[~np.isnan(mod)]
    
    if len(xs) != len(ys):
        hn = min(len(xs), len(ys))
        ys = mstats.mquantiles(ys, prob=np.linspace(0, 1, hn), alphap=1/3, betap=1/3)
        xs = mstats.mquantiles(xs, prob=np.linspace(0, 1, hn), alphap=1/3, betap=1/3)
        #ys = np.quantile(ys, np.linspace(0, 1, hn), method='median_unbiased')
        #xs = np.quantile(xs, np.linspace(0, 1, hn), method='median_unbiased')
    else:
        xs = np.sort(xs)
        ys = np.sort(ys)

    if isinstance(wet_day, (int, float)) and not isinstance(wet_day, bool):
        q0 = ys >= wet_day
        empirical_prob = np.sum(q0) / np.sum(~q0) if np.sum(~q0) > 0 else np.inf
        ys = ys[q0]
        xs = xs[q0]
    elif isinstance(wet_day, bool):
        if wet_day:
            q0 = ys > 0
            empirical_prob = np.sum(q0) / np.sum(~q0) if np.sum(~q0) > 0 else np.inf
            ys = ys[q0]
            xs = xs[q0]
            wet_day = xs[0]
        else:
            wet_day = None
    else:
        raise ValueError("'wet_day' should be 'numeric' or 'logical'")

    nn = len(ys)
    #newx = np.quantile(xs, np.arange(0, 1 + qstep, qstep), method='median_unbiased')
    
    # Compute evaluation points for local regression by taking empirical quantiles of `xs`.
    # `newx` contains x-values corresponding to evenly spaced quantile levels (e.g., 0%, 25%, ..., 100%),
    # using Harrell–Davis quantile estimator (alphap=1/3, betap=1/3).
    newx = mstats.mquantiles(xs, prob=np.arange(0, 1 + qstep, qstep), alphap=1/3, betap=1/3)
    # newx = np.quantile(xs, np.arange(0, 1 + qstep, qstep), method='median_unbiased')
    fit = np.full((len(newx), 2, nboot), np.nan)
    nlls2 = min(nlls, nn)

    for j in range(nboot):
        if nboot == 1:
            xss = np.sort(xs)
            yss = np.sort(ys)
        else:
            indices = np.random.choice(len(xs), size=nn, replace=False)
            xss = np.sort(xs[indices])
            yss = np.sort(ys[indices])

        for i in range(len(newx)):
            # Center x-values around the current evaluation point `newx[i]`
            # for local linear regression. This simplifies the regression
            # so that the intercept directly estimates the value at `newx[i]`.
            xc = xss - newx[i]
            # Distance to the nlls2-th nearest x-value (used to select local neighbors)
            mdist = np.sort(np.abs(xc))[nlls2 - 1]
            # Select data points whose distance from newx[i] is within the nlls2-th nearest distance
            k = np.abs(xc) <= mdist
            # Construct the design matrix for local linear regression:
            # column of ones for the intercept, and centered x-values for the slope term
            xc = np.column_stack((np.ones(np.sum(k)), xc[k]))
            # Compute the matrix product X^T * X for the local regression,
            # which is needed to solve the normal equations for linear regression coefficients.
            a = np.dot(xc.T, xc)

            # Calculate the determinant of matrix `a` (X^T * X) to check if it is invertible.
            # If determinant is near zero (matrix nearly singular), fall back to the simple local mean.
            if np.abs(d := a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) < 1e-10:
                # When matrix is singular or ill-conditioned, estimate fit at newx[i]
                # as the mean of the neighboring y-values.
                fit[i, 0, j] = np.mean(yss[k])
            else:
                # Otherwise, solve normal equations to get linear regression coefficients:
                # beta = (X^T X)^(-1) X^T y, computed here via Cramer's rule for 2x2 matrix.
                b = np.dot(xc.T, yss[k])
                fit[i, :, j] = [(b[0] * a[1, 1] - b[1] * a[0, 1]) / d, # Intercept (estimate at newx[i])
                                (b[1] * a[0, 0] - b[0] * a[1, 0]) / d] # Slope (local linear trend)
            

    if wet_day is not None:
        if np.any(fit[:, 0, :] < 0):
            fit[:, 0, :][fit[:, 0, :] < 0] = 0

    fitted = np.nanmean(fit[:, 0, :], axis=1) if nboot > 1 else fit[:, 0, :]
    slope = np.nanmean(fit[:, 1, :], axis=1) if nboot > 1 else fit[:, 1, :]
    slope[slope< 1e-10] = np.nan
    slope = slope[np.isfinite(slope)]
    slope = slope[np.array([0, -1])].reshape(-1, 1)

    ppar = {
        'modq': newx.reshape(-1, 1),
        'fitq': fitted.reshape(-1, 1),
        'slope': slope
    }
    op = {
        'par': ppar,
        'wet_day': wet_day,
        'wet_ratio': empirical_prob,
        "class": ["fitRQMAP"]
    }
    return op

def doRQMAP(x_given, 
                 fobj, 
                 slope_bound={'lower': 0, 'upper': float('inf')}, 
                 qmap_type='linear', 
                 slice_exted=True, **kwargs):
    """
    Apply quantile mapping with ratio adjustment (RQUANT) for bias correction 
    of precipitation or similar variables.
    
    This function uses a fitted quantile mapping object (`fobj`), produced by 
    :func:`fitRQMAP`, to correct the distribution of an input time series.
    It adjusts the wet/dry day ratio to match the fitted model and applies 
    interpolation or extrapolation depending on the chosen mapping type.
    
    Parameters
    ----------
    x : pandas.Series or numpy.ndarray
        Input time series of values (e.g., precipitation) to be bias-corrected.
        Must be 1D.
    fobj : dict
        Fitted quantile mapping object returned by :func:`fitRQMAP`.
        Contains:
            - ``fobj['par']['modq']`` : ndarray of model quantiles (reference x-axis).
            - ``fobj['par']['fitq']`` : ndarray of fitted/observed quantiles (mapped y-axis).
            - ``fobj['par']['slope']`` : array-like of shape (2, 1), local slopes 
              at lower and upper distribution edges, used for extrapolation.
            - ``fobj['wet_day']`` : float or None, wet-day threshold. Values below 
              this are considered dry days. If None, all days are wet.
            - ``fobj['wet_ratio']`` : float, ratio of wet to dry days in observed data.
            - ``fobj['class']`` : list identifying the object, e.g. ["fitRQMAP"].
    slope_bound : dict, optional
        Lower and upper bounds for extrapolation slopes. Default is 
        ``{'lower': 0, 'upper': float('inf')}``.
    qmap_type : {'linear', 'linear2', 'tricub'}, optional
        Type of quantile mapping to apply:
            - ``'linear'`` : Linear interpolation; extrapolation by subtracting 
              a fixed offset at the upper tail.
            - ``'linear2'`` : Linear interpolation; extrapolation using the local 
              slopes at lower and upper tails.
            - ``'tricub'`` : Cubic spline interpolation with natural boundary 
              conditions.
    slice_exted : bool, optional
        If True, when the simulated wet-day ratio is lower than the fitted wet ratio, 
        it can results in two scenarios:
            wet_day (min rain treshold) is defined not correctly and it is re-defined.
            or
            the model data are overall too dry, then zeros (dry days) are replaced with small random positive values 
            close to zero. 
        -> both scenarios it extends the lower slice of the distribution and increases the wet-day ratio.
        Default is True.
        Using on pooled stations should be treated with caution as pooled stations could already influence wet-dry ratio
        If fitRQMAP is trained on pooled stations, I recommend to check wet-dry ratio 
        on the stations and model before after applying
    **kwargs : dict
        Additional keyword arguments (currently unused).
    
    Returns
    -------
    out : numpy.ndarray
        Bias-corrected values, same length as input `x`.
    
    Notes
    -----
    - If `fobj['wet_day']` is not None, the function iteratively adjusts the 
      wet-day threshold in `x` until the wet/dry ratio matches `fobj['wet_ratio']`.
    - Dry days (`x < wet_day`) are always mapped to zero in the output.
    - Negative corrected values are forced to zero.
    - Extrapolation behavior depends on `qmap_type`:
        * 'linear': fixed offset adjustment beyond maximum quantile.
        * 'linear2': slope-based adjustment at tails.
        * 'tricub': smooth cubic spline extrapolation.
    """
    
    # Adjust slope boundaries
    x = x_given.copy() 
    fobj["par"]["slope"][0] = max(fobj["par"]["slope"][0], slope_bound["lower"])
    fobj["par"]["slope"][1] = min(fobj["par"]["slope"][1], slope_bound["upper"])
    fitQmap_wet_ratio = fobj['wet_ratio'] #this is fitted wet_ratio
    
    # Determine wet day threshold
    if fobj["wet_day"] is not None: 
        wet = x >= fobj["wet_day"] 
        wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf
        if wet_ratio >= fitQmap_wet_ratio:
            # Step 1: Sort the data
            sorted_x = x.sort_values().reset_index(drop=True)
            # Step 2: Find the first non-zero value in the sorted data
            first_non_zero_idx = sorted_x[sorted_x > 0].index[0]

            while wet_ratio > (fitQmap_wet_ratio+0.0015):
                first_non_zero_idx += 1
                new_wet_day = sorted_x.iloc[first_non_zero_idx]
                wet = x >= new_wet_day
                wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf
                
            
        elif wet_ratio < fitQmap_wet_ratio:
            if slice_exted == True:
                # Step 1: Sort the data
                sorted_x = x.sort_values().reset_index(drop=True)
                # Step 2: Find the first non-zero value in the sorted data
                first_non_zero_idx = sorted_x[sorted_x > 0].index[0]
                
                #check if fobj["wet_day"] is defined correctly
                # this is especially crucial if fitQMAP was applied to pooled stations
                # in such case, the fobj["wet_day"] (threshold) can be too large
                temp_wet_day = sorted_x.iloc[first_non_zero_idx]
                wet = x >= temp_wet_day
                wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf
                if wet_ratio >= fitQmap_wet_ratio:
                    # Step 1: Sort the data
                    sorted_x = x.sort_values().reset_index(drop=True)
                    # Step 2: Find the first non-zero value in the sorted data
                    first_non_zero_idx = sorted_x[sorted_x > 0].index[0]

                    while wet_ratio > (fitQmap_wet_ratio+0.0015):
                        first_non_zero_idx += 1
                        new_wet_day = sorted_x.iloc[first_non_zero_idx]
                        wet = x >= new_wet_day
                        wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf
                        
                else:
                    not_wet = x < fobj["wet_day"]
                    num_replacements = not_wet.sum()
                    unique_replacements = np.linspace(0.08, fobj["wet_day"]-0.01, num=num_replacements, 
                                                      endpoint=False,  dtype=np.float32)  # Unique values
                    # Shuffle to randomize assignment
                    np.random.shuffle(unique_replacements)
                    # Replace the 0 values with random small values close to 0 values
                    x.loc[not_wet] = unique_replacements.astype(np.float32)
                    # get unique value that correspond to the quantile 
                    
                    #reset 
                    sorted_x = x.sort_values().reset_index(drop=True)
                    while wet_ratio < (fitQmap_wet_ratio-0.0015):
                        first_non_zero_idx -= 1
                        new_wet_day = sorted_x.iloc[first_non_zero_idx]
                        wet = x >= new_wet_day
                        wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf
                        
                
    else:
        wet = np.ones_like(x, dtype=bool)
    
    out = np.full(len(x), np.nan)
    
    if qmap_type in ["linear", "linear2"]:
        interp_func = interp1d(fobj['par']['modq'][:, 0], # Original quantile points (input x-values for interpolation)
                               fobj['par']['fitq'][:, 0], # Fitted quantile values (output y-values for interpolation)
                               kind='linear',   # Use linear interpolation between points
                               bounds_error=False, # Allow extrapolation outside the original quantile range without error
                               fill_value = (fobj['par']['fitq'][:, 0][0], 
                                             fobj['par']['fitq'][:, 0][-1]),)
        out[wet] = interp_func(x[wet])
                 
        # 'linear': Use linear interpolation between modeled quantiles.
        # For values beyond the max quantile, extrapolate by subtracting a fixed offset (max_delta).
        # Slopes at edges are not used here.
        
        # 'linear2': Use linear interpolation within quantiles.
        # For values beyond quantile range, use local slopes at edges for linear extrapolation.
        # This preserves the trend at the edges, making extrapolation more realistic.
        
        # 'tricub': Use cubic spline interpolation for smooth, nonlinear mapping.
        # Extrapolation follows natural spline boundary conditions.
        
        if qmap_type == "linear2":
            if any(k := x > np.max(fobj['par']['modq'])):
                out[k] = np.max(fobj['par']['fitq']) + fobj['par']['slope'][1] * (x[k] - np.max(fobj['par']['modq']))
            if any(k := x < np.min(fobj['par']['modq'])):
                out[k] = np.min(fobj['par']['fitq']) + fobj['par']['slope'][0] * (x[k] - np.min(fobj['par']['modq']))
        elif qmap_type == "linear":
            nq = fobj['par']['modq'].shape[0]
            largex = x > fobj['par']['modq'][nq - 1, 0]
            if any(largex):
                max_delta = fobj['par']['modq'][nq - 1, 0] - fobj['par']['fitq'][nq - 1, 0]
                out[largex] = x[largex] - max_delta
    
    elif qmap_type == "tricub":
        sfun = CubicSpline(fobj['par']['modq'][:, 0], fobj['par']['fitq'][:, 0], bc_type='natural')
        out[wet] = sfun(x[wet])
    
    # not wet are pushed to 0 
    out[~wet] = 0
    if 'wet_day' in fobj and fobj['wet_day'] is not None:
        out[out < 0] = 0
    
    return out