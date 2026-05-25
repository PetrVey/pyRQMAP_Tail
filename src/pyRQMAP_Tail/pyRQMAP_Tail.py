# -*- coding: utf-8 -*-
"""
RQMAP with tail correction.

Public API
----------
fitRQMAP_Tail          -- fit the quantile mapping model
doRQMAP_Tail           -- apply the fitted model
get_optimal_threshold_multi -- auto-detect optimal tail censoring threshold
get_pval_Marra_wbl_test     -- Marra et al. p-value tail test

Author: Petr Vohnicky (petr.vohnicky@unipd.it)
"""

import numpy as np
from scipy.interpolate import interp1d, CubicSpline
import math
import statsmodels.api as sm
from scipy.stats import weibull_min, kstest, mstats
import matplotlib.pyplot as plt
import pandas as pd
from pyRQMAP_Tail.wbl_tail_test import estimate_smev_param_without_AM, create_synthetic_records
from pyRQMAP_Tail.wbl_tail_test import check_confidence_interval, find_optimal_threshold, plot_curve


def fitRQMAP_Tail(obs, mod, wet_day=True, qstep=0.01, nlls=10, nboot=10,
                       data_portion=[0.8, 1], **kwargs):

    ys = obs[~np.isnan(obs)]
    xs = mod[~np.isnan(mod)]

    if len(xs) != len(ys):
        hn = min(len(xs), len(ys))
        ys = mstats.mquantiles(ys, prob=np.linspace(0, 1, hn), alphap=1/3, betap=1/3)
        xs = mstats.mquantiles(xs, prob=np.linspace(0, 1, hn), alphap=1/3, betap=1/3)
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

    dict_stack = {"ys": ys, "xs": xs}
    weibul_tail_param = {}
    to_use_idx = {}

    for name, value in dict_stack.items():
        ECDF = (np.arange(1, 1 + len(value)) / (1 + len(value)))
        fidx = max(1, math.floor((len(value)) * data_portion[0]))
        tidx = math.ceil(len(value) * data_portion[1])
        to_use = np.arange(fidx - 1, tidx)
        to_use_array = value[to_use]

        X = (np.log(np.log(1 / (1 - ECDF[to_use]))))
        Y = (np.log(to_use_array))
        X = sm.add_constant(X)
        model = sm.OLS(Y, X)
        results = model.fit()
        param = results.params

        slope = param[1]
        intercept = param[0]
        shape = 1 / slope
        scale = np.exp(intercept)

        weibul_tail_param[name] = [shape, scale]
        to_use_idx[name] = to_use

    ys_body = ys
    xs_body = xs

    nn = len(ys_body)
    newx = mstats.mquantiles(xs_body, prob=np.arange(0, 1 + qstep, qstep), alphap=1/3, betap=1/3)
    fit = np.full((len(newx), 2, nboot), np.nan)
    nlls2 = min(nlls, nn)

    newx_second_to_last = newx[-2]
    prob_newx_second_to_last = np.linspace(0, 1, len(newx))[-2]
    larger_than_ = xs[xs > newx_second_to_last]
    if len(larger_than_) < 2:
        raise ValueError(
            f"Only {len(larger_than_)} data point(s) fall above the second-to-last quantile "
            f"({newx_second_to_last:.4f}, i.e. the {prob_newx_second_to_last*100:.1f}th percentile) — "
            f"need at least 2 to fit the upper tail. "
            f"This typically means qstep is too small relative to your sample size "
            f"({len(xs)} wet values): with qstep={qstep} you have {len(newx)} quantiles, "
            f"placing newx[-2] at the {(1-qstep)*100:.1f}th percentile where almost no data exist. "
            f"Use a coarser qstep (e.g. qstep=0.01) or provide more data."
        )
    larger_than_ = np.insert(larger_than_, 0, newx_second_to_last)
    n_tail = len(larger_than_)
    prob_tail = np.linspace(prob_newx_second_to_last, 1, n_tail)

    y = np.log(1 - prob_tail[:-1])
    log_prob_newx_second_to_last = np.log(1 - prob_newx_second_to_last)
    coef = np.polyfit(larger_than_[:-1] - newx_second_to_last, y - log_prob_newx_second_to_last, 1)
    upper_tail_fit_mod = {
        "a": coef[0],
        "b": np.log(1 - prob_newx_second_to_last) - coef[0] * newx_second_to_last,
        "p_start": prob_newx_second_to_last,
        "p_start_val": newx_second_to_last,
    }

    for j in range(nboot):
        if nboot == 1:
            xss = np.sort(xs_body)
            yss = np.sort(ys_body)
        else:
            indices = np.random.choice(len(xs_body), size=nn, replace=False)
            xss = np.sort(xs_body[indices])
            yss = np.sort(ys_body[indices])

        for i in range(len(newx)):
            xc = xss - newx[i]
            mdist = np.sort(np.abs(xc))[nlls2 - 1]
            k = np.abs(xc) <= mdist
            xc = np.column_stack((np.ones(np.sum(k)), xc[k]))
            a = np.dot(xc.T, xc)

            if np.abs(d := a[0, 0] * a[1, 1] - a[0, 1] * a[1, 0]) < 1e-10:
                fit[i, 0, j] = np.mean(yss[k])
            else:
                b = np.dot(xc.T, yss[k])
                fit[i, :, j] = [(b[0] * a[1, 1] - b[1] * a[0, 1]) / d,
                                (b[1] * a[0, 0] - b[0] * a[1, 0]) / d]

    if wet_day is not None:
        if np.any(fit[:, 0, :] < 0):
            fit[:, 0, :][fit[:, 0, :] < 0] = 0

    fitted = np.nanmean(fit[:, 0, :], axis=1) if nboot > 1 else fit[:, 0, :]
    slope = np.nanmean(fit[:, 1, :], axis=1) if nboot > 1 else fit[:, 1, :]
    slope[slope < 1e-10] = 0
    slope = slope[np.isfinite(slope)]
    slope = slope[np.array([0, -1])].reshape(-1, 1)

    ppar = {
        'modq': newx.reshape(-1, 1),
        'fitq': fitted.reshape(-1, 1),
        'slope': slope,
        'tail': weibul_tail_param,
        'data_portion': data_portion,
        'mod_emp_upper_tail': upper_tail_fit_mod,
    }
    op = {
        'par': ppar,
        'wet_day': wet_day,
        'wet_ratio': empirical_prob,
        "class": ["fitRQMAP_Tail"],
    }
    return op


def doRQMAP_Tail(x, fobj, slope_bound={'lower': 0, 'upper': float('inf')},
                      qmap_type='linear', slice_exted=True, fake_storms=True, **kwargs):

    x = x.copy()
    fobj["par"]["slope"][0] = max(fobj["par"]["slope"][0], slope_bound["lower"])
    fobj["par"]["slope"][1] = min(fobj["par"]["slope"][1], slope_bound["upper"])
    fitQmap_wet_ratio = fobj['wet_ratio']

    if fobj["wet_day"] is not None:
        wet = x >= fobj["wet_day"]
        wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf

        if wet_ratio >= fitQmap_wet_ratio:
            sorted_x = x.sort_values().reset_index(drop=True)
            first_non_zero_idx = sorted_x[sorted_x > 0].index[0]
            while wet_ratio > (fitQmap_wet_ratio + 0.0015):
                first_non_zero_idx += 1
                new_wet_day = sorted_x.iloc[first_non_zero_idx]
                wet = x >= new_wet_day
                wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf

        elif wet_ratio < fitQmap_wet_ratio:
            if slice_exted:
                sorted_x = x.sort_values().reset_index(drop=True)
                first_non_zero_idx = sorted_x[sorted_x > 0].index[0]
                temp_wet_day = sorted_x.iloc[first_non_zero_idx]
                wet = x >= temp_wet_day
                wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf

                if wet_ratio >= fitQmap_wet_ratio:
                    sorted_x = x.sort_values().reset_index(drop=True)
                    first_non_zero_idx = sorted_x[sorted_x > 0].index[0]
                    while wet_ratio > (fitQmap_wet_ratio + 0.0015):
                        first_non_zero_idx += 1
                        new_wet_day = sorted_x.iloc[first_non_zero_idx]
                        wet = x >= new_wet_day
                        wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf
                else:
                    not_wet = x < fobj["wet_day"]
                    num_replacements = not_wet.sum()
                    unique_replacements = np.linspace(0.08, fobj["wet_day"] - 0.01,
                                                      num=num_replacements,
                                                      endpoint=False, dtype=np.float32)
                    np.random.shuffle(unique_replacements)
                    x.loc[not_wet] = unique_replacements.astype(np.float32)
                    sorted_x = x.sort_values().reset_index(drop=True)
                    while wet_ratio < (fitQmap_wet_ratio - 0.0015):
                        first_non_zero_idx -= 1
                        new_wet_day = sorted_x.iloc[first_non_zero_idx]
                        wet = x >= new_wet_day
                        wet_ratio = np.sum(wet) / np.sum(~wet) if np.sum(~wet) > 0 else np.inf
    else:
        wet = np.ones_like(x, dtype=bool)

    def F_tail(x, a, b):
        return 1 - np.exp(a * x + b)

    model_upper_tail = fobj['par']['mod_emp_upper_tail']
    upper_tail_censor = model_upper_tail["p_start_val"]
    wet_upper_tail_mask = (x >= upper_tail_censor) & wet
    quantile_censor = x[wet].quantile(fobj["par"]["data_portion"][0])
    wet_tail_mask = (x > quantile_censor) & (x < upper_tail_censor) & wet
    wet_body_mask = wet & ~wet_tail_mask & ~wet_upper_tail_mask
    wet_tail = wet_tail_mask.copy()
    wet_upper_tail = wet_upper_tail_mask.copy()
    wet_body = wet_body_mask.copy()
    eps = 1e-12

    out = np.full(len(x), np.nan)

    if qmap_type in ["linear", "linear2"]:
        interp_func = interp1d(fobj['par']['modq'][:, 0],
                               fobj['par']['fitq'][:, 0],
                               kind='linear',
                               bounds_error=False,
                               fill_value=(fobj['par']['fitq'][:, 0][0],
                                           fobj['par']['fitq'][:, 0][-1]))
        out[wet_body] = interp_func(x[wet_body])

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

        shape_obs, scale_obs = fobj['par']["tail"]['ys']
        p_grid = np.linspace(0, 1, len(fobj['par']['modq'][:, 0]))
        Fmod_interp = interp1d(fobj['par']['modq'][:, 0], p_grid,
                               kind='linear', bounds_error=False,
                               fill_value=(0.0, 1.0))
        mod_cdf = Fmod_interp(x[wet_tail])
        out[wet_tail] = weibull_min.ppf(mod_cdf, c=shape_obs, scale=scale_obs)

        if np.sum(wet_upper_tail) > 0:
            mod_cdf_upper_tail = F_tail(x[wet_upper_tail], model_upper_tail["a"], model_upper_tail["b"])
            mod_cdf_upper_tail = np.clip(mod_cdf_upper_tail, None, 1 - eps)
            out[wet_upper_tail] = weibull_min.ppf(mod_cdf_upper_tail, c=shape_obs, scale=scale_obs)

    elif qmap_type == "tricub":
        sfun = CubicSpline(fobj['par']['modq'][:, 0], fobj['par']['fitq'][:, 0], bc_type='natural')
        out[wet] = sfun(x[wet])

    out[~wet] = 0
    if 'wet_day' in fobj and fobj['wet_day'] is not None:
        out[out < 0] = 0

    return out


def get_pval_Marra_wbl_test(ys, xs, make_plot=True, exclude_q99=True):

    if len(xs) != len(ys):
        hn = min(len(xs), len(ys))
        ys = mstats.mquantiles(ys, prob=np.linspace(0, 1, hn), alphap=1/3, betap=1/3)
        xs = mstats.mquantiles(xs, prob=np.linspace(0, 1, hn), alphap=1/3, betap=1/3)
    else:
        xs = np.sort(xs)
        ys = np.sort(ys)

    q0 = ys > 0
    ys = ys[q0]
    xs = xs[q0]

    seed_random = 42
    synthetic_records_amount = 500
    p_confidence = 0.2

    q99 = np.quantile(ys, 0.99)
    mask = ys > q99
    idx = np.where(mask)
    annual_max = ys[mask].tolist()
    annual_max_indexes = idx[0].tolist() if exclude_q99 else []
    record_size = len(ys)

    censor_values = np.arange(0.89, 0.99, 0.01)
    p_out_dicts_lst = []
    shape_scale_dict = {}

    for censor_value in censor_values:
        censor_value = censor_value.round(2)
        try:
            shape, scale = estimate_smev_param_without_AM(ys, censor_value, annual_max_indexes)
        except Exception as e:
            print(f"Error occurred: {e}")
            print("SMEV parameters cannot be estimated — likely too few events after censoring.")

        records_df = create_synthetic_records(seed_random, synthetic_records_amount,
                                              record_size, shape, scale)
        p_out_dicts_lst = check_confidence_interval(annual_max_indexes, records_df,
                                                    p_confidence, annual_max,
                                                    censor_value, p_out_dicts_lst)
        shape_scale_dict[censor_value.round(2)] = [shape, scale]

    optimal_threshold, range_of_optimal = find_optimal_threshold(p_out_dicts_lst, p_confidence)
    if optimal_threshold != 1:
        estimated_params = shape_scale_dict[optimal_threshold]
    else:
        estimated_params = None

    if make_plot:
        plot_curve(p_out_dicts_lst, p_confidence, optimal_threshold)

    cv_dict = {}
    qs = list(shape_scale_dict.keys())
    for i in range(1, len(qs) - 1):
        q_center = qs[i]
        shapes = np.array([shape_scale_dict[qs[i - 1]][0],
                           shape_scale_dict[qs[i]][0],
                           shape_scale_dict[qs[i + 1]][0]])
        if np.any(~np.isfinite(shapes)):
            cv_dict[q_center] = np.inf
        else:
            cv_dict[q_center] = np.std(shapes) / np.mean(shapes)

    return p_out_dicts_lst, cv_dict


def align_dict_to_fractions(d, fractions):
    """Align a dict (or list-of-dicts) of values to a fixed fractions array."""
    fractions = [round(f, 2) for f in fractions]
    if isinstance(d, list):
        aligned = []
        for f in fractions:
            val = next((v for dic in d if round(list(dic.keys())[0], 2) == f
                        for v in [list(dic.values())[0]]), np.nan)
            aligned.append(val)
        return np.array(aligned)
    else:
        d_rounded = {round(k, 2): v for k, v in d.items()}
        return np.array([d_rounded.get(f, np.nan) for f in fractions])


def get_optimal_threshold_multi(ys, xs, make_plot=False):

    wet_day = True
    if len(xs) != len(ys):
        hn = min(len(xs), len(ys))
        ys = mstats.mquantiles(ys, prob=np.linspace(0, 1, hn), alphap=1/3, betap=1/3)
        xs = mstats.mquantiles(xs, prob=np.linspace(0, 1, hn), alphap=1/3, betap=1/3)
    else:
        xs = np.sort(xs)
        ys = np.sort(ys)

    if isinstance(wet_day, (int, float)) and not isinstance(wet_day, bool):
        q0 = ys >= wet_day
        ys = ys[q0]
        xs = xs[q0]
    elif isinstance(wet_day, bool):
        if wet_day:
            q0 = ys > 0
            ys = ys[q0]
            xs = xs[q0]
        else:
            wet_day = None
    else:
        raise ValueError("'wet_day' should be 'numeric' or 'logical'")

    p_confidence = 0.2
    fractions = np.arange(0.89, 0.98, 0.01)

    p_out_dicts, cv_dict_q99 = get_pval_Marra_wbl_test(ys, 
                                                       xs,
                                                       make_plot=make_plot)

    p_out_aligned = align_dict_to_fractions(p_out_dicts, fractions)
    cv_q99_aligned = align_dict_to_fractions(cv_dict_q99, fractions)

    valid_mask = p_out_aligned <= p_confidence
    fractions_filtered = fractions[valid_mask]
    cv_q99_filtered = cv_q99_aligned[valid_mask]
    p_out_filtered = p_out_aligned[valid_mask]

    p_out_rank = pd.Series(p_out_filtered, index=fractions_filtered).rank(method="min")
    cv_q99_rank = pd.Series(cv_q99_filtered, index=fractions_filtered).rank(method="min")

    rank_df = pd.DataFrame({
        "p_out_Marra": p_out_rank,
        "CV_q99": cv_q99_rank,
    }, index=fractions)
    rank_df['total_rank'] = rank_df.mean(axis=1, skipna=False)

    ranked = pd.DataFrame({"ys": rank_df['total_rank']})
    ranked['total_rank'] = ranked.mean(axis=1).where(~ranked.isna().any(axis=1))

    if ranked['total_rank'].notna().any():
        best_fraction = round(ranked['total_rank'].idxmin(skipna=True), 2)
    else:
        best_fraction = 0.97
        print("Tail test not passed, defaulting to highest censor (0.97)")

        p_out_with_q99, _ = get_pval_Marra_wbl_test(ys, ys, make_plot=False, exclude_q99=False)
        _, range_with_q99 = find_optimal_threshold(p_out_with_q99, p_confidence)
        if range_with_q99:
            print("  → Re-test WITH q99 in fit: PASSED — tail is Weibull when all data included")
        else:
            print("  → Re-test WITH q99 in fit: also rejected")

        if make_plot:
            _plot_weibull_tail_diagnostic(ys, best_fraction)

    return best_fraction


def _plot_weibull_tail_diagnostic(ys, fraction):
    ys_sorted = np.sort(ys[ys > 0])
    n = len(ys_sorted)
    ecdf = np.arange(1, n + 1) / (n + 1)

    fidx = int(n * fraction)
    tail_y = ys_sorted[fidx:]
    tail_ecdf = ecdf[fidx:]

    X = np.log(np.log(1 / (1 - tail_ecdf)))
    Y = np.log(tail_y)
    results = sm.OLS(Y, sm.add_constant(X)).fit()
    intercept, slope = results.params
    Y_fit = intercept + slope * X

    _, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(X, Y, s=18, color="steelblue", alpha=0.7, label=f"Data (top {int((1-fraction)*100)}%)")
    ax.plot(X, Y_fit, color="red", linewidth=1.5,
            label=f"Weibull fit  (shape={1/slope:.2f})")
    ax.set_xlabel("log(log(1 / (1 - ECDF)))")
    ax.set_ylabel("log(precipitation)")
    ax.set_title(f"Weibull tail check at q{int(fraction*100)} — test rejected but tail looks linear")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
