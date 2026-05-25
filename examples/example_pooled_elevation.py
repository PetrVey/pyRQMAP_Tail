# -*- coding: utf-8 -*-
"""
Pooled-elevation example — pyQMAP_Tail
=======================================
Replicates the elevation-group pooling used in the original
BiasCorr_TESAF_framework: observations and CPM data from all stations in the
same elevation band are concatenated, one RQMAP / RQMAP-Tail model is trained
on the pooled series per calendar month, and then corrections are applied to
each station's CPM individually.  All three elevation groups are processed:
low (<1300 m, 15 stations), mid (1300–1800 m, 16 stations), high (>1800 m,
13 stations).  Station metadata is read from examples/data/stations_metadata.csv.

Notes
-----
Obs precipitation is thresholded at 0.2 mm (values below set to zero) to
ensure long-term consistency: the station network changed its minimum
detectable value from 0.2 mm to 0.1 mm around 2010, so enforcing 0.2 mm
throughout avoids an artificial wet-frequency increase in later years.
The CPM data is left unthresholded.

Workflow
--------
1. Load station metadata; group by elevation band
2. For each group: pool obs + CPM, fit RQMAP and RQMAP-Tail per month (qstep=0.002)
3. Apply group model to each station's CPM individually
4. Compute SMEV return levels (1h and 24h) per station
5. Print mean bias at 20-year RP per elevation group
6. Save boxplot: 3 rows (groups) x 2 columns (1h / 24h)
"""

import io
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from pyTENAX import smev

from pyRQMAP_Tail.pyRQMAP import fitRQMAP, doRQMAP
from pyRQMAP_Tail.pyRQMAP_Tail import (
    fitRQMAP_Tail,
    doRQMAP_Tail,
    get_optimal_threshold_multi,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODEL   = "ETH"   # ETH, KIT, CMCC, HCLIMcom, KNMI, CNRM

RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
DATA_DIR = Path(__file__).parent / "data"

GROUPS = [
    ("low",  "Low (<1300 m)"),
    ("mid",  "Mid (1300–1800 m)"),
    ("high", "High (>1800 m)"),
]

# Load station metadata
meta_df = pd.read_csv(DATA_DIR / "stations_metadata.csv")
group_stations = {g: meta_df[meta_df["group"] == g]["station_id"].tolist()
                  for g, _ in GROUPS}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_smev():
    return smev.SMEV(
        return_period=RETURN_PERIODS,
        durations=[60, 1440],
        time_resolution=60,
        min_event_duration=30,
        storm_separation_time=24,
        left_censoring=[0.9, 1],
        min_rain=0.1,
    )


def drop_zero_months(series, dt_series):
    mask = pd.Series(dt_series).dt.to_period("M")
    keep = mask.isin(mask[series > 0].unique())
    return series[keep.values], dt_series[keep.values]


def apply_monthly(df_cpm, fits, apply_fn, extra_kwarg=None):
    parts = []
    for month in range(1, 13):
        m = df_cpm[df_cpm["time"].dt.month == month]
        kwargs = extra_kwarg(fits[month]) if extra_kwarg else {}
        corr = apply_fn(m["pr_mm"], fits[month], qmap_type="linear", **kwargs)
        parts.append(pd.DataFrame({"time": m["time"].values, "pr_mm": corr}))
    return (pd.concat(parts)
              .sort_values("time")
              .reset_index(drop=True)
              .assign(time=lambda d: pd.to_datetime(d["time"]).dt.tz_localize(None).astype("datetime64[ns]")))


def smev_return_levels(time_arr, prec_arr):
    s = make_smev()
    df = pd.DataFrame({"value": np.array(prec_arr, dtype=float)},
                      index=pd.to_datetime(time_arr))
    df["value"] = df["value"].where(df["value"] >= 0.1, 0.0)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    df = s.remove_incomplete_years(df, name_col="value")
    clean_arr   = df["value"].to_numpy()
    clean_dates = df.index.to_numpy().astype("datetime64[ns]")
    idx_oe = s.get_ordinary_events(data=clean_arr, dates=clean_dates, check_gaps=True)
    _, arr_dates, n_per_year = s.remove_short(idx_oe)
    n_oe = n_per_year["index"].mean()
    dict_oe, _ = s.get_ordinary_events_values(data=clean_arr, dates=clean_dates,
                                               arr_dates_oe=arr_dates)
    out = {}
    for dur in ["60", "1440"]:
        P = dict_oe[dur]["ordinary"].to_numpy()
        shape, scale = s.estimate_smev_parameters(P, s.left_censoring)
        out[dur] = np.array(s.smev_return_values(RETURN_PERIODS, shape, scale, n_oe))
    return out


# ---------------------------------------------------------------------------
# Main loop — one elevation group at a time
# ---------------------------------------------------------------------------
# results[group_key][station_id][dur] = {bias_raw, bias_rqmap, bias_tail, ...}
results = {g: {} for g, _ in GROUPS}

obs_all = pd.read_parquet(DATA_DIR / "obs_all.parquet")
cpm_all = pd.read_parquet(DATA_DIR / f"cpm_{MODEL}_all.parquet")

for group_key, group_label in GROUPS:
    stations = group_stations[group_key]
    print(f"\n{'='*60}")
    print(f"Group: {group_label}  ({len(stations)} stations)")

    # Load data for this group
    obs_grp = obs_all[obs_all["station_id"].isin(stations)].copy()
    cpm_grp = cpm_all[cpm_all["station_id"].isin(stations)].copy()
    obs_grp["time"] = pd.to_datetime(obs_grp["time"])
    cpm_grp["time"] = pd.to_datetime(cpm_grp["time"])
    obs_grp["precip_mm"] = obs_grp["precip_mm"].where(obs_grp["precip_mm"] >= 0.2, 0.0)

    # Per-month pooled fit
    fits_rqmap = {}
    fits_tail  = {}

    for month in range(1, 13):
        obs_parts, cpm_parts = [], []
        for st in stations:
            obs_m = obs_grp[(obs_grp["station_id"] == st) &
                            (obs_grp["time"].dt.month == month)]["precip_mm"].values
            dt_m  = obs_grp[(obs_grp["station_id"] == st) &
                            (obs_grp["time"].dt.month == month)]["time"].values
            obs_m, _ = drop_zero_months(obs_m, dt_m)
            obs_parts.append(pd.Series(obs_m))
            cpm_parts.append(
                cpm_grp[(cpm_grp["station_id"] == st) &
                        (cpm_grp["time"].dt.month == month)]["precip_mm"]
            )

        pooled_obs = pd.concat(obs_parts, ignore_index=True)
        pooled_cpm = pd.concat(cpm_parts, ignore_index=True)

        buf_obs = io.StringIO()
        with contextlib.redirect_stdout(buf_obs):
            censor = get_optimal_threshold_multi(pooled_obs, 
                                                 pooled_cpm, 
                                                 make_plot=False)
        obs_buf = buf_obs.getvalue()
        obs_ok = "not pass" if "not passed" in obs_buf else "pass"

        buf_cpm = io.StringIO()
        with contextlib.redirect_stdout(buf_cpm):
            get_optimal_threshold_multi(pooled_cpm,
                                        pooled_cpm,
                                        make_plot=False)
        cpm_ok = "not pass" if "not passed" in buf_cpm.getvalue() else "pass"

        print(f"  Month {month:02d}: obs censoring = {censor:.3f} (tail test: {obs_ok}) | CPM tail test: {cpm_ok}")
        for line in obs_buf.splitlines():
            if "Re-test" in line:
                print(f"    {line.strip()}")

        fits_rqmap[month] = fitRQMAP(pooled_obs, pooled_cpm,
                                           wet_day=True, qstep=0.002, nboot=10)
        fit_t = fitRQMAP_Tail(pooled_obs, pooled_cpm, data_portion=[censor, 1],
                                    qstep=0.002, nboot=10, wet_day=True)
        fit_t["censoring"] = [censor, 1]
        fits_tail[month]   = fit_t

    # Apply to each station and compute SMEV
    for st in stations:
        obs_st = obs_grp[obs_grp["station_id"] == st].copy()
        cpm_st = (cpm_grp[cpm_grp["station_id"] == st]
                  .rename(columns={"precip_mm": "pr_mm"})
                  .reset_index(drop=True))

        df_rqmap = apply_monthly(cpm_st, fits_rqmap, doRQMAP)
        df_tail  = apply_monthly(cpm_st, fits_tail, doRQMAP_Tail,
                                 extra_kwarg=lambda f: {"data_portion": f["censoring"],
                                                        "slice_exted": True})

        obs_rl   = smev_return_levels(obs_st["time"].values,   obs_st["precip_mm"].values)
        raw_rl   = smev_return_levels(cpm_st["time"].values,   cpm_st["pr_mm"].values)
        rqmap_rl = smev_return_levels(df_rqmap["time"].values, df_rqmap["pr_mm"].values)
        tail_rl  = smev_return_levels(df_tail["time"].values,  df_tail["pr_mm"].values)

        results[group_key][st] = {
            dur: {
                "bias_raw":   (raw_rl[dur]   - obs_rl[dur]) / obs_rl[dur] * 100,
                "bias_rqmap": (rqmap_rl[dur] - obs_rl[dur]) / obs_rl[dur] * 100,
                "bias_tail":  (tail_rl[dur]  - obs_rl[dur]) / obs_rl[dur] * 100,
            }
            for dur in ["60", "1440"]
        }
        print(f"  {st}: done")


# ---------------------------------------------------------------------------
# Summary: mean bias at 20-year RP per group
# ---------------------------------------------------------------------------
rp20_idx = RETURN_PERIODS.index(20)
print(f"\n{'='*60}")
print(f"Mean bias at 20-year RP  —  {MODEL}")
print(f"  {'Group':<22}  {'Dur':>4}  {'Raw':>8}  {'RQMAP':>8}  {'RQMAP-Tail':>12}")
for group_key, group_label in GROUPS:
    stations = list(results[group_key].keys())
    for dur, label in [("60", "1h"), ("1440", "24h")]:
        mean_raw   = np.mean([results[group_key][s][dur]["bias_raw"][rp20_idx]   for s in stations])
        mean_rqmap = np.mean([results[group_key][s][dur]["bias_rqmap"][rp20_idx] for s in stations])
        mean_tail  = np.mean([results[group_key][s][dur]["bias_tail"][rp20_idx]  for s in stations])
        print(f"  {group_label:<22}  [{label}]  {mean_raw:>+8.1f}  {mean_rqmap:>+8.1f}  {mean_tail:>+12.1f}")


# ---------------------------------------------------------------------------
# Boxplot: 3 rows (groups) x 2 columns (1h / 24h)
# ---------------------------------------------------------------------------
colors       = {"Raw": "#D7301F", "RQMAP": "#74ADD1", "RQMAP-Tail": "#2171B5"}
method_order = ["Raw", "RQMAP", "RQMAP-Tail"]
method_keys  = ["bias_raw", "bias_rqmap", "bias_tail"]
x     = np.arange(len(RETURN_PERIODS))
width = 0.22
rp_labels = [str(r) for r in RETURN_PERIODS]

fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharey=False)

for row, (group_key, group_label) in enumerate(GROUPS):
    stations = list(results[group_key].keys())
    for col, (dur, dur_label) in enumerate([("60", "1h"), ("1440", "24h")]):
        ax = axes[row, col]
        for k, (method, key) in enumerate(zip(method_order, method_keys)):
            data_per_rp = [
                [results[group_key][s][dur][key][i] for s in stations]
                for i in range(len(RETURN_PERIODS))
            ]
            positions = x + (k - 1) * width
            bp = ax.boxplot(data_per_rp, positions=positions, widths=width * 0.85,
                            patch_artist=True, manage_ticks=False,
                            medianprops=dict(color="black", linewidth=1.8),
                            whiskerprops=dict(linewidth=1.2),
                            capprops=dict(linewidth=1.2),
                            flierprops=dict(marker="o", markersize=4, linestyle="none",
                                            markeredgecolor=colors[method]))
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[method])
                patch.set_alpha(0.7)

        ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
        ax.set_xticks(x)
        ax.set_xticklabels(rp_labels)
        ax.set_ylabel("Bias [%]")
        ax.set_title(f"{group_label}  —  {dur_label}  (n={len(stations)})")
        ax.grid(True, alpha=0.4, axis="y")

        if row == 0 and col == 0:
            handles = [plt.Rectangle((0, 0), 1, 1, facecolor=colors[m], alpha=0.7)
                       for m in method_order]
            ax.legend(handles, method_order, loc="upper left")

for ax in axes[-1, :]:
    ax.set_xlabel("Return period [yr]")

fig.suptitle(f"{MODEL} CPM — Pooled bias by elevation group", fontweight="bold")
plt.tight_layout()
out_fname = f"example_pooled_boxplot_{MODEL}.png"
# plt.savefig(Path(__file__).parent / out_fname, dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved → examples/{out_fname}")
