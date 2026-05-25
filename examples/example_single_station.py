# -*- coding: utf-8 -*-
"""
Single-station example — pyQMAP_Tail
=====================================
Compares plain RQMAP vs RQMAP-Tail on one station paired with the colocated
ETH CPM grid point.

Workflow
--------
1. Load obs and CPM parquet files
2. Per-month: fit plain RQMAP and RQMAP-Tail, apply both corrections
3. Compute SMEV return levels (1h and 24h) for obs / raw / RQMAP / RQMAP-Tail
4. Plot return levels and bias per return period (two figures: 1h and 24h)

Available stations are listed in examples/data/stations_metadata.csv.
Set STATION below to any station_id from that file.

Notes
-----
Obs precipitation is thresholded at 0.2 mm (values below set to zero) to
ensure long-term consistency: the station network changed its minimum
detectable value from 0.2 mm to 0.1 mm around 2010, so enforcing 0.2 mm
throughout avoids an artificial wet-frequency increase in later years.
The CPM data is left unthresholded.
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

STATION = "AA_03000110"   # change to any station_id from examples/data/stations_metadata.csv
MODEL   = "ETH"           # ETH, KIT, CMCC, HCLIMcom, KNMI, CNRM

RETURN_PERIODS = [2, 5, 10, 20, 50, 100]
DATA_DIR = Path(__file__).parent / "data"

_meta = pd.read_csv(DATA_DIR / "stations_metadata.csv").set_index("station_id")
STATION_ELEV  = int(_meta.loc[STATION, "elev_dem"])
STATION_GROUP = _meta.loc[STATION, "group"].capitalize()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def drop_zero_months(df, time_col, val_col):
    """Drop entire year-months where all values are zero."""
    periods = df[time_col].dt.to_period("M")
    valid   = periods[df[val_col] > 0].unique()
    return df[periods.isin(valid)].reset_index(drop=True)


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
    """Returns dict with keys '60' and '1440', each a numpy array of return levels."""
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
# Load data and run
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"Station: {STATION}")

df_obs = pd.read_parquet(DATA_DIR / "obs_all.parquet",
                         filters=[("station_id", "==", STATION)])
df_cpm = pd.read_parquet(DATA_DIR / f"cpm_{MODEL}_all.parquet",
                         filters=[("station_id", "==", STATION)])

df_obs = df_obs.rename(columns={"time": "datetime", "precip_mm": "prec[mm]"})
df_cpm = df_cpm.rename(columns={"precip_mm": "pr_mm"})
df_obs["datetime"] = pd.to_datetime(df_obs["datetime"])
df_cpm["time"]     = pd.to_datetime(df_cpm["time"])
df_obs["prec[mm]"] = df_obs["prec[mm]"].where(df_obs["prec[mm]"] >= 0.2, 0.0)
df_obs = drop_zero_months(df_obs, "datetime", "prec[mm]")

print(f"  OBS: {df_obs['datetime'].min().date()} → {df_obs['datetime'].max().date()} "
      f"({df_obs['datetime'].dt.year.nunique()} yr)  |  "
      f"CPM: {df_cpm['time'].min().date()} → {df_cpm['time'].max().date()}")

# Per-month fit
fits_rqmap = {}
fits_tail  = {}

for month in range(1, 13):
    obs_m = df_obs[df_obs["datetime"].dt.month == month]["prec[mm]"]
    cpm_m = df_cpm[df_cpm["time"].dt.month     == month]["pr_mm"]

    fits_rqmap[month] = fitRQMAP(obs_m, cpm_m, wet_day=True, qstep=0.02, nboot=10)

    with contextlib.redirect_stdout(io.StringIO()):
        censor = get_optimal_threshold_multi(obs_m, cpm_m, make_plot=False)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        get_optimal_threshold_multi(cpm_m, cpm_m, make_plot=False)
    cpm_ok = "not pass" if "not passed" in buf.getvalue() else "pass"

    fit_t = fitRQMAP_Tail(obs_m, cpm_m, data_portion=[censor, 1],
                                qstep=0.02, nboot=10, wet_day=True)
    fit_t["censoring"] = [censor, 1]
    fits_tail[month]   = fit_t
    print(f"  Month {month:02d}: obs censoring = {censor:.3f} | CPM tail test: {cpm_ok}")

# Apply corrections
df_rqmap = apply_monthly(df_cpm, fits_rqmap, doRQMAP)
df_tail  = apply_monthly(df_cpm, fits_tail, doRQMAP_Tail,
                         extra_kwarg=lambda f: {"data_portion": f["censoring"], "slice_exted": True})

# SMEV return levels
print("  Computing SMEV return levels …")
obs_rl   = smev_return_levels(df_obs["datetime"].values, df_obs["prec[mm]"].values)
raw_rl   = smev_return_levels(df_cpm["time"].values,     df_cpm["pr_mm"].values)
rqmap_rl = smev_return_levels(df_rqmap["time"].values,   df_rqmap["pr_mm"].values)
tail_rl  = smev_return_levels(df_tail["time"].values,    df_tail["pr_mm"].values)

# Summary: bias at 20-year RP
rp20_idx = RETURN_PERIODS.index(20)
print(f"\n{'='*60}")
print(f"Bias at 20-year RP  —  {STATION}")
print(f"  {'':8}  {'Raw':>8}  {'RQMAP':>8}  {'RQMAP-Tail':>12}")
for dur, label in [("60", "1h"), ("1440", "24h")]:
    b_raw   = (raw_rl[dur][rp20_idx]   - obs_rl[dur][rp20_idx]) / obs_rl[dur][rp20_idx] * 100
    b_rqmap = (rqmap_rl[dur][rp20_idx] - obs_rl[dur][rp20_idx]) / obs_rl[dur][rp20_idx] * 100
    b_tail  = (tail_rl[dur][rp20_idx]  - obs_rl[dur][rp20_idx]) / obs_rl[dur][rp20_idx] * 100
    print(f"  [{label}]    {b_raw:>+8.1f}  {b_rqmap:>+8.1f}  {b_tail:>+12.1f}")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot_figure(duration_key, dur_label, fname):
    x = np.arange(len(RETURN_PERIODS))
    rp_labels = [str(r) for r in RETURN_PERIODS]
    tag  = f"{STATION}  |  {STATION_GROUP}  |  {STATION_ELEV} m"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.plot(x, obs_rl[duration_key],   "k-o",  linewidth=2,   markersize=7, label="OBS")
    ax.plot(x, raw_rl[duration_key],   color="#D7301F", linestyle="--", marker="s",
            linewidth=1.8, markersize=6, label="Raw CPM")
    ax.plot(x, rqmap_rl[duration_key], color="#74ADD1", linestyle="-",  marker="^",
            linewidth=1.8, markersize=6, label="RQMAP")
    ax.plot(x, tail_rl[duration_key],  color="#2171B5", linestyle="-",  marker="D",
            linewidth=1.8, markersize=6, label="RQMAP-Tail")
    ax.set_xticks(x); ax.set_xticklabels(rp_labels)
    ax.set_xlabel("Return period [yr]")
    ax.set_ylabel(f"{dur_label} return level [mm]")
    ax.set_title(f"{tag} — Return levels")
    ax.legend(); ax.grid(True, alpha=0.4)

    bias_raw   = (raw_rl[duration_key]   - obs_rl[duration_key]) / obs_rl[duration_key] * 100
    bias_rqmap = (rqmap_rl[duration_key] - obs_rl[duration_key]) / obs_rl[duration_key] * 100
    bias_tail  = (tail_rl[duration_key]  - obs_rl[duration_key]) / obs_rl[duration_key] * 100

    ax = axes[1]
    ax.axhline(0, color="red", linestyle="--", linewidth=1.2)
    ax.plot(x, bias_raw,   color="#D7301F", linestyle="--", marker="s",
            linewidth=1.8, markersize=6, label="Raw CPM")
    ax.plot(x, bias_rqmap, color="#74ADD1", linestyle="-",  marker="^",
            linewidth=1.8, markersize=6, label="RQMAP")
    ax.plot(x, bias_tail,  color="#2171B5", linestyle="-",  marker="D",
            linewidth=1.8, markersize=6, label="RQMAP-Tail")
    ax.set_xticks(x); ax.set_xticklabels(rp_labels)
    ax.set_xlabel("Return period [yr]")
    ax.set_ylabel("Bias [%]")
    ax.set_title(f"{tag} — Bias vs OBS")
    ax.legend(); ax.grid(True, alpha=0.4)

    fig.suptitle(f"{MODEL} CPM — RQMAP vs RQMAP-Tail ({dur_label})", fontweight="bold")
    plt.tight_layout()
    # plt.savefig(Path(__file__).parent / fname, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Saved → examples/{fname}")


plot_figure("60",   "1h",  "example_output_1h.png")
plot_figure("1440", "24h", "example_output_24h.png")
