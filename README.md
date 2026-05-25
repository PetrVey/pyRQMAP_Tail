# pyRQMAP_Tail

**pyRQMAP_Tail** is a Python package for bias correction of climate model precipitation using quantile mapping with an automatic tail correction. It is designed for correcting Convection Permitting Models (CPMs) or any gridded precipitation dataset against station observations, with particular focus on preserving the statistical behaviour of extreme events in high (sub daily) resolution.

The package provides two bias correction methods:

- **RQMAP** — Robust Quantile Mapping, a Python port of the `fitQmapRQUANT` / `doQmapRQUANT` functions from the R package [qmap](https://doi.org/10.32614/CRAN.package.qmap) (Gudmundsson, 2016), based on the method of Boe et al. (2007).
- **RQMAP-Tail** — RQMAP extended with an automatic tail correction. The body of the distribution is corrected by RQMAP, while the upper tail (above an automatically detected censoring threshold) is corrected by mapping through a fitted parametric tail distribution. This preserves the frequency and magnitude of rare extreme events that standard quantile mapping tends to underestimate.

The optimal censoring threshold (body/tail split) is determined automatically using the Weibull tail test of Marra et al. (2023), implemented in [pyTENAX](https://github.com/PetrVey/pyTENAX).

## Spatial application

The correction is trained at the station level against the co-located CPM grid point. When station density is sufficient, an **elevation-based pooling** strategy can be used: observations and CPM data from all stations within the same elevation band are concatenated before fitting, and the resulting model is then applied individually to each grid point. This pooling approach allows correction of CPM grid points that lie outside the training station network, provided they fall within an elevation group covered by the pooled model.

## Installation

```bash
pip install -e .
```

Or create a dedicated conda environment first:

```bash
conda env create -f environment.yml
conda activate pyRQMAP_Tail_env
pip install -e .
```

## Modules

| Module | Description |
|---|---|
| `pyRQMAP` | Core RQMAP: `fitRQMAP`, `doRQMAP` |
| `pyRQMAP_Tail` | RQMAP-Tail: `fitRQMAP_Tail`, `doRQMAP_Tail`, `get_optimal_threshold_multi` |
| `wbl_tail_test` | Weibull tail test (numba-optimised), ported from [pyTENAX](https://github.com/PetrVey/pyTENAX) |

## Quick start

See `examples/example_single_station.py` for a complete workflow: fitting both methods on a single station, computing SMEV return levels, and plotting the bias per return period.

## References

Boe, J.; Terray, L.; Habets, F. & Martin, E. (2007). Statistical and dynamical downscaling of the Seine basin climate for hydro-meteorological studies. *International Journal of Climatology*, 27, 1643–1655. https://doi.org/10.1002/joc.1602

Gudmundsson, L. (2016). qmap: Statistical transformations for post-processing climate model output. R package. https://doi.org/10.32614/CRAN.package.qmap

Marra, F.; Amponsah, W. & Papalexiou, S.M. (2023). Non-asymptotic Weibull tails explain the statistics of extreme daily precipitation. *Advances in Water Resources*, 173, 104388. https://doi.org/10.1016/j.advwatres.2023.104388

## Author

Petr Vohnicky (petr.vohnicky@unipd.it)
