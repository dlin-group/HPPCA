# Simulation Code

This folder contains the simulation code for the HPPCA manuscript and Web
Appendix. The real data analysis code is not included here because the real data
cannot be redistributed. The reusable method implementation is in `src/hppca/`;
the scripts in this folder call that package rather than duplicating the method
code.

All scripts use relative default paths under `sim_results/`. No original local
machine paths are required.

## Folder Map

### `section_3_1_3_4_hppca/`

Corresponds to manuscript Sections 3.1 and 3.4 and Web Appendix Section C.3.

- `simulate_and_fit_hppca.py`: simulates data from the HPPCA model, fits HPPCA,
  and writes per-run result CSVs, filled data, and latent `Z` tables.
- `summarize_simulate_and_fit_hppca_init.R`: combines per-run HPPCA result CSVs
  and creates summary tables for initialization and estimation performance.

### `section_3_2_method_comparison/`

Corresponds to manuscript Section 3.2.

- `simulate_and_fit_comparison_models.py`: simulates comparison-study data and
  fits PPCA and mFPCA. The older MissMDA/MLSCA path and subject-aggregated PPCA
  helper are intentionally omitted from this package version.
- `summarize_comparison_export.R`: combines per-run comparison result CSVs and
  merges them with HPPCA simulation summaries.
- `plot_comparison_methods_avgMSE.py`: creates the missing-entry MSE comparison
  plot for HPPCA, PPCA, and mFPCA.

### `section_3_3_lds/`

Corresponds to manuscript Section 3.3 and Web Appendix Section C.1.

- `simulate_and_fit_lds_diag_comparison.py`: simulates the LDS scenario and fits
  HPPCA, PPCA, and mFPCA.
- `summarize_results_lds_diag.py`: combines per-run LDS comparison result CSVs.
- `plot_comparison_methods_avgMSE_lds.py`: creates the LDS missing-entry MSE
  comparison plot.

### `web_appendix_c2_lds_ar2/`

Corresponds to Web Appendix Section C.2.

- `simulate_and_fit_lds_diag_comparison_ar2.py`: simulates the AR(2) LDS
  scenario and fits HPPCA, PPCA, and mFPCA.
- `summarize_results_lds_ar2.py`: combines per-run AR(2) LDS comparison result
  CSVs.
- `plot_comparison_methods_avgMSE_lds_ar2.py`: creates the AR(2) LDS
  missing-entry MSE comparison plot.

## Notes

- Large simulation grids should be run as batch jobs by varying seeds and
  scenario parameters.
- Some comparison scripts require optional dependencies such as `pyppca`,
  `FDApy`, `seaborn`, and R packages including `dplyr`, `readr`, `purrr`, and
  `stringr`.

## External Software Citations

The comparison simulation scripts use `pyppca` for PPCA and `FDApy` for mFPCA.

```bibtex
@misc{green2019pyppca,
  author       = {Green, Sheridan},
  title        = {{pyppca}: Probabilistic PCA with Missing Values},
  year         = {2019},
  howpublished = {Python package, version 0.0.4},
  url          = {https://pypi.org/project/pyppca/}
}

@article{golovkine_2024_fdapy_paper,
  title   = {{{FDApy}}: A {{Python}} Package for Functional Data},
  author  = {Golovkine, Steven},
  date    = {2025-03-04},
  journal = {Journal of Open Source Software},
  volume  = {10},
  year    = {2025},
  number  = {107},
  pages   = {7526},
  issn    = {2475-9066},
  doi     = {10.21105/joss.07526},
  url     = {https://joss.theoj.org/papers/10.21105/joss.07526}
}
```
