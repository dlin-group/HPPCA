# HPPCA

Hierarchical Probabilistic Principal Component Analysis (HPPCA) for longitudinal data with item-level missingness and Gaussian-process temporal latent factors.

This directory is the package-oriented version of the research code in `../hppca`.

## Install

From this directory:

```bash
pip install -e .
```

The pybind11 extension links against system OpenBLAS and LAPACKE. On systems without those libraries, install the corresponding development packages before building.

## Basic Usage

```python
from hppca import fit_hppca

result = fit_hppca(
    Y_obs,
    d1=2,
    d2=2,
    kernel_method="gp_rbf_single_ell",
    init_method="algo2_cs",
    max_iter=50000,
    tol=1e-4,
    n_cpus=4,
    seed=42,
)
```

The main public functions are:

- `fit_hppca`
- `fit_hppca_alg1`
- `fit_hppca_alg2_cs`

## Key Arguments

### `kernel_method`

`kernel_method` controls the temporal prior for the visit-level latent factor `Z2`.

- `"gp_iid"`: Independent visit-level latent factors across time. This is the simplest option and does not estimate a length-scale.
- `"gp_rbf_single_ell"`: RBF kernel with one shared length-scale across all `d2` dynamic latent dimensions. This is a good default when the latent trajectories are expected to be smooth.
- `"gp_rbf_multi_ell"`: RBF kernel with one length-scale per dynamic latent dimension. This is more flexible than `single_ell`, but less identifiable and may need more data/iterations.
- `"gp_matern52_single_ell"`: Matern-5/2 kernel with one shared length-scale. This allows rougher trajectories than RBF while still modeling temporal correlation.
- `"gp_matern52_multi_ell"`: Matern-5/2 kernel with one length-scale per dynamic latent dimension. This is the most flexible Matern option.

Suggested starting point: use `"gp_rbf_single_ell"` for smooth longitudinal data, `"gp_matern52_single_ell"` if trajectories look less smooth, and `"gp_iid"` as a baseline or diagnostic.

### `init_method`

`init_method` controls how EM is initialized.

- `"algo2_cs"`: Compound-symmetry initializer using Algorithm 2 in the paper. This is usually a robust and fast default.
- `"algo1"`: Known/shared covariance initializer using an estimated temporal covariance projected to the paper's Assumption 1 (Algorithm 1 in the paper). This can be useful when the temporal covariance estimate is reliable.
- `"random"`: Random loading initialization. This is mainly useful for debugging or sensitivity checks; it usually needs more EM iterations.

Suggested starting point: use `"algo2_cs"` unless you specifically want to compare initializers.

### `max_iter` and `tol`

The EM algorithm can require many iterations, especially with missing data, larger latent dimensions, or GP length-scale updates.

Suggested values:

- `max_iter=50000`
- `tol=1e-4`

For quick smoke tests or examples, use a much smaller `max_iter` such as `3` or `10`. For real analyses, use the suggested values above and check the reported convergence status.

### `n_cpus`

`n_cpus` controls parallelization of the E-step across participants.

- `n_cpus=1`: Run serially. This is easiest for debugging and avoids multiprocessing overhead.
- `n_cpus>1`: Use multiprocessing across participants. This is usually faster for larger datasets.

Suggested value: use the number of CPU cores you want to allocate, for example `n_cpus=4` or `n_cpus=8`. For very small datasets, `n_cpus=1` may be faster because multiprocessing overhead can dominate.

### `seed`

`seed` sets the NumPy random seed used for initialization and reproducibility.

Suggested value: set an integer such as `seed=42` for reproducible analyses.

See `examples/` for small runnable examples and `scripts/` for command-line workflows.
