# HPPCA

Hierarchical Probabilistic Principal Component Analysis (HPPCA) for longitudinal data with item-level missingness and Gaussian-process temporal latent factors.

This directory is the package-oriented version of the research code in `../hppca`.

Simulation code is in `simulations/`. See `examples/` for small runnable examples and `scripts/` for command-line workflows.

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

## Output of `fit_hppca`

`fit_hppca` returns a tuple. By default, the tuple has 18 entries because
`return_latent_dataframe=True`. Set `return_latent_dataframe=False` to return
only the first 17 entries.

Return flag defaults:

- `return_filled_dataframe=True` by default. With this default, `Y_filled_out` is returned as a DataFrame. For 3D tensor input, set `return_filled_dataframe=False` to get `Y_filled_out` as a tensor with shape `(n_orig, J, p)`. Set `return_filled_dataframe=None` for input-dependent behavior: tensor input returns a tensor, and tabular input returns a DataFrame.
- `return_latent_dataframe=True` by default. With this default, the extra latent-factor DataFrame `latent_Z_df` is appended to the tuple.

Shape notation used below:

- `n_orig`: number of original participants or unique subjects.
- `n_eff`: number of participants with at least one observed visit used by EM.
- `J`: number of survey times in the global survey grid.
- `J_i`: number of observed visits for effective participant `i`.
- `p`: number of observed features/items.
- `d1`: number of participant-level latent dimensions.
- `d2`: number of visit-level temporal latent dimensions.
- `n_rows`: number of rows in the input table when `Y_obs` is tabular.

The default output can be unpacked as:

```python
(
    W1f,
    W2f,
    s2f,
    ell_pf,
    EZ1i_list,
    EZ2ij_list,
    Y_filled_out,
    iteration_num,
    converged,
    W1_init,
    W2_init,
    sigma2_init,
    ell_param_init,
    Y_list,
    participant_survey_indices,
    participant_original_indices,
    survey_times,
    latent_Z_df,
) = fit_hppca(...)
```

Detailed output list:

- `W1f`: final loading matrix for the participant-level latent factor `Z1`; shape `(p, d1)`.
- `W2f`: final loading matrix for the visit-level temporal latent factor `Z2`; shape `(p, d2)`.
- `s2f`: final observation noise variance, also called `sigma2`; scalar `float`.
- `ell_pf`: final GP length-scale parameter. It is `None` for `"gp_iid"`, a scalar `float` for `*_single_ell`, and a NumPy array with shape `(d2,)` for `*_multi_ell`.
- `EZ1i_list`: posterior means of participant-level latent variables. This is a list of length `n_eff`; entry `i` has shape `(d1,)` and is `E[Z1_i | data]`.
- `EZ2ij_list`: posterior means of visit-level latent variables. This is a list of length `n_eff`; entry `i` has shape `(d2, J_i)` and is `E[Z2_i | data]`. Its columns correspond to `participant_survey_indices[i]`.
- `Y_filled_out`: data after filling missing values using the fitted model. Its type and dimensions depend on the input and `return_filled_dataframe`.
- `iteration_num`: number of EM iterations actually run; integer.
- `converged`: whether EM stopped because the maximum relative parameter change was below `tol`; Boolean.
- `W1_init`: initial `W1` used by EM; shape `(p, d1)`.
- `W2_init`: initial `W2` used by EM; shape `(p, d2)`.
- `sigma2_init`: initial observation noise variance used by EM; scalar `float`.
- `ell_param_init`: initial GP length-scale parameter. It follows the same convention as `ell_pf`: `None`, scalar, or shape `(d2,)`.
- `Y_list`: internal participant-level observed-data matrices used by EM. This is a list of length `n_eff`; entry `i` has shape `(p, J_i)`.
- `participant_survey_indices`: list of length `n_eff`; entry `i` has shape `(J_i,)` and maps columns of `Y_list[i]` and `EZ2ij_list[i]` back to positions in `survey_times`.
- `participant_original_indices`: list of length `n_eff`; entry `i` maps effective participant `i` back to the original participant/subject index.
- `survey_times`: global survey-time grid used by the GP prior; shape `(J,)`.
- `latent_Z_df`: latent-factor table returned by default; shape `(sum_i J_i, d1 + d2 + 2)`. It is omitted only when `return_latent_dataframe=False`.

`Y_filled_out` dimensions:

- For 3D tensor input `Y_obs` with shape `(n_orig, J, p)` and the default `return_filled_dataframe=True`, `Y_filled_out` is a long pandas DataFrame with shape `(n_orig * J, p + 2)`. Columns are `subject_id`, `visit_time`, and `feature_0` through `feature_{p-1}`.
- For 3D tensor input with `return_filled_dataframe=False`, `Y_filled_out` is a NumPy array with shape `(n_orig, J, p)`.
- For tabular input, `Y_filled_out` is a pandas DataFrame with shape `(n_rows, p + 2)`. It keeps one row per original input record; columns are `subject_id`, `visit_time`, and the feature columns. It does not expand to the full `n_orig * J` grid.

When `return_latent_dataframe=False`, the output has only the first 17 entries:

```python
(
    W1f,
    W2f,
    s2f,
    ell_pf,
    EZ1i_list,
    EZ2ij_list,
    Y_filled_out,
    iteration_num,
    converged,
    W1_init,
    W2_init,
    sigma2_init,
    ell_param_init,
    Y_list,
    participant_survey_indices,
    participant_original_indices,
    survey_times,
) = fit_hppca(..., return_latent_dataframe=False)
```

`latent_Z_df` is a long pandas DataFrame with shape `(sum_i J_i, d1 + d2 + 2)`.
It has one row per observed visit used by EM. Columns are `subject_id`,
`visit_time`, `z1_dim_1` through `z1_dim_d1`, and `z2_dim_1` through
`z2_dim_d2`. The `Z1` values are participant-level values, so they repeat
across visits for the same participant.

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
