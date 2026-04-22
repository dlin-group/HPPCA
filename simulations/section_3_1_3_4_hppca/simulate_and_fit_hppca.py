# simulate_and_fit_correct_iid_model.py

import numpy as np
import time
import pandas as pd
import argparse
import os
import sys

from hppca import fit_hppca
from hppca.em_worker import rbf_kernel, matern52_kernel
from hppca.hppca_method import principal_angles

def matrix_rank_num(A, tol=None):
    """Numerical rank using numpy's default (S.max * eps * max(m,n)) unless tol provided."""
    return int(np.linalg.matrix_rank(A, tol=tol))


# Argument Parser
parser = argparse.ArgumentParser(description="EM Algorithm for HPPCA with GP Extension")
parser.add_argument("--iter", type=int, default=50000, help="Maximum Number of EM Iterations")
parser.add_argument("--tolerance", type=float, default=1e-4, help="Tolerance for convergence")
parser.add_argument("--n", type=int, default=15000, help="Number of participants")
parser.add_argument("--J", type=int, default=10, help="Number of total survey waves")
parser.add_argument("--p", type=int, default=60, help="Number of features")
parser.add_argument("--d1", type=int, default=2, help="Latent dimension for Z1i")
parser.add_argument("--d2", type=int, default=2, help="Latent dimension for Z2ij")
parser.add_argument("--rate", type=float, default=0.2, help="Missing rate")
parser.add_argument("--truesigma2", type=float, default=0.25, help="True sigma2")
parser.add_argument("--ell_true", type=float, nargs='+', default=[5.0, 5.0], help="True GP length-scale(s).")
parser.add_argument("--survey_times", type=float, nargs='+', default=None,
                    help="Survey times (length J). If omitted, uses default spaced times.")
parser.add_argument("--seed", type=int, default=42, help="Random Seed")
parser.add_argument("--n_cpus", type=int, default=1, help="Number of CPUs for E-step")
parser.add_argument(
    "--savepath",
    type=str,
    default=os.path.join("sim_results", "results_hppca_init"),
    help="Path to save results",
)
parser.add_argument("--kernel_method", type=str, default="gp_rbf_single_ell",
                    choices=["gp_rbf_single_ell", "gp_rbf_multi_ell",
                             "gp_matern52_single_ell", "gp_matern52_multi_ell",
                             "gp_iid"],
                    help="Method for GP kernel and ell handling.")
parser.add_argument(
    "--init_method",
    type=str,
    default="algo1",
    choices=["algo2_cs", "algo1", "random"],
    help="Initialization strategy passed to fit_hppca (previous random init available via 'random').",
)
parser.add_argument(
    "--input_format",
    type=str,
    default="dataframe",
    choices=["dataframe"],
    help="Input structure passed to fit_hppca. Only long DataFrame is supported.",
)


try:
    if 'ipykernel' in sys.modules and len(sys.argv) > 1 and sys.argv[1].endswith('json'):
        args = parser.parse_args([])
    elif len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args([])
except SystemExit:
    print("Running with default args (SystemExit fallback):")
    args = parser.parse_args([])

savepath = args.savepath
init_method = args.init_method
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print(f"Created directory: {savepath}")

n_orig, J, p = args.n, args.J, args.p
d1, d2 = args.d1, args.d2
missing_rate = args.rate
seed = args.seed
iters = args.iter
tolerance = args.tolerance
sigma2_true = args.truesigma2
n_cpus = args.n_cpus
kernel_method = args.kernel_method
input_format = args.input_format
ell_true_input = args.ell_true
# For gp_iid, ell is not used in simulation/estimation; skip validation
if "iid" in kernel_method:
    ell_true_vec = None
else:
    if len(ell_true_input) == 1:
        ell_true_vec = np.full(d2, ell_true_input[0])
    elif len(ell_true_input) == d2:
        ell_true_vec = np.array(ell_true_input)
    else:
        raise ValueError(f"--ell_true values ({len(ell_true_input)}) must be 1 or d2 ({d2}).")
if args.survey_times is not None:
    survey_times = np.asarray(args.survey_times, dtype=float)
    if survey_times.shape[0] != J:
        raise ValueError(f"--survey_times length ({survey_times.shape[0]}) must equal J ({J}).")
else:
    # Default equally spaced times from 10 to 10*J (inclusive)
    survey_times = np.linspace(10, 10 * J, J)
print(f"Settings: n={n_orig}, J={J}, p={p}, d1={d1}, d2={d2}, rate={missing_rate}, sigma2_true={sigma2_true}, seed={seed}")
print(f"True ell vector (for simulation): {ell_true_vec}")
print(f"Kernel Method for Estimation: {kernel_method}")
print(f"Survey Times (len {len(survey_times)}): {survey_times}")
print(f"Initialization method: {init_method}")
print(f"Input format: {input_format}")

np.random.seed(seed)

# ---- Simulate True Data ----
Z_1_true = np.random.randn(n_orig, d1)
Z_2_true = np.zeros((n_orig, J, d2))
K_true_list_sim = []
simulation_kernel_type_for_Z2 = "rbf"
if "matern52" in kernel_method:
    simulation_kernel_type_for_Z2 = "matern52"
elif "iid" in kernel_method: 
    simulation_kernel_type_for_Z2 = "iid"

print(f"Simulating Z2_true using {simulation_kernel_type_for_Z2.upper()} kernel as per estimation method.")
if simulation_kernel_type_for_Z2 in ("rbf", "matern52"):
    for k_sim in range(d2):
        current_ell_sim = ell_true_vec[k_sim]
        if simulation_kernel_type_for_Z2 == "rbf":
            K_true_k_sim, _ = rbf_kernel(survey_times, current_ell_sim)
        elif simulation_kernel_type_for_Z2 == "matern52":
            K_true_k_sim, _ = matern52_kernel(survey_times, current_ell_sim)
        else:
            raise ValueError(f"Unknown simulation kernel type: {simulation_kernel_type_for_Z2}")
        K_true_list_sim.append(K_true_k_sim)
for i in range(n_orig):
    for k_sim in range(d2):
        if simulation_kernel_type_for_Z2 == "iid":
            Z_2_true[i, :, k_sim] = np.random.randn(J)
        else:
            Z_2_true[i, :, k_sim] = np.random.multivariate_normal(np.zeros(J), K_true_list_sim[k_sim], check_valid='warn', tol=1e-8)

W1_true = np.random.randn(p, d1)
W2_true = np.random.randn(p, d2)
Y_full = np.zeros((n_orig, J, p))
for i in range(n_orig):
    for j_idx_sim in range(J):
        Y_full[i, j_idx_sim, :] = W1_true @ Z_1_true[i] + W2_true @ Z_2_true[i, j_idx_sim] + \
                                  np.sqrt(sigma2_true) * np.random.randn(p)
observed_mask = np.random.rand(n_orig, J, p) < (1 - missing_rate)
Y_obs = Y_full.copy()
Y_obs[~observed_mask] = np.nan

fit_kwargs = {
    "survey_times": survey_times,
}

include_visit_time = "iid" not in kernel_method
if not include_visit_time:
    # For gp_iid, visit_time is not used; allow missing by construction.
    fit_kwargs["survey_times"] = None

feature_cols = [f"feature_{k}" for k in range(p)]
record_rows = []
for i in range(n_orig):
    for j in range(J):
        y_ij = Y_obs[i, j, :]
        # One row per observed visit in long format.
        if np.all(np.isnan(y_ij)):
            continue
        row = {"subject_id": i}
        if include_visit_time:
            row["visit_time"] = float(survey_times[j])
        for k in range(p):
            row[feature_cols[k]] = y_ij[k]
        record_rows.append(row)

df_cols = ["subject_id", *([] if not include_visit_time else ["visit_time"]), *feature_cols]
fit_input = pd.DataFrame(record_rows, columns=df_cols)
fit_kwargs.update(
    {
        "subject_id_col": "subject_id",
        "visit_time_col": ("visit_time" if include_visit_time else None),
        "feature_cols": feature_cols,
    }
)
print(
    f"Built {len(fit_input)} long-format rows for {input_format} input "
    f"(include_visit_time={include_visit_time})."
)

# ---- Fit HPPCA  ----
start_time = time.time()
(
    W1_final, W2_final, sigma2_final, ell_param_final,
    EZ1i_final_list_out, EZ2ij_final_list_out, Y_filled_final_out,
    iteration_num, converged,
    W1_init, W2_init, sigma2_init, ell_param_init,
    Y_list, participant_survey_indices, participant_original_indices, survey_times,
    latent_Z_df,
) = fit_hppca(
    Y_obs=fit_input,
    d1=d1,
    d2=d2,
    kernel_method=kernel_method,
    n_cpus=n_cpus,
    init_method=init_method,
    max_iter=iters,
    tol=tolerance,
    seed=seed,
    return_latent_dataframe=True,
    **fit_kwargs,
)
total_time = time.time() - start_time
n = len(Y_list)  # Effective number of participants from the wrapper

total_time = time.time() - start_time
print(f"\n--- EM Finished ({kernel_method}) ---")
print(f"Time: {total_time:.2f}s. Converged: {converged} in {iteration_num} iterations.")
print(f"Final sigma2: {sigma2_final:.6f} (True: {sigma2_true:.6f})")

# ---------- HPPCA evaluation metrics ----------
# Combined loading space and nominal model dimension
W_concat = np.concatenate([W1_final, W2_final], axis=1) if d1 > 0 else W2_final
num_rank = matrix_rank_num(W_concat)  # numerical rank of HPPCA loadings

d_model = (d1 + d2)

# ---------- Subspace comparison (principal angles) vs. ground truth ----------
# ---------- Principal angles vs. ground truth ----------
def _safe_pa(A, B):
    try:
        return principal_angles(A, B)
    except Exception:
        return None

def _max_angle(a):
    return float(np.nanmax(a)) if (a is not None and getattr(a, "size", 0) > 0) else np.nan

# HPPCA vs TRUE
angles_W1_vs_true          = _safe_pa(W1_final, W1_true) if d1 > 0 else None
angles_W2_vs_true          = _safe_pa(W2_final, W2_true)

sum_angles_W1_vs_true          = float(np.nansum(angles_W1_vs_true))          if angles_W1_vs_true          is not None else np.nan
sum_angles_W2_vs_true          = float(np.nansum(angles_W2_vs_true))          if angles_W2_vs_true          is not None else np.nan

max_angle_W1_vs_true          = _max_angle(angles_W1_vs_true)
max_angle_W2_vs_true          = _max_angle(angles_W2_vs_true)


# initial W1 and W2 vs true
angles_W1_init_vs_true     = _safe_pa(W1_init, W1_true) if d1 > 0 else None
angles_W2_init_vs_true     = _safe_pa(W2_init, W2_true)

sum_angles_W1_init_vs_true     = float(np.nansum(angles_W1_init_vs_true))     if angles_W1_init_vs_true     is not None else np.nan
sum_angles_W2_init_vs_true     = float(np.nansum(angles_W2_init_vs_true))     if angles_W2_init_vs_true     is not None else np.nan
max_angle_W1_init_vs_true     = _max_angle(angles_W1_init_vs_true)
max_angle_W2_init_vs_true     = _max_angle(angles_W2_init_vs_true)

def _fmt_angles(a):
    return ";".join(f"{float(x):.6f}" for x in (a.tolist() if a is not None else []))

def _fmt_ell(x, precision=6):
    """Format ell params that may be None, scalar, or array."""
    if x is None:
        return "None"
    try:
        arr = np.asarray(x)
    except Exception:
        return str(x)
    if arr.size == 1:
        return f"{float(arr):.{precision}f}"
    return ";".join(f"{float(v):.{precision}f}" for v in arr.ravel())

# --- Save Numerical Results ---
filename_base = (
    f"GP_init_{init_method}_{kernel_method}_iter{iters}_n{n_orig}_J{J}_p{p}_d1{d1}_d2{d2}_"
    f"r{missing_rate}_s{sigma2_true}_ellT{_fmt_ell(ell_true_vec, precision=4)}_"
    f"in{input_format}_sd{seed}"
)

# explainable fields for CSV
results_for_csv = {
    "init_method": init_method,
    "kernel_method": kernel_method,
    "input_format": input_format,
    "n_orig": n_orig, "n_eff": n, "J": J, "p": p,
    "d1": d1, "d2": d2, "d_model": d_model,
    "missing_rate": missing_rate,
    "sigma2_final": sigma2_final, "sigma2_true": sigma2_true,
    "sigma2_init": sigma2_init,
    "ell_init": _fmt_ell(ell_param_init, precision=6),
    "ell_final": _fmt_ell(ell_param_final, precision=6),
    "ell_true_sim": _fmt_ell(ell_true_vec, precision=4),
    "converged": converged, "iteration_num": iteration_num, "total_time": total_time,
    "seed": seed, "tolerance": tolerance, "n_cpus": n_cpus,

    # rank (HPPCA)
    "hppca_matrix_rank": num_rank,

    # principal angles vs truth
    # ----- HPPCA vs TRUE -----
    "angles_W1_vs_true_deg": _fmt_angles(angles_W1_vs_true),
    "sum_angles_W1_vs_true_deg": sum_angles_W1_vs_true,
    "max_angle_W1_vs_true_deg": max_angle_W1_vs_true,

    "angles_W2_vs_true_deg": _fmt_angles(angles_W2_vs_true),
    "sum_angles_W2_vs_true_deg": sum_angles_W2_vs_true,
    "max_angle_W2_vs_true_deg": max_angle_W2_vs_true,

    # ----- INIT vs TRUE -----
    "angles_W1_init_vs_true_deg": _fmt_angles(angles_W1_init_vs_true),
    "sum_angles_W1_init_vs_true_deg": sum_angles_W1_init_vs_true,
    "max_angle_W1_init_vs_true_deg": max_angle_W1_init_vs_true,

    "angles_W2_init_vs_true_deg": _fmt_angles(angles_W2_init_vs_true),
    "sum_angles_W2_init_vs_true_deg": sum_angles_W2_init_vs_true,
    "max_angle_W2_init_vs_true_deg": max_angle_W2_init_vs_true,
}


results_df = pd.DataFrame([results_for_csv])
results_filename_csv = os.path.join(savepath, f"{filename_base}_results.csv")
results_df.to_csv(results_filename_csv, index=False)
print(f"Saved results to: {results_filename_csv}")

# Save W matrices and Y_filled
if isinstance(Y_filled_final_out, pd.DataFrame):
    yfilled_path = os.path.join(savepath, f"{filename_base}_Yfilled.csv")
    Y_filled_final_out.to_csv(yfilled_path, index=False)
else:
    yfilled_path = os.path.join(savepath, f"{filename_base}_Yfilled.npy")
    np.save(yfilled_path, Y_filled_final_out)
print(f"Saved Y_filled_final_out to: {yfilled_path}")

latent_path = os.path.join(savepath, f"{filename_base}_latent_Z_df.csv")
latent_Z_df.to_csv(latent_path, index=False)
print(f"Saved latent_Z_df to: {latent_path}")
