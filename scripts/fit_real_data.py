# fit_real_data.py

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

from hppca import fit_hppca


def _parse_optional_col_name(col_name: str | None) -> str | None:
    if col_name is None:
        return None
    text = str(col_name).strip()
    if text.lower() in {"", "none", "null", "na", "nan"}:
        return None
    return text


def load_real_data_dataframe(path: str, sep: str = ",") -> pd.DataFrame:
    if path is None:
        raise ValueError("--data_path is required and must point to a CSV file.")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    if not path.lower().endswith(".csv"):
        raise ValueError("This script expects CSV input. Please provide a .csv file.")
    df = pd.read_csv(path, sep=sep)
    if df.empty:
        raise ValueError(f"Loaded DataFrame is empty: {path}")
    return df


def _fmt_ell(x, precision: int = 6) -> str:
    if x is None:
        return "None"
    try:
        arr = np.asarray(x)
    except Exception:
        return str(x)
    if arr.size == 1:
        return f"{float(arr):.{precision}f}"
    return ";".join(f"{float(v):.{precision}f}" for v in arr.ravel())


def _pad_int_lists(lst, fill: int = -1, dtype=np.int32):
    n = len(lst)
    arr_list = [np.asarray(x, dtype=dtype) for x in lst]
    jmax = max((len(x) for x in arr_list), default=0)
    out = np.full((n, jmax), fill, dtype=dtype)
    mask = np.zeros((n, jmax), dtype=bool)
    for i, x in enumerate(arr_list):
        m = len(x)
        if m > 0:
            out[i, :m] = x
            mask[i, :m] = True
    return out, mask


def save_hppca_npz_padded(
    path: str,
    W1_final: np.ndarray,
    W2_final: np.ndarray,
    sigma2_final: float,
    ell_param_final,
    EZ1i_final_list_out: list,
    EZ2ij_final_list_out: list,
    Y_filled_final_out,
    participant_survey_indices: list,
    participant_original_indices: list,
    survey_times: np.ndarray,
    subject_id_col: str,
    visit_time_col: str | None,
    feature_cols: list[str],
):
    n_eff = len(EZ1i_final_list_out)
    d1 = W1_final.shape[1]
    d2 = W2_final.shape[1]

    EZ1i_arr = np.stack(
        [np.asarray(z, dtype=np.float32).reshape(d1) for z in EZ1i_final_list_out],
        axis=0,
    ) if n_eff > 0 else np.zeros((0, d1), dtype=np.float32)

    Ji = np.array(
        [0 if z is None else np.asarray(z).shape[1] for z in EZ2ij_final_list_out],
        dtype=np.int32,
    )
    J_max = int(Ji.max(initial=0))
    EZ2ij_pad = np.zeros((n_eff, d2, J_max), dtype=np.float32)
    EZ2ij_mask = np.zeros((n_eff, J_max), dtype=bool)
    for i, z in enumerate(EZ2ij_final_list_out):
        if z is None:
            continue
        A = np.asarray(z)
        if A.ndim != 2:
            raise ValueError(f"EZ2ij[{i}] is not 2D (got ndim={A.ndim}).")
        if A.shape[0] != d2 and A.shape[1] == d2:
            A = A.T
        if A.shape[0] != d2:
            raise ValueError(f"EZ2ij[{i}] has incompatible shape {A.shape}; expected (d2, J_i).")
        m = A.shape[1]
        EZ2ij_pad[i, :, :m] = A.astype(np.float32, copy=False)
        EZ2ij_mask[i, :m] = True

    psi_pad, psi_mask = _pad_int_lists(participant_survey_indices, fill=-1, dtype=np.int32)
    if ell_param_final is None:
        ell_arr = np.array([], dtype=np.float32)
    else:
        ell_arr = np.atleast_1d(np.asarray(ell_param_final, dtype=np.float32))

    payload = {
        "W1": W1_final.astype(np.float32, copy=False),
        "W2": W2_final.astype(np.float32, copy=False),
        "sigma2": np.array(sigma2_final, dtype=np.float32),
        "ell": ell_arr,
        "EZ1i": EZ1i_arr,
        "EZ2ij_pad": EZ2ij_pad,
        "EZ2ij_mask": EZ2ij_mask,
        "Ji": Ji,
        "participant_survey_indices_pad": psi_pad,
        "participant_survey_indices_mask": psi_mask,
        "participant_original_indices": np.asarray(participant_original_indices, dtype=np.int32),
        "survey_times": np.asarray(survey_times, dtype=np.float32),
        "feature_names": np.asarray(feature_cols, dtype=str),
    }

    if isinstance(Y_filled_final_out, pd.DataFrame):
        payload["y_filled_subject"] = Y_filled_final_out[subject_id_col].to_numpy()
        if visit_time_col is not None and visit_time_col in Y_filled_final_out.columns:
            payload["y_filled_visit"] = Y_filled_final_out[visit_time_col].to_numpy(dtype=np.float32, copy=False)
        payload["y_filled_values"] = Y_filled_final_out.loc[:, feature_cols].to_numpy(dtype=np.float32, copy=True)
    else:
        payload["Y_filled"] = np.asarray(Y_filled_final_out, dtype=np.float32)

    out_path = os.path.abspath(path)
    np.savez_compressed(out_path, **payload)
    print(f"Saved padded bundle to: {out_path}")


parser = argparse.ArgumentParser(description="Fit HPPCA on real tabular longitudinal data from CSV.")
parser.add_argument("--iter", type=int, default=5, help="Maximum number of EM iterations")
parser.add_argument("--tol", type=float, default=1e-4, help="Convergence tolerance")
parser.add_argument("--d1", type=int, default=2, help="Latent dimension for Z1i")
parser.add_argument("--d2", type=int, default=2, help="Latent dimension for Z2ij")
parser.add_argument("--survey_times", type=float, nargs="+", default=None,
                    help="Optional global survey times. If omitted, inferred from visit_time or defaults.")
parser.add_argument("--seed", type=int, default=42, help="Random seed")
parser.add_argument("--n_cpus", type=int, default=3, help="Number of CPUs for E-step")
parser.add_argument("--savepath", type=str, default="results/", help="Path to save results")
parser.add_argument("--kernel_method", type=str, default="gp_rbf_single_ell",
                    choices=["gp_rbf_single_ell", "gp_rbf_multi_ell",
                             "gp_matern52_single_ell", "gp_matern52_multi_ell",
                             "gp_iid"],
                    help="Kernel method for Z2 temporal structure")
parser.add_argument("--init_method", type=str, default="algo1",
                    choices=["algo2_cs", "algo1", "random"],
                    help="Initialization strategy passed to fit_hppca")

parser.add_argument("--data_path", type=str, required=True, help="Path to input CSV")
parser.add_argument("--csv_sep", type=str, default=",", help="CSV separator")
parser.add_argument("--data_label", type=str, default="REAL", help="Dataset label for filenames")
parser.add_argument("--subject_id_col", type=str, default="subject_id", help="Subject ID column name")
parser.add_argument("--visit_time_col", type=str, default="visit_time",
                    help="Visit-time column name. Set to 'none' to disable.")
parser.add_argument("--feature_cols", type=str, nargs="+", default=None,
                    help="Feature columns. If omitted, use all except subject_id/visit_time.")

try:
    if "ipykernel" in sys.modules and len(sys.argv) > 1 and sys.argv[1].endswith("json"):
        args = parser.parse_args([])
    elif len(sys.argv) > 1:
        args = parser.parse_args()
    else:
        args = parser.parse_args([])
except SystemExit:
    print("Running with default args (SystemExit fallback).")
    args = parser.parse_args([])

savepath = args.savepath
if not os.path.exists(savepath):
    os.makedirs(savepath)
    print(f"Created directory: {savepath}")

d1, d2 = args.d1, args.d2
seed = args.seed
iters = args.iter
tol = args.tol
n_cpus = args.n_cpus
kernel_method = args.kernel_method
data_label = args.data_label

subject_id_col = args.subject_id_col
visit_time_col = _parse_optional_col_name(args.visit_time_col)

df_input = load_real_data_dataframe(args.data_path, sep=args.csv_sep)
if subject_id_col not in df_input.columns:
    raise ValueError(f"subject_id column '{subject_id_col}' not found in CSV.")
if visit_time_col is not None and visit_time_col not in df_input.columns:
    raise ValueError(f"visit_time column '{visit_time_col}' not found in CSV.")

if args.feature_cols is None:
    excluded = {subject_id_col}
    if visit_time_col is not None:
        excluded.add(visit_time_col)
    feature_cols = [c for c in df_input.columns if c not in excluded]
else:
    feature_cols = list(args.feature_cols)
    missing_features = [c for c in feature_cols if c not in df_input.columns]
    if missing_features:
        raise ValueError(f"feature_cols not found in CSV: {missing_features}")

if len(feature_cols) == 0:
    raise ValueError("No feature columns were selected.")
if subject_id_col in feature_cols:
    raise ValueError("subject_id_col must not appear in feature_cols.")
if visit_time_col is not None and visit_time_col in feature_cols:
    raise ValueError("visit_time_col must not appear in feature_cols.")

df_input = df_input.copy()
for col in feature_cols:
    df_input[col] = pd.to_numeric(df_input[col], errors="coerce")

if visit_time_col is not None:
    df_input[visit_time_col] = pd.to_numeric(df_input[visit_time_col], errors="coerce")

n_orig = int(df_input[subject_id_col].nunique(dropna=True))
missing_rate = float(df_input.loc[:, feature_cols].isna().mean().mean())
print(
    f"Loaded CSV {args.data_path} with {len(df_input)} rows, "
    f"n_subjects={n_orig}, n_features={len(feature_cols)}."
)
print(f"Empirical missing rate over feature cells: {missing_rate:.6f}")

if args.survey_times is not None:
    survey_times_arg = np.asarray(args.survey_times, dtype=float)
else:
    survey_times_arg = None

np.random.seed(seed)
print(f"Kernel Method for Estimation: {kernel_method}")
print(f"visit_time_col used: {visit_time_col}")
print(f"feature_cols count: {len(feature_cols)}")

start_time = time.time()
(
    W1_final, W2_final, sigma2_final, ell_param_final,
    EZ1i_final_list_out, EZ2ij_final_list_out, Y_filled_final_out,
    iteration_num, converged,
    W1_init_used, W2_init_used, sigma2_init_used, ell_param_init_used,
    Y_list, participant_survey_indices, participant_original_indices, survey_times_out,
    latent_X_df,
) = fit_hppca(
    Y_obs=df_input,
    d1=d1,
    d2=d2,
    kernel_method=kernel_method,
    survey_times=survey_times_arg,
    init_method=args.init_method,
    n_cpus=n_cpus,
    max_iter=iters,
    tol=tol,
    seed=seed,
    subject_id_col=subject_id_col,
    visit_time_col=visit_time_col,
    feature_cols=feature_cols,
    return_filled_dataframe=True,
    return_latent_dataframe=True,
)
total_time = time.time() - start_time

survey_times = np.asarray(survey_times_out, dtype=float)
J = int(survey_times.shape[0])
p = int(len(feature_cols))
n_eff = len(Y_list)

print(f"\n--- EM Finished ({kernel_method}) ---")
print(f"Time: {total_time:.2f}s. Converged: {converged} in {iteration_num} iterations.")
print(f"Final sigma2: {sigma2_final:.6f}")
print(f"Inferred timeline length J={J}")

filename_base = (
    f"{data_label}_{args.init_method}_{kernel_method}_iter{iters}_tol{tol}_"
    f"n{n_orig}_J{J}_p{p}_d1{d1}_d2{d2}_r{missing_rate:.4f}_sd{seed}"
)

results_for_csv = {
    "data_mode": "real_csv_dataframe",
    "data_path": args.data_path,
    "kernel_method": kernel_method,
    "init_method": args.init_method,
    "subject_id_col": subject_id_col,
    "visit_time_col": str(visit_time_col),
    "n_rows_input": int(len(df_input)),
    "n_orig": n_orig,
    "n_eff": n_eff,
    "J": J,
    "p": p,
    "d1": d1,
    "d2": d2,
    "missing_rate": missing_rate,
    "sigma2_final": sigma2_final,
    "sigma2_init": sigma2_init_used,
    "ell_init": _fmt_ell(ell_param_init_used, precision=6),
    "ell_final": _fmt_ell(ell_param_final, precision=6),
    "converged": converged,
    "iteration_num": iteration_num,
    "total_time": total_time,
    "seed": seed,
    "tol": tol,
    "n_cpus": n_cpus,
}

results_df = pd.DataFrame([results_for_csv])
results_filename_csv = os.path.join(savepath, f"{filename_base}_results.csv")
results_df.to_csv(results_filename_csv, index=False)
print(f"Saved results to: {results_filename_csv}")

if isinstance(Y_filled_final_out, pd.DataFrame):
    yfilled_path = os.path.join(savepath, f"{filename_base}_Yfilled.csv")
    Y_filled_final_out.to_csv(yfilled_path, index=False)
else:
    yfilled_path = os.path.join(savepath, f"{filename_base}_Yfilled.npy")
    np.save(yfilled_path, np.asarray(Y_filled_final_out))
print(f"Saved filled output to: {yfilled_path}")

latent_path = os.path.join(savepath, f"{filename_base}_latent_X_df.csv")
latent_X_df.to_csv(latent_path, index=False)
print(f"Saved latent factors to: {latent_path}")

estimates_npz_path = os.path.join(savepath, f"{filename_base}_estimates.npz")
save_hppca_npz_padded(
    path=estimates_npz_path,
    W1_final=W1_final,
    W2_final=W2_final,
    sigma2_final=sigma2_final,
    ell_param_final=ell_param_final,
    EZ1i_final_list_out=EZ1i_final_list_out,
    EZ2ij_final_list_out=EZ2ij_final_list_out,
    Y_filled_final_out=Y_filled_final_out,
    participant_survey_indices=participant_survey_indices,
    participant_original_indices=participant_original_indices,
    survey_times=survey_times,
    subject_id_col=subject_id_col,
    visit_time_col=visit_time_col,
    feature_cols=feature_cols,
)
