
# simulate_and_fit_comparison_models.py  (TAGGED FILENAMES)
# -----------------------------------------------------------------------------
# Methods: PPCA (Python) and mFPCA (Python/FDApy)
#
# All intermediate files are saved with a unique TAG derived from parameters:
#   TAG = n{n}_J{J}_p{p}_d1{d1}_d2{d2}_r{rate}_s{truesigma2}_km{kernel_method}_ell{ell_str}_sd{seed}
# where ell_str is "ell1xell2x..." if multiple, with 4-decimal precision.
# -----------------------------------------------------------------------------

import os, argparse, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyppca import ppca

# Optional dependency: FDApy for mFPCA
try:
    # --- add IrregularValues to the imports at the top ---
    # from FDApy.representation import DenseArgvals, IrregularArgvals, DenseFunctionalData, IrregularFunctionalData, MultivariateFunctionalData
    # replace with:
    from FDApy.representation import (
        DenseArgvals, IrregularArgvals, IrregularValues,
        DenseFunctionalData, IrregularFunctionalData, MultivariateFunctionalData
    )
    from FDApy.preprocessing import MFPCA
    _HAVE_FDAPY = True
except Exception as e:
    warnings.warn(f"FDApy not available; mFPCA will be skipped. ({e})")
    _HAVE_FDAPY = False

try:
    from hppca.em_worker import rbf_kernel, matern52_kernel
except Exception:
    def rbf_kernel(x, ell, sigma_f=1.0):
        x = np.asarray(x).reshape(-1, 1)
        d2 = (x - x.T)**2
        return (sigma_f**2)*np.exp(-0.5*d2/ell**2), None
    def matern52_kernel(x, ell, sigma_f=1.0):
        x = np.asarray(x).reshape(-1, 1)
        d = np.sqrt((x - x.T)**2)
        s = np.sqrt(5.0)*d/ell
        return (sigma_f**2)*(1 + s + s**2/3.0)*np.exp(-s), None

# ---------------- Utilities ----------------
def _ensure_spd(mat, eps=1e-9, max_tries=6):
    sym = 0.5*(mat+mat.T)
    eye = np.eye(sym.shape[0])
    jitter = eps
    for _ in range(max_tries):
        try:
            np.linalg.cholesky(sym + jitter*eye)
            return sym + jitter*eye
        except np.linalg.LinAlgError:
            jitter *= 10.0
    w,v = np.linalg.eigh(sym)
    w[w<eps] = eps
    return v @ np.diag(w) @ v.T

def _masked_mse(pred, truth, mask):
    if mask.sum()==0: return float('nan')
    d = pred[mask] - truth[mask]
    return float(np.mean(d*d))

def _fmt_ell(x, precision=4):
    if x is None: return "None"
    arr = np.asarray(x).ravel()
    if arr.size==1: return f"{float(arr[0]):.{precision}f}"
    return "_".join(f"{float(v):.{precision}f}" for v in arr)

def _make_tag(n,J,p,d1,d2,rate,truesigma2,kernel_method,ell_true,seed):
    return (f"n{n}_J{J}_p{p}_d1{d1}_d2{d2}_r{rate}_s{truesigma2}_"
            f"km{kernel_method}_ell{_fmt_ell(ell_true)}_sd{seed}")

# ---------------- Simulation (same as HPPCA driver) ----------------
def simulate_data(n=100, J=5, p=20, d1=2, d2=2, rate=0.2, truesigma2=0.25,
                  ell_true=None, survey_times=None, seed=42, kernel_method="gp_iid"):
    rng = np.random.default_rng(seed)
    if survey_times is None:
        survey_times = np.linspace(10, 10*J, J).astype(float)

    if "iid" in kernel_method:
        ell_true_vec = None; kernel = "iid"
    else:
        if ell_true is None:
            ell_true_vec = np.full(d2, 10.0)
        else:
            e = np.asarray(ell_true, dtype=float).ravel()
            ell_true_vec = np.full(d2, float(e[0])) if e.size==1 else e
        kernel = "matern52" if "matern52" in kernel_method else "rbf"

    Z1 = rng.standard_normal((n, d1))
    Z2 = np.zeros((n, J, d2))

    K = []
    if kernel in ("rbf","matern52"):
        for k in range(d2):
            ellk = ell_true_vec[k]
            Kk, _ = (rbf_kernel if kernel=="rbf" else matern52_kernel)(survey_times, ellk)
            K.append(Kk)

    for i in range(n):
        for k in range(d2):
            if kernel=="iid":
                Z2[i, :, k] = rng.standard_normal(J)
            else:
                Z2[i, :, k] = rng.multivariate_normal(np.zeros(J), K[k])

    W1 = rng.standard_normal((p, d1))
    W2 = rng.standard_normal((p, d2))

    Y_full = np.zeros((n, J, p))
    for i in range(n):
        for j in range(J):
            Y_full[i, j, :] = W1 @ Z1[i] + W2 @ Z2[i, j] + np.sqrt(truesigma2)*rng.standard_normal(p)

    observed_mask = rng.random((n, J, p)) < (1.0 - rate)
    Y_obs = Y_full.copy(); Y_obs[~observed_mask] = np.nan

    truths = dict(Z1_true=Z1, Z2_true=Z2, W1_true=W1, W2_true=W2,
                  ell_true_vec=ell_true_vec, sigma2_true=float(truesigma2),
                  survey_times=survey_times, kernel_for_Z2=kernel)
    return Y_full, Y_obs, observed_mask, truths

# ---------------- Baseline PPCA ----------------
def fit_baseline_ppca(Y_obs, d_rank):
    """
    Baseline PPCA on (n*J) by p with rank = d_rank (treat (subject, survey) as i.i.d. rows).
    
    Now robust to cases where specific (subject, survey) pairs are completely missing.

    Returns
    -------
    Y_hat : (n, J, p)
    cov_signal : (J*p, J*p) = kron(I_J, C C^T)
    cov_full : (J*p, J*p)   = kron(I_J, C C^T + sigma2 I_p)
    sigma2_est : float
    latents : (n, J, d_rank)
    loadings : (p, d_rank)
    """
    n, J, p = Y_obs.shape
    n_total = n * J
    Y = Y_obs.reshape(n_total, p)

    # --- FIX START: Handle completely missing rows ---
    # In Baseline PPCA, a "row" is one timepoint for one subject.
    # We filter out rows that are entirely NaN.
    valid_mask = np.any(np.isfinite(Y), axis=1)
    n_valid = np.sum(valid_mask)
    
    # Log warning if data is dropped
    if n_valid < n_total:
        # Optional: print(f"[Info] fit_baseline_ppca: Dropping {n_total - n_valid} all-NaN rows.")
        Y_in = Y[valid_mask]
    else:
        Y_in = Y
    # --- FIX END ---

    # Fit PPCA on valid subset
    # C: (p, r), X: (r, n_valid), Ye: (n_valid, p)
    C, ss, M, X, Ye = ppca(Y_in, d_rank, dia=False) 

    # --- FIX START: Reconstruct full-size matrices ---
    
    # 1. Reconstruct Latents
    # Initialize with 0 (prior mean)
    latents_flat = np.zeros((n_total, d_rank))
    
    # Handle X shape (r, n_valid) -> Transpose to assign to (n_valid, r)
    if X.shape[1] == n_valid:
        latents_valid = X.T
    else:
        latents_valid = X
        
    latents_flat[valid_mask, :] = latents_valid
    latents = latents_flat.reshape(n, J, d_rank)

    # 2. Reconstruct Y_hat
    # Initialize with NaN
    Y_hat_flat = np.full((n_total, p), np.nan)
    
    # Fill valid predictions
    if Ye.shape[0] == n_valid:
        Y_hat_flat[valid_mask, :] = Ye
    elif Ye.shape[1] == n_valid:
        Y_hat_flat[valid_mask, :] = Ye.T

    # Fill completely missing rows with the Mean (M)
    # If a row was all NaNs, the best guess is the global mean vector (which should be 0 in the simulation)
    if M is not None:
        mean_vec = M.flatten()
        invalid_mask = ~valid_mask
        Y_hat_flat[invalid_mask, :] = mean_vec

    Y_hat = Y_hat_flat.reshape(n, J, p)
    # --- FIX END ---

    # Covariance construction remains the same (based on C and ss)
    cov_signal_single = C @ C.T
    cov_signal = np.kron(np.eye(J), cov_signal_single)
    cov_full = np.kron(np.eye(J), cov_signal_single + float(ss)*np.eye(p))

    return Y_hat, cov_signal, cov_full, float(ss), latents, C

# ---------------- mFPCA (FDApy, PACE) ----------------
def fit_mfpca_fdapy(Y_obs, survey_times, d1, d2, ncomp_uni,
                    methods=("covariance", "inner-product"),
                    Y_truth=None, plot_examples=0, plot_features=2,
                    plot_savepath=None, plot_seed=None):
    """
    Multivariate FPCA (FDApy) with optional covariance / inner-product fitting strategies.

    Returns
    -------
    dict : mapping method_name -> {
        "Y_hat": (n,J,p) reconstruction on the survey grid,
        "sigma2_est": float residual MSE on observed entries,
        "scores_subject": (n,R) matrix of multivariate scores,
        "R_final": retained component count,
        "transform_method": string used for scoring
    }
    If Y_truth is provided and plot_examples > 0, a comparison plot between the truth and
    both reconstructions is generated (saved to plot_savepath if provided, otherwise shown).
    """
    n, J, p = Y_obs.shape
    survey_times = np.asarray(survey_times, dtype=float)

    # --- Build per-feature IrregularFunctionalData correctly (dicts; 1-D values) ---
    components = []
    for k in range(p):
        arg_d = {}
        val_d = {}
        for i in range(n):
            m = np.isfinite(Y_obs[i, :, k])
            t_i = survey_times[m]
            y_i = np.asarray(Y_obs[i, m, k], dtype=float)
            if t_i.size == 0:
                t_i = np.array([survey_times[0]], dtype=float)
                y_i = np.array([0.0], dtype=float)
            arg_d[i] = DenseArgvals({'input_dim_0': t_i})
            val_d[i] = y_i

        irr_arg = IrregularArgvals(arg_d)
        irr_val = IrregularValues(val_d)
        components.append(IrregularFunctionalData(irr_arg, irr_val))

    mfd = MultivariateFunctionalData(components)

    # --- Choose target dimension and caps ---
    R_target = int(max(1, d1 + d2))
    R_cap = min(n - 1, p * ncomp_uni)
    R_final = max(1, min(R_target, R_cap))

    uni_exp = [{
        "method": "UFPCA",
        "n_components": ncomp_uni,
        "method_smoothing": "PS"
    } for _ in range(p)]

    def _reconstruct_on_grid(mfd_obj):
        Y_hat = np.zeros((n, J, p))
        for fk in range(p):
            comp = mfd_obj.data[fk]
            if hasattr(comp, "values") and hasattr(comp.values, "data"):
                arr = np.asarray(comp.values.data)
                if arr.ndim == 2:
                    Y_hat[:, :, fk] = arr[:, :J]
                else:
                    Y_hat[:, :, fk] = np.tile(arr, (n, 1))[:, :J]
            else:
                for ii in range(n):
                    t_i = np.asarray(comp.argvals[ii]['input_dim_0'], dtype=float)
                    if hasattr(comp, "values"):
                        y_i = np.asarray(comp.values[ii], dtype=float)
                    else:
                        y_i = np.asarray(comp.data[ii], dtype=float)
                    if t_i.size >= 2:
                        Y_hat[ii, :, fk] = np.interp(survey_times, t_i, y_i, left=y_i[0], right=y_i[-1])
                    else:
                        Y_hat[ii, :, fk] = y_i[0]
        return Y_hat

    results = {}
    for method_name in methods:
        method_key = method_name.replace("_", "-")
        if method_key == "inner-product":
            mfpca = MFPCA(n_components=R_final, method=method_key)
            mfpca.fit(mfd, method_smoothing="PS")
            transform_method = "InnPro"
            scores = mfpca.transform(method=transform_method)
        elif method_key == "covariance":
            mfpca = MFPCA(n_components=R_final, method=method_key, univariate_expansions=uni_exp)
            mfpca.fit(mfd, method_smoothing="PS")
            transform_method = "NumInt"
            scores = mfpca.transform(mfd, method=transform_method)

        mfd_recon = mfpca.inverse_transform(scores)
        Y_hat = _reconstruct_on_grid(mfd_recon)
        results[method_name] = dict(
            Y_hat=Y_hat,
            scores_subject=np.asarray(scores),
            R_final=int(R_final),
            transform_method=transform_method
        )
        print(f"[mFPCA {method_name}] Retained R_final={R_final} components using {transform_method} scoring.")

    if (Y_truth is not None and plot_examples > 0 and
            "covariance" in results and "inner-product" in results):
        _plot_mfpca_comparison(
            Y_truth,
            results["covariance"]["Y_hat"],
            results["inner-product"]["Y_hat"],
            survey_times,
            n_subjects=plot_examples,
            n_features=plot_features,
            savepath=plot_savepath,
            seed=plot_seed
        )

    return results


# ---------------- I/O helpers ----------------
def save_full_dataset_npz(save_dir, Y_full, Y_obs, observed_mask, truths, tag):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"dataset_full_{tag}.npz")
    np.savez_compressed(path,
                        Y_full=Y_full, Y_obs=Y_obs, observed_mask=observed_mask,
                        W1_true=truths["W1_true"], W2_true=truths["W2_true"],
                        Z1_true=truths["Z1_true"], Z2_true=truths["Z2_true"],
                        survey_times=truths["survey_times"],
                        ell_true_vec=(None if truths["ell_true_vec"] is None else np.asarray(truths["ell_true_vec"])),
                        sigma2_true=np.array([truths["sigma2_true"]], float))
    return path

def compute_metrics(Y_full, Y_hat, observed_mask):
    mse        = _masked_mse(Y_hat, Y_full, ~observed_mask)
    return dict(mse_missing=mse, num_missing=int(np.sum(~observed_mask)))

def _make_metric_row(method_name, metrics, seed):
    """Compact row for partial metrics CSV."""
    return dict(method=method_name,
                seed=int(seed),
                num_missing=int(metrics.get("num_missing", 0)),
                mse_missing=float(metrics.get("mse_missing", np.nan)))

def _write_metric_rows(path, rows):
    """Write metric rows with only the requested columns."""
    cols = ["method", "seed", "num_missing", "mse_missing"]
    compact = [{col: row.get(col, np.nan) for col in cols} for row in rows]
    pd.DataFrame(compact, columns=cols).to_csv(path, index=False)

def _plot_mfpca_comparison(Y_truth, recon_cov, recon_inn, survey_times,
                           n_subjects=5, n_features=2, savepath=None, seed=None):
    """Plot truth vs. two mFPCA reconstructions for selected subjects/features."""
    if Y_truth is None or recon_cov is None or recon_inn is None:
        return
    n, J, p = Y_truth.shape
    if n == 0 or p == 0 or J == 0:
        return
    n_subjects = min(max(1, int(n_subjects)), n)
    n_features = min(max(1, int(n_features)), p)
    rng = np.random.default_rng(seed)
    subj_idx = rng.choice(n, size=n_subjects, replace=False)
    feat_idx = np.arange(n_features)

    fig, axes = plt.subplots(nrows=n_subjects, ncols=n_features,
                             figsize=(4 * n_features, 3 * n_subjects),
                             squeeze=False)
    colors_truth = "black"
    colors_cov = (0.85, 0.1, 0.1, 1.0)
    colors_inn = (0.1, 0.6, 0.8, 1.0)
    for r, i in enumerate(subj_idx):
        for c, k in enumerate(feat_idx):
            ax = axes[r, c]
            ax.plot(survey_times, Y_truth[i, :, k], label="Truth", color=colors_truth, linewidth=2)
            ax.plot(survey_times, recon_cov[i, :, k], label="mFPCA covariance", color=colors_cov, linestyle="--")
            ax.plot(survey_times, recon_inn[i, :, k], label="mFPCA inner-product", color=colors_inn, linestyle=":")
            ax.set_title(f"Subject {i}, Feature {k}")
            ax.set_xlabel("Time")
            ax.set_ylabel("Value")
            if r == 0 and c == 0:
                ax.legend()
    plt.tight_layout()
    if savepath:
        save_dir = os.path.dirname(savepath)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
        fig.savefig(savepath, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser(description="PPCA and mFPCA comparison simulation pipeline.")
    ap.add_argument(
        "--savepath",
        type=str,
        default=os.path.join("sim_results", "sim_comparison_section_3_2"),
    )
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--J", type=int, default=5)
    ap.add_argument("--p", type=int, default=20)
    ap.add_argument("--d1", type=int, default=2)
    ap.add_argument("--d2", type=int, default=2)
    ap.add_argument("--rate", type=float, default=0.9)
    ap.add_argument("--truesigma2", type=float, default=0.25)
    ap.add_argument("--ell_true", type=float, nargs="+", default=[10.0])
    ap.add_argument("--kernel_method", type=str, default="gp_rbf_single_ell",
                    choices=["gp_rbf_single_ell","gp_rbf_multi_ell","gp_matern52_single_ell","gp_matern52_multi_ell","gp_iid"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--survey_times", type=float, nargs="+", default=None)
    ap.add_argument("--ncomp_uni", type=int, default=5,
                    help="Number of univariate FPCA components per feature for mFPCA (FDApy).")
    ap.add_argument("--plot_examples", type=int, default=0,
                    help="If >0, number of subjects to plot when comparing mFPCA reconstructions to truth.")
    ap.add_argument("--plot_features", type=int, default=2,
                    help="Number of features to include per subject in the comparison plot.")
    ap.add_argument("--plot_savepath", type=str, default=None,
                    help="Optional path for saving the mFPCA comparison plot. If omitted, defaults to savepath when plotting.")
    ap.add_argument("--plot_seed", type=int, default=42,
                    help="Random seed for selecting subjects in the comparison plot.")

    args = ap.parse_args()
    os.makedirs(args.savepath, exist_ok=True)
    ds_dir = os.path.join(args.savepath, "datasets")
    os.makedirs(ds_dir, exist_ok=True)

    # Build the unique TAG for filenames
    tag = _make_tag(args.n, args.J, args.p, args.d1, args.d2, args.rate, args.truesigma2,
                    args.kernel_method, args.ell_true, args.seed)

    # Run simulation and Python comparison methods.
    Y_full, Y_obs, observed, truths = simulate_data(
        n=args.n, J=args.J, p=args.p, d1=args.d1, d2=args.d2,
        rate=args.rate, truesigma2=args.truesigma2,
        ell_true=args.ell_true, survey_times=np.asarray(args.survey_times) if args.survey_times is not None else None,
        seed=args.seed, kernel_method=args.kernel_method
    )
    full_npz = save_full_dataset_npz(ds_dir, Y_full, Y_obs, observed, truths, tag)
    np.savez_compressed(os.path.join(args.savepath, f"truth_bundle_{tag}.npz"),
                        Y_full=Y_full, Y_obs=Y_obs, observed_mask=observed,
                        survey_times=truths["survey_times"], sigma2_true=args.truesigma2,
                        n=args.n, J=args.J, p=args.p, d1=args.d1, d2=args.d2,
                        rate=args.rate, kernel_method=args.kernel_method,
                        ell_true=np.asarray(args.ell_true), seed=args.seed, tag=tag)

    rows = []
    plot_path = args.plot_savepath
    if args.plot_examples > 0 and not plot_path:
        plot_path = os.path.join(args.savepath, f"mfpca_comparison_{tag}.png")

    try:
        Y_ppca, cs_ppca, cf_ppca, ss_ppca, lat_ppca, C_ppca = fit_baseline_ppca(Y_obs, d_rank=args.d1+args.d2)
        m_ppca = compute_metrics(Y_full, Y_ppca, observed)
        rows.append(_make_metric_row("PPCA_iid", m_ppca, args.seed))
    except Exception as e:
        warnings.warn(f"PPCA_iid FAILED for seed {args.seed}. Reason: {e}")
        rows.append({
            "method": "PPCA_iid",
            "seed": int(args.seed),
            "num_missing": int(np.sum(~observed)),
            "mse_missing": float("nan")
        })

    if _HAVE_FDAPY:
        try:
            mfpca_results = fit_mfpca_fdapy(
                Y_obs, truths["survey_times"], args.d1, args.d2, args.ncomp_uni,
                methods=("covariance", "inner-product"),
                Y_truth=Y_full,
                plot_examples=args.plot_examples,
                plot_features=args.plot_features,
                plot_savepath=plot_path,
                plot_seed=args.plot_seed
            )
            for method_name, res in mfpca_results.items():
                try:
                    method_label = f"mFPCA_{method_name}"
                    m_metrics = compute_metrics(Y_full, res["Y_hat"], observed)
                    rows.append(_make_metric_row(method_label, m_metrics, args.seed))
                except Exception as e:
                    warnings.warn(f"mFPCA method {method_name} failed: {e}")
                    rows.append({
                        "method": method_label,
                        "seed": int(args.seed),
                        "num_missing": int(np.sum(~observed)),
                        "mse_missing": float("nan")
                    })
        except Exception as e:
            warnings.warn(f"mFPCA failed: {e}")
    else:
        warnings.warn("Skipping mFPCA; FDApy not installed.")

    partial_path = os.path.join(args.savepath, f"compare_metrics_partial_{tag}.csv")
    _write_metric_rows(partial_path, rows)

    print("\n[SIMULATION DONE]")
    print(f"  Full NPZ:      {full_npz}")
    print(f"  Truth bundle:  {os.path.join(args.savepath, f'truth_bundle_{tag}.npz')}")
    print(f"  Metrics CSV:   {partial_path}")


if __name__ == "__main__":
    main()
