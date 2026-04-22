# simulate_and_fit_lds_diag_comparison_ar2.py

import numpy as np
import time
import pandas as pd
import argparse
import os
import warnings
import matplotlib.pyplot as plt

from hppca import fit_hppca

# --- Imports for comparison models ---
try:
    from pyppca import ppca
    _HAVE_PPCA = True
except ImportError:
    warnings.warn("pyppca not installed. PPCA methods will be skipped.")
    _HAVE_PPCA = False

try:
    from FDApy.representation import (
        DenseArgvals, IrregularArgvals, IrregularValues,
        IrregularFunctionalData, MultivariateFunctionalData
    )
    from FDApy.preprocessing import MFPCA
    _HAVE_FDAPY = True
except ImportError:
    warnings.warn("FDApy not installed. mFPCA will be skipped.")
    _HAVE_FDAPY = False


# =============================================================================
# Utilities
# =============================================================================

def _masked_mse(pred, truth, mask):
    """mask=True entries will be used."""
    if mask.sum() == 0:
        return float("nan")
    d = pred[mask] - truth[mask]
    return float(np.mean(d * d))


# =============================================================================
# [VISUALIZATION] Plotting Function
# =============================================================================

def visualize_reconstructions(
    Y_true, Y_obs, time_points, predictions_dict,
    n_subs_to_plot=3, n_feats_to_plot=2,
    save_path=None
):
    """
    Plots a grid of trajectories: Rows=Subjects, Cols=Features.
    Visualizes Truth, Observed Data, and Method Reconstructions.
    """
    n, J, p = Y_true.shape

    obs_counts = np.sum(np.isfinite(Y_obs), axis=(1, 2))
    valid_subs = np.where(obs_counts > 2)[0]

    if len(valid_subs) >= n_subs_to_plot:
        rng_plot = np.random.default_rng(999)
        sub_indices = rng_plot.choice(valid_subs, n_subs_to_plot, replace=False)
    else:
        sub_indices = np.arange(min(n, n_subs_to_plot))

    feat_indices = np.arange(min(p, n_feats_to_plot))

    styles = {
        "Truth":    {"c": "gray",    "ls": "-",  "lw": 3.5, "alpha": 0.4, "label": "Truth", "zorder": 1},
        "Observed": {"c": "black",   "marker": "o", "s": 30, "label": "Observed", "zorder": 10},
        "HPPCA":    {"c": "#d62728", "ls": "-",  "lw": 2.5, "alpha": 0.9, "label": "HPPCA (Ours)", "zorder": 5},
        "PPCA":     {"c": "#1f77b4", "ls": "--", "lw": 1.5, "alpha": 0.7, "label": "PPCA", "zorder": 3},
        "mFPCA":    {"c": "#2ca02c", "ls": "-.", "lw": 1.8, "alpha": 0.8, "label": "mFPCA", "zorder": 4},
    }

    fig, axes = plt.subplots(
        nrows=len(sub_indices),
        ncols=len(feat_indices),
        figsize=(5 * len(feat_indices), 3 * len(sub_indices)),
        squeeze=False, sharex=True
    )

    for r, i in enumerate(sub_indices):
        for c, k in enumerate(feat_indices):
            ax = axes[r, c]

            ax.plot(time_points, Y_true[i, :, k], **styles["Truth"])

            m = np.isfinite(Y_obs[i, :, k])
            if np.any(m):
                ax.scatter(
                    time_points[m], Y_obs[i, m, k],
                    color=styles["Observed"]["c"], s=styles["Observed"]["s"],
                    zorder=styles["Observed"]["zorder"],
                    label=styles["Observed"]["label"] if r == 0 and c == 0 else ""
                )

            for method_name, Y_hat in predictions_dict.items():
                if Y_hat is None:
                    continue
                if np.all(np.isnan(Y_hat[i, :, k])):
                    continue

                st = {"c": "purple", "ls": ":", "lw": 1, "zorder": 2}
                if "HPPCA" in method_name:
                    st = styles["HPPCA"]
                elif "PPCA" in method_name:
                    st = styles["PPCA"]
                elif "mFPCA" in method_name:
                    st = styles["mFPCA"]

                lbl = method_name if (r == 0 and c == 0) else ""
                ax.plot(
                    time_points, Y_hat[i, :, k],
                    color=st["c"], linestyle=st["ls"], linewidth=st["lw"],
                    alpha=st.get("alpha", 1), zorder=st["zorder"], label=lbl
                )

            if r == 0:
                ax.set_title(f"Feature {k}", fontweight="bold")
            if c == 0:
                ax.set_ylabel(f"Subject {i}\nValue")
            if r == len(sub_indices) - 1:
                ax.set_xlabel("Time (0-1)")
            ax.grid(True, linestyle=":", alpha=0.5)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(), by_label.keys(),
        loc="upper center", bbox_to_anchor=(0.5, 1.05),
        ncol=5, frameon=False, fontsize=11
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        print(f"Comparison plot saved to: {save_path}")


# =============================================================================
# SCENARIO: Standard Linear Dynamical System (LDS, AR(2))
# =============================================================================

def _ar2_spectral_radius(phi1, phi2):
    """Spectral radius of the AR(2) companion matrix."""
    companion = np.array([[phi1, phi2], [1.0, 0.0]], dtype=float)
    eigvals = np.linalg.eigvals(companion)
    return float(np.max(np.abs(eigvals)))


def simulate_standard_lds_ar2(n, J, p, d1, d2, phi1, phi2, rate, seed):
    """
    Simulates LDS data with a single AR(2) pair (phi1, phi2)
    shared by all latent dimensions. Rigorously initialized to maintain unit variance.
    """
    rng = np.random.default_rng(seed)
    n_latent = int(d1 + d2)

    rad = _ar2_spectral_radius(phi1, phi2)
    if rad >= 1.0:
        warnings.warn(
            f"AR(2) coefficients are non-stationary (spectral_radius={rad:.3f}). "
            "Simulated trajectories may explode."
        )

    C = rng.standard_normal((p, n_latent))
    C, _ = np.linalg.qr(C)

    Y_full = np.zeros((n, J, p))
    obs_noise_std = 0.5

    # 1. ensure the AR(2) process has unit variance by computing the innovation variance
    var_w = ((1.0 + phi2) / (1.0 - phi2)) * ((1.0 - phi2)**2 - phi1**2)
    if var_w <= 0:
        raise ValueError(f"Parameters phi1={phi1}, phi2={phi2} do not define a valid stationary AR(2) process (variance <= 0).")
    Q_std = np.sqrt(var_w)

    # 2. compute the implied lag-1 autocorrelation (rho1) for proper initialization of Z[1]
    rho1 = phi1 / (1.0 - phi2)

    for i in range(n):
        Z = np.zeros((J, n_latent))
        
        # Step t=0: start from stationary distribution (standard normal) to avoid transient effects in variance
        Z[0] = rng.standard_normal(n_latent)

        if J > 1:
            # Step t=1: condition on Z[0] to ensure correct AR(2) structure from the start
            cond_mean = rho1 * Z[0]
            cond_std = np.sqrt(1.0 - rho1**2)
            Z[1] = cond_mean + rng.normal(0, cond_std, n_latent)

        # Steps t=2,...,J-1: follow the AR(2) recursion
        for t in range(2, J):
            innovation = rng.normal(0, Q_std, n_latent)
            Z[t] = phi1 * Z[t - 1] + phi2 * Z[t - 2] + innovation

        Y_full[i] = Z @ C.T

    Y_full += rng.normal(0, obs_noise_std, size=Y_full.shape)

    observed_mask = rng.random((n, J, p)) < (1.0 - rate)
    Y_obs = Y_full.copy()
    Y_obs[~observed_mask] = np.nan

    return Y_full, Y_obs, observed_mask


# =============================================================================
# Comparison Model Wrappers
# =============================================================================

def fit_baseline_ppca(Y_obs, d_rank, debug=True):
    n, J, p = Y_obs.shape
    Y = Y_obs.reshape(n * J, p)
    r = max(1, min(int(d_rank), min(n * J, p) - 1))
    C, ss, M, X, Ye = ppca(Y, r, dia=False)

    if debug:
        obs = np.isfinite(Y)
        if np.any(obs):
            max_abs_diff = np.nanmax(np.abs(Ye[obs] - Y[obs]))
            mean_abs_diff = np.nanmean(np.abs(Ye[obs] - Y[obs]))
            print(f"[PPCA DEBUG] observed-entry |Ye - Y|: max={max_abs_diff:.6e}, mean={mean_abs_diff:.6e}")
        else:
            print("[PPCA DEBUG] No observed entries found in Y (unexpected).")

    return Ye.reshape(n, J, p)


def fit_mfpca_fdapy_wrapper(Y_obs, survey_times, d1, d2, ncomp_uni=5, methods=("covariance",)):
    n, J, p = Y_obs.shape
    survey_times = np.asarray(survey_times, dtype=float)

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
            arg_d[i] = DenseArgvals({"input_dim_0": t_i})
            val_d[i] = y_i
        components.append(IrregularFunctionalData(IrregularArgvals(arg_d), IrregularValues(val_d)))

    mfd = MultivariateFunctionalData(components)

    R_final = max(1, min(int(d1 + d2), min(n - 1, p * ncomp_uni)))
    uni_exp = [{"method": "UFPCA", "n_components": ncomp_uni, "method_smoothing": "PS"} for _ in range(p)]

    results = {}
    for method_name in methods:
        method_key = method_name.replace("_", "-")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)

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
                else:
                    continue

            if np.any(np.isnan(scores)):
                print(f"  [Info] mFPCA {method_name} produced NaNs (Method failed on sparse data).")
                results[method_name] = np.full((n, J, p), np.nan)
                continue

            mfd_recon = mfpca.inverse_transform(scores)

            Y_hat = np.zeros((n, J, p))
            for fk in range(p):
                comp = mfd_recon.data[fk]
                if hasattr(comp, "values") and hasattr(comp.values, "data"):
                    arr = np.asarray(comp.values.data)
                    Y_hat[:, :, fk] = arr[:, :J] if arr.ndim == 2 else np.tile(arr, (n, 1))[:, :J]
                else:
                    for ii in range(n):
                        t_i = np.asarray(comp.argvals[ii]["input_dim_0"], dtype=float)
                        y_i = np.asarray(comp.values[ii] if hasattr(comp, "values") else comp.data[ii], dtype=float)
                        if t_i.size >= 2:
                            Y_hat[ii, :, fk] = np.interp(survey_times, t_i, y_i, left=y_i[0], right=y_i[-1])
                        else:
                            Y_hat[ii, :, fk] = y_i[0]

            results[method_name] = Y_hat
        except Exception as e:
            print(f"  [Warning] mFPCA {method_name} failed: {e}")
            results[method_name] = np.full((n, J, p), np.nan)

    return results


# =============================================================================
# Main Execution
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Compare HPPCA vs Others on AR(2)-LDS Simulated Data")

    # Simulation Args
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--J", type=int, default=10)
    parser.add_argument("--p", type=int, default=20)
    parser.add_argument("--rate", type=float, default=0.2, help="Missing rate")

    parser.add_argument("--phi1", type=float, default=0.8,
                        help="AR(2) phi1 shared by all latent dimensions.")
    parser.add_argument("--phi2", type=float, default=0.1,
                        help="AR(2) phi2 shared by all latent dimensions.")

    parser.add_argument("--seed", type=int, default=42)

    # Fitting Settings
    parser.add_argument("--d1", type=int, default=2)
    parser.add_argument("--d2", type=int, default=2)
    parser.add_argument("--iter", type=int, default=5)
    parser.add_argument("--n_cpus", type=int, default=3)
    parser.add_argument("--kernel_method", type=str, default="gp_rbf_single_ell")
    parser.add_argument("--init_method", type=str, default="algo2_cs")
    parser.add_argument("--ncomps_uni", type=int, default=5)

    parser.add_argument(
        "--savepath",
        type=str,
        default=os.path.join("sim_results", "sim_0318_results_LDSdiag_cmp_ar2"),
    )

    args = parser.parse_args()

    os.makedirs(args.savepath, exist_ok=True)
    print("--- Simulating Standard LDS (AR(2), single shared coefficient pair) ---")
    print(f"Using AR(2) coefficients: phi1={args.phi1}, phi2={args.phi2}")

    Y_full, Y_obs, mask = simulate_standard_lds_ar2(
        n=args.n, J=args.J, p=args.p,
        d1=args.d1, d2=args.d2,
        phi1=args.phi1, phi2=args.phi2,
        rate=args.rate,
        seed=args.seed
    )

    survey_times = np.linspace(0, 1, args.J)

    tag = (
        f"LDSdiagAR2_J{args.J}_p{args.p}_n{args.n}"
        f"_d1{args.d1}_d2{args.d2}"
        f"_phi1{args.phi1}_phi2{args.phi2}"
        f"_r{args.rate}_sd{args.seed}"
        f"_km{args.kernel_method}_init{args.init_method}"
        f"_iter{args.iter}"
    )

    results_list = []
    plot_predictions = {}

    missing_mask = ~mask

    # ---------------------------------------------------------
    # Fit HPPCA
    # ---------------------------------------------------------
    print("\n--- Fitting HPPCA ---")
    t0 = time.time()
    try:
        (
            W1, W2, s2, ell, _, _, Y_filled_hppca,
            iters_run, converged, _, _, _, _, _, _, _, _
        ) = fit_hppca(
            Y_obs=Y_obs, d1=args.d1, d2=args.d2, kernel_method=args.kernel_method,
            survey_times=survey_times, n_cpus=args.n_cpus, init_method=args.init_method,
            max_iter=args.iter, tol=1e-4, seed=args.seed,
            return_filled_dataframe=False, return_latent_dataframe=False
        )
        dt_hppca = time.time() - t0

        mse_missing = _masked_mse(Y_filled_hppca, Y_full, missing_mask)

        results_list.append({
            "method": f"HPPCA_{args.kernel_method}",
            "mse_missing": mse_missing,
            "time_sec": dt_hppca
        })
        print(f"HPPCA Done. MSE_missing: {mse_missing:.6f}")
    except Exception as e:
        print(f"HPPCA Failed: {e}")

    # ---------------------------------------------------------
    # Fit PPCA
    # ---------------------------------------------------------
    if _HAVE_PPCA:
        print("\n--- Fitting PPCA ---")
        try:
            t0 = time.time()
            Y_ppca_iid = fit_baseline_ppca(Y_obs, d_rank=args.d1 + args.d2, debug=True)

            mse_missing = _masked_mse(Y_ppca_iid, Y_full, missing_mask)

            results_list.append({
                "method": "PPCA_iid",
                "mse_missing": mse_missing,
                "time_sec": time.time() - t0
            })
            print(f"PPCA (iid) Done. MSE_missing: {mse_missing:.6f}")
        except Exception as e:
            print(f"PPCA (iid) Failed: {e}")

    # ---------------------------------------------------------
    # Fit mFPCA
    # ---------------------------------------------------------
    if _HAVE_FDAPY:
        print("\n--- Fitting mFPCA ---")
        t0 = time.time()
        try:
            methods_to_try = ("covariance", "inner-product")
            mfpca_res = fit_mfpca_fdapy_wrapper(
                Y_obs, survey_times, args.d1, args.d2,
                ncomp_uni=args.ncomps_uni,
                methods=methods_to_try
            )

            for m_name, Y_hat_mfpca in mfpca_res.items():
                mse_missing = _masked_mse(Y_hat_mfpca, Y_full, missing_mask)

                results_list.append({
                    "method": f"mFPCA_{m_name}",
                    "mse_missing": mse_missing,
                    "time_sec": time.time() - t0
                })

                print(f"mFPCA ({m_name}) Done. MSE_missing: {mse_missing:.6f}")
        except Exception as e:
            print(f"mFPCA Failed: {e}")

    # ---------------------------------------------------------
    # Save & Print
    # ---------------------------------------------------------
    df_res = pd.DataFrame(results_list)
    out_csv = os.path.join(args.savepath, f"results_{tag}.csv")
    df_res.to_csv(out_csv, index=False)

    print("\n=======================================")
    print("Comparison Finished.")
    print(f"CSV saved to: {out_csv}")
    cols_to_show = [c for c in ["method", "mse_missing", "time_sec"] if c in df_res.columns]
    print(df_res[cols_to_show])
    print("=======================================")


if __name__ == "__main__":
    main()
