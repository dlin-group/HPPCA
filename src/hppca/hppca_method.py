# hppca_method.py

import numpy as np
from scipy.linalg import block_diag, solve, inv, cholesky, LinAlgWarning
from numpy.linalg import svd, det
import time
import multiprocessing as mp
import traceback  # For more detailed error messages from multiprocessing
import warnings
from typing import Any, Tuple, Optional, Dict

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

# --- Import from em_worker.py ---
from .em_worker import init_pool_worker, _e_step_worker_for_pool
from .em_worker import _execute_e_step_logic  # This is the core E-step logic
from .em_worker import rbf_kernel, matern52_kernel, rbf_kernel_grad, matern52_kernel_grad1, matern52_kernel_grad2
from .em_worker import EPSILON_THRESH, ELL_MIN_THRESH, SQRT5  # Constants if needed by other parts
from .em_worker import is_observed_feature_mask, indices_for_Yp_values
from .em_worker import is_missing_feature_mask, indices_for_EYm_rows

# --- Import from hppca_mle_algorithm.py ---
from .hppca_mle_algorithms import fit_hppca_alg1, fit_hppca_alg2_cs


# ---- Helper Functions ----
def principal_angles(A, B):
    try:
        qA, _ = np.linalg.qr(A); qB, _ = np.linalg.qr(B)
        s = svd(qA.T @ qB, compute_uv=False); s = np.clip(s, -1.0, 1.0)
        return np.arccos(s) * 180 / np.pi
    except np.linalg.LinAlgError: return np.full(min(A.shape[1] if A.ndim > 1 else 1, B.shape[1] if B.ndim > 1 else 1), np.nan)

def max_relative_error(M_old, M_new, eps=EPSILON_THRESH):
    if np.isscalar(M_old) or np.isscalar(M_new):
        diff, denom = np.abs(M_new - M_old), np.abs(M_old) + eps
        return diff / denom if denom > 0 else diff 
    if M_old.shape != M_new.shape:
        if np.isscalar(M_old) and M_new.size == 1: M_new = M_new.item()
        elif np.isscalar(M_new) and M_old.size == 1: M_old = M_old.item()
        else: return np.inf
    diff, denom = np.abs(M_new - M_old), np.abs(M_old) + eps
    denom[denom == 0] = eps; return np.max(diff / denom)


def _invert_kernel_with_jitter(
    K: np.ndarray,
    jitter_schedule: tuple[float, ...] = (1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5),
) -> np.ndarray:
    """
    Numerically stable inverse for kernel/covariance matrices.
    Tries increasing diagonal jitter and falls back to pseudo-inverse.
    """
    K = np.asarray(K, dtype=float)
    n = K.shape[0]
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    I = np.eye(n)
    for jitter in jitter_schedule:
        try:
            K_try = K.copy()
            K_try[np.diag_indices_from(K_try)] += jitter
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                return solve(K_try, I, assume_a='sym', check_finite=False)
        except (np.linalg.LinAlgError, LinAlgWarning):
            continue
    return np.linalg.pinv(K)

def construct_index_mapping(Y_list_map):
    """
    Build per-participant, per-survey index maps that tell us:
      - which feature rows are observed vs. missing,
      - how to map each observed row to its position in the compact Y_p vector,
      - how to map each missing row to its position in the compact E[Y_m · Z^T] rows.

    Inputs
    ------
    Y_list_map : list of np.ndarray
        Length n_eff list. For participant i, Yi_map = Y_list_map[i] has shape (p, J_i),
        where rows are features and columns are this participant's observed survey waves.
        Entries may have NaNs within a column for missing features.

    Returns
    -------
    idx_map_list : list[list[np.ndarray]]
        For each participant i, a list of length J_i; each element is an idx_map
        of shape (p, 3) for survey j:
          - idx_map[:, 0] = np.arange(p)                # original feature row indices
          - idx_map[r, 1] = position of row r in Y_p    # 0..(#observed-1) or NaN if missing
          - idx_map[r, 2] = position of row r in Y_m    # 0..(#missing-1)  or NaN if observed

    Side effects (globals filled for fast access in E-step & G-construction)
    -----------------------------------------------------------------------
    is_observed_feature_mask[i][j] : bool array of shape (p,)
        True where feature row r is observed in participant i, survey j.
    indices_for_Yp_values[i][j] : int array of shape (#observed,)
        The positions 0..(#observed-1) corresponding to observed rows (used to index Y_p).
    is_missing_feature_mask[i][j] : bool array of shape (p,)
        True where feature row r is missing in participant i, survey j.
    indices_for_EYm_rows[i][j] : int array of shape (#missing,)
        The positions 0..(#missing-1) corresponding to missing rows (used to index rows of E[Y_m Z^T]).

    Notes
    -----
    - These structures let us vectorize G-construction:
        * Observed rows:   G[obs, :]  = outer(Y_p, E[Z]^T)
        * Missing rows:    G[miss, :] = E[Y_m Z^T]  (already in compact (rows=#missing) form)
      We place each compact piece back into the full p×d matrices using the masks and indices.
    - This function clears and rebuilds the global mask/index lists each call to avoid
      stale state when fitting multiple datasets in the same Python process.
    """

    # Reset global caches so repeated fit calls in one Python process do not
    # append stale mappings and accidentally index the wrong participant.
    is_observed_feature_mask.clear()
    indices_for_Yp_values.clear()
    is_missing_feature_mask.clear()
    indices_for_EYm_rows.clear()

    idx_map_list = []
    for Yi_map in Y_list_map:

        p_map, Ji_map = Yi_map.shape
        sublist_i = []
        obs_mask, Yp_index, miss_mask, EYm_index= [], [], [], []
        feature_idx = np.arange(p_map, dtype=float)

        for j_idx_map in range(Ji_map):
            # idx_map columns:
            #   col 0: original feature row index (0..p-1)
            #   col 1: index within observed slice Y_p (NaN if missing)
            #   col 2: index within missing slice Y_m (NaN if observed)
            Yij_map = Yi_map[:, j_idx_map]
            obs_idx = ~np.isnan(Yij_map)   # True where feature r was observed at survey j
            miss_idx = ~obs_idx            # True where feature r was missing at survey j

            # Keep the same output shape/semantics as before.
            idx_map = np.full((p_map, 3), np.nan, dtype=float)
            idx_map[:, 0] = feature_idx
            n_obs = int(np.sum(obs_idx))
            n_miss = p_map - n_obs
            if n_obs > 0:
                idx_map[obs_idx, 1] = np.arange(n_obs, dtype=float)
            if n_miss > 0:
                idx_map[miss_idx, 2] = np.arange(n_miss, dtype=float)

            sublist_i.append(idx_map)
            obs_mask.append(obs_idx)
            Yp_index.append(np.arange(n_obs, dtype=np.int32))
            miss_mask.append(miss_idx)
            EYm_index.append(np.arange(n_miss, dtype=np.int32))

        idx_map_list.append(sublist_i)
        is_observed_feature_mask.append(obs_mask)
        indices_for_Yp_values.append(Yp_index)
        is_missing_feature_mask.append(miss_mask)
        indices_for_EYm_rows.append(EYm_index)
    return idx_map_list

# ---- Initialization ----
def initialize_parameters(p_init, d1_init, d2_init, J_init, survey_times_init, kernel_method_init):
    W1, W2, sigma2 = np.random.randn(p_init, d1_init)*0.1, np.random.randn(p_init, d2_init)*0.1, 1.0
    time_range = np.max(survey_times_init) - np.min(survey_times_init) if J_init > 1 else 1.0
    ell_init_val = max(time_range / 3.0, 5.0) 
    if "single_ell" in kernel_method_init: ell_param = ell_init_val
    elif "multi_ell" in kernel_method_init: ell_param = np.full(d2_init, ell_init_val)
    elif "iid" in kernel_method_init:
        ell_param = None
    else: raise ValueError(f"Unknown kernel_method for init: {kernel_method_init}")
    return W1, W2, sigma2, ell_param

# ---- Wrapper for Serial E-step execution (using the logic from em_worker.py) ----
# This function is used if n_cpus=1 or if multiprocessing fails
def _process_participant_e_step_serial_wrapper(args_tuple):
    i_w, Y_lw, W1w, W2w, s2w, ell_pw, kmeth_w, st_w_full, psi_w_list, iml_w_list, Yobs_w_full, poi_w_list = args_tuple
    
    # Extract data specific to participant i_w for _execute_e_step_logic
    Yi = Y_lw[i_w]
    oi = poi_w_list[i_w]
    pw = W1w.shape[0]
    Jiw = Yi.shape[1] if Yi.ndim == 2 and Yi.shape[1] > 0 else 0

    if Jiw == 0:
        d1w, d2w = W1w.shape[1], W2w.shape[1]
        return i_w,0.,0.,np.zeros((pw,d1w)),np.zeros((pw,d2w)),np.zeros((d1w,d1w)),np.zeros((d1w,d2w)),np.zeros((d2w,d2w)),None,None,np.zeros(d1w),np.zeros((d2w,0))

    orig_idx_iw = psi_w_list[i_w]
    map_li_w = iml_w_list[i_w]
    d1w, d2w = W1w.shape[1], W2w.shape[1]
    
    return _execute_e_step_logic(i_w, Yi, oi, W1w, W2w, s2w, ell_pw, kmeth_w,
                                 st_w_full, orig_idx_iw, map_li_w, Yobs_w_full,
                                 pw, Jiw, d1w, d2w)


# ---- Main EM Step Function (onestep_em_gp) ---
def onestep_em_gp(
    Y_list,
    W1,
    W2,
    sigma2,
    ell_param,
    kernel_method,
    survey_times,
    participant_survey_indices,
    index_mapping_list,
    Y_obs,
    participant_original_indices,
    n_cpus=1,
):
    """
    Run one EM step (E over all participants, then M) for the GP-HPPCA model.

    Parameters
    ----------
    Y_list : list[np.ndarray]
        Length-n list; for participant i, Y_list[i] has shape (p, J_i).
        Rows are features (p), columns are that participant's observed survey waves (J_i).
        Entries may be NaN for missing features at a wave.
    W1 : np.ndarray
        Loading matrix for Z1 (survey-invariant latent), shape (p, d1).
    W2 : np.ndarray
        Loading matrix for Z2 (time-varying latent), shape (p, d2).
    sigma2 : float
        Observation noise variance.
    ell_param : float | np.ndarray
        GP length-scale parameter(s).
        - If kernel_method contains "single_ell": a scalar float.
        - If kernel_method contains "multi_ell": array of shape (d2,).
        - If kernel_method is "gp_iid": unused (typically None).
    kernel_method : str
        One of {"gp_rbf_single_ell", "gp_rbf_multi_ell",
                "gp_matern52_single_ell", "gp_matern52_multi_ell",
                "gp_iid"}.
    survey_times : np.ndarray
        Global timeline of possible survey times, shape (J,).
    participant_survey_indices : list[np.ndarray]
        For participant i, an integer array giving which indices into `survey_times`
        were actually observed for that participant (length J_i).
    index_mapping_list : list[list[np.ndarray]]
        Output of `construct_index_mapping(Y_list)`. For participant i, a length-J_i
        list; each entry is a (p,3) matrix that maps each row to compact observed/missing
        indices used for vectorized G construction.
    Y_obs : np.ndarray
        Original observed data tensor with NaNs, shape (n_orig, J, p). Used only to
        produce a per-participant working copy and to later reassemble imputed values.
    participant_original_indices : list[int]
        Maps Y_list index (0..n-1) back to the original participant index (0..n_orig-1)
        so we know which slice of Y_obs each participant belongs to.
    n_cpus : int, optional (default=1)
        If n_cpus>1, use multiprocessing for E-steps.

    Returns
    -------
    W1_new : np.ndarray
        Updated W1, shape (p, d1).
    W2_new : np.ndarray
        Updated W2, shape (p, d2).
    sigma2_new : float
        Updated noise variance.
    ell_param_new : float | np.ndarray
        Updated length-scale(s), same type/shape as input ell_param.
    EZ1i_list : list[np.ndarray]
        For participant i, E[Z1_i] of shape (d1,).
    EZ2ij_list : list[np.ndarray]
        For participant i, E[Z2_ij] stacked as a (d2, J_i) array.
    Y_filled_list : list
        Legacy placeholder list (entries are currently None) kept for backward
        compatibility of return tuple shape.
    """
    n = len(Y_list)                 # effective participants
    p, d1 = W1.shape
    d2 = W2.shape[1]

    tot_eyy_s2, tot_den_s2 = 0.0, 0.0
    tot_sG1 = np.zeros((p, d1))
    tot_sG2 = np.zeros((p, d2))
    Om_b00 = np.zeros((d1, d1))
    Om_b01 = np.zeros((d1, d2))
    Om_b11 = np.zeros((d2, d2))

    # For ell updates (store per-participant S_Z2 terms)
    S_Z2i_list = []
    S_Z2ir_lists = [[] for _ in range(d2)]

    # Per-participant outputs for caller
    Y_filled_list = [None] * n
    EZ1i_list = [None] * n
    EZ2ij_list = [None] * n

    processed_idx = []

    results = []
    if n_cpus > 1 and n > 0:
        init_args_tuple = (
            Y_list, W1, W2, sigma2, ell_param, kernel_method,
            survey_times, participant_survey_indices,
            index_mapping_list, Y_obs, participant_original_indices
        )
        with mp.Pool(processes=n_cpus, initializer=init_pool_worker, initargs=init_args_tuple) as pool:
            try:
                participant_indices = list(range(n))
                chunksize = max(1, n // (n_cpus * 4))
                results = pool.map(_e_step_worker_for_pool, participant_indices, chunksize=chunksize)
            except Exception as e:
                print(f"Parallel E-step Error: {e}")
                print("Traceback for parallel error:")
                traceback.print_exc()
                print("Falling back to serial execution for E-step.")
                results = [
                    _process_participant_e_step_serial_wrapper(
                        (
                            i, Y_list, W1, W2, sigma2, ell_param, kernel_method,
                            survey_times, participant_survey_indices, index_mapping_list,
                            Y_obs, participant_original_indices
                        )
                    )
                    for i in range(n)
                ]
    else:
        if n > 0:
            results = [
                _process_participant_e_step_serial_wrapper(
                    (
                        i, Y_list, W1, W2, sigma2, ell_param, kernel_method,
                        survey_times, participant_survey_indices, index_mapping_list,
                        Y_obs, participant_original_indices
                    )
                )
                for i in range(n)
            ]

    # ---- aggregate E-step results across participants ----
    for tup in results:
        if tup is None:
            continue
        try:
            (pi, eyy_s2_i, den_s2_i,
             sG1_i, sG2_i,
             EZ1Z1_i, sZ1Z2T_i, sZ2Z2T_i,
             S_Z2_i, Yfilled_i, EZ1_i, EZ2_i) = tup

            # For kernels using ell parameter we require S_Z2_i
            if S_Z2_i is None and "ell" in kernel_method:
                continue
        except (TypeError, ValueError):
            continue

        tot_eyy_s2 += eyy_s2_i
        tot_den_s2 += den_s2_i
        tot_sG1 += sG1_i
        tot_sG2 += sG2_i

        Ji_val = Y_list[pi].shape[1] if Y_list[pi].ndim == 2 else 0
        if Ji_val > 0:
            Om_b00 += Ji_val * EZ1Z1_i
            Om_b01 += sZ1Z2T_i
            Om_b11 += sZ2Z2T_i

        processed_idx.append(pi)

        # collect S_Z2 for ell update
        if S_Z2_i is not None:
            if "single_ell" in kernel_method:
                S_Z2i_list.append(S_Z2_i)
            elif "multi_ell" in kernel_method and isinstance(S_Z2_i, list) and len(S_Z2_i) == d2:
                for r in range(d2):
                    S_Z2ir_lists[r].append(S_Z2_i[r])

        # stash participant-level expectations
        if pi < n:
            Y_filled_list[pi] = Yfilled_i
            EZ1i_list[pi] = EZ1_i
            EZ2ij_list[pi] = EZ2_i

    # ---- M-step: update W1, W2 ----
    Om_b10 = Om_b01.T
    G_tot = np.hstack([tot_sG1, tot_sG2])
    try:
        Om_b = np.block([[Om_b00, Om_b01], [Om_b10, Om_b11]])
        W_tilde = solve(Om_b, G_tot.T, assume_a='sym', check_finite=False).T
        W1_new, W2_new = W_tilde[:, :d1], W_tilde[:, d1:]
    except np.linalg.LinAlgError:
        try:
            Om_b_pinv = np.linalg.pinv(Om_b)
            W_tilde = G_tot @ Om_b_pinv
            W1_new, W2_new = W_tilde[:, :d1], W_tilde[:, d1:]
        except np.linalg.LinAlgError:
            print("W update failed catastrophically. Using old W.")
            W1_new, W2_new = W1.copy(), W2.copy()

    # ---- M-step: update sigma2 ----
    sigma2_new = sigma2
    if tot_den_s2 > EPSILON_THRESH:
        W_tilde_s2 = np.hstack([W1_new, W2_new])
        cross_term = float(np.sum(W_tilde_s2 * G_tot))
        quad_term = float(np.trace((W_tilde_s2.T @ W_tilde_s2) @ Om_b))
        num_s2_new = tot_eyy_s2 - 2.0 * cross_term + quad_term
        sigma2_new = max(num_s2_new / tot_den_s2, EPSILON_THRESH * 10)

    # ---- M-step: update ell ----
    ell_param_new = ell_param.copy() if isinstance(ell_param, np.ndarray) else ell_param
    ktype = "matern52" if "matern52" in kernel_method else "rbf"
    kern_f = matern52_kernel if ktype == "matern52" else rbf_kernel
    grad1_f = matern52_kernel_grad1 if ktype == "matern52" else rbf_kernel_grad
    grad2_f = matern52_kernel_grad2 if ktype == "matern52" else None

    if "single_ell" in kernel_method:
        ell = ell_param
        g_ell, h_ell = 0.0, 0.0
        valid = 0
        kernel_cache = {}

        for idx_in_S, pi_idx in enumerate(processed_idx):
            if idx_in_S >= len(S_Z2i_list):
                continue
            S_Z2i = S_Z2i_list[idx_in_S]
            Ji = S_Z2i.shape[0]
            if Ji == 0:
                continue

            if ell <= ELL_MIN_THRESH:
                continue
            survey_key = tuple(np.asarray(participant_survey_indices[pi_idx], dtype=int).tolist())
            cached = kernel_cache.get(survey_key)
            if cached is None and survey_key not in kernel_cache:
                ti = survey_times[np.asarray(survey_key, dtype=int)]
                Sigma_i, dist_or_r = kern_f(ti, ell)
                dSigma_dell = grad1_f(ti, ell, Sigma_i, dist_or_r)
                Sigma_inv = _invert_kernel_with_jitter(Sigma_i)

                if ktype == "rbf":
                    sqdist = dist_or_r
                    d2Sigma = dSigma_dell * (sqdist / (ell**3) - (3.0 / ell))
                else:  # matern52
                    d2Sigma = grad2_f(ti, ell, Sigma_i, dist_or_r)

                dSigma_Sinv_dSigma = dSigma_dell @ Sigma_inv @ dSigma_dell
                Sinv_dSigma = Sigma_inv @ dSigma_dell
                tr_sinvds_sq = np.trace(Sinv_dSigma @ Sinv_dSigma)
                cached = (Sigma_i, Sigma_inv, dSigma_dell, d2Sigma, dSigma_Sinv_dSigma, tr_sinvds_sq)
                kernel_cache[survey_key] = cached
            elif cached is None:
                continue

            Sigma_i, Sigma_inv, dSigma_dell, d2Sigma, dSigma_Sinv_dSigma, tr_sinvds_sq = cached
            A = Sigma_inv @ (S_Z2i - d2 * Sigma_i) @ Sigma_inv
            g_ell += 0.5 * np.trace(A @ dSigma_dell)
            term1 = np.trace(A @ d2Sigma)
            term2 = -2 * np.trace(A @ dSigma_Sinv_dSigma)
            term3 = -d2 * tr_sinvds_sq
            h_ell += 0.5 * (term1 + term2 + term3)
            valid += 1

        if valid > 0:
            # --- ROBUST UPDATE START ---
            # 1. Check for zero/NaN hessian to avoid division by zero
            if abs(h_ell) < 1e-12 or not np.isfinite(h_ell) or not np.isfinite(g_ell):
                # Fallback: gradient descent step if Newton fails
                update_step = -1e-5 * np.sign(g_ell) * ell 
            else:
                # Standard Newton step
                update_step = -g_ell / h_ell

            # 2. Clamp the step size (Trust Region)
            # Prevent the parameter from changing by more than 50% in a single step
            # This stops explosive updates when h_ell is tiny but non-zero.
            max_change = 0.5 * ell
            update_step = np.clip(update_step, -max_change, max_change)

            # 3. Apply update with floor
            ell_param_new = max(ell + update_step, ELL_MIN_THRESH * 10)
            # --- ROBUST UPDATE END ---

    elif "multi_ell" in kernel_method:
        ell_vec = ell_param
        ell_vec_new = ell_param.copy()
        for r in range(d2):
            ell_r = ell_vec[r]
            g_r, h_r = 0.0, 0.0
            valid_r = 0
            kernel_cache_r = {}

            for idx_in_S, pi_idx in enumerate(processed_idx):
                if r >= len(S_Z2ir_lists) or idx_in_S >= len(S_Z2ir_lists[r]):
                    continue
                S_Z2ir = S_Z2ir_lists[r][idx_in_S]
                Ji = S_Z2ir.shape[0]
                if Ji == 0:
                    continue

                if ell_r <= ELL_MIN_THRESH:
                    continue
                survey_key = tuple(np.asarray(participant_survey_indices[pi_idx], dtype=int).tolist())
                cached_r = kernel_cache_r.get(survey_key)
                if cached_r is None and survey_key not in kernel_cache_r:
                    ti = survey_times[np.asarray(survey_key, dtype=int)]
                    Sigma_r, dist_or_r = kern_f(ti, ell_r)
                    dSigma_r = grad1_f(ti, ell_r, Sigma_r, dist_or_r)
                    Sigma_inv_r = _invert_kernel_with_jitter(Sigma_r)

                    if ktype == "rbf":
                        sqdist = dist_or_r
                        d2Sigma_r = dSigma_r * (sqdist / (ell_r**3) - (3.0 / ell_r))
                    else:  # matern52
                        d2Sigma_r = grad2_f(ti, ell_r, Sigma_r, dist_or_r)

                    dSigma_Sinv_dSigma_r = dSigma_r @ Sigma_inv_r @ dSigma_r
                    Sinv_dSigma_r = Sigma_inv_r @ dSigma_r
                    tr_sinvds_sq_r = np.trace(Sinv_dSigma_r @ Sinv_dSigma_r)
                    cached_r = (
                        Sigma_r,
                        Sigma_inv_r,
                        dSigma_r,
                        d2Sigma_r,
                        dSigma_Sinv_dSigma_r,
                        tr_sinvds_sq_r,
                    )
                    kernel_cache_r[survey_key] = cached_r
                elif cached_r is None:
                    continue

                Sigma_r, Sigma_inv_r, dSigma_r, d2Sigma_r, dSigma_Sinv_dSigma_r, tr_sinvds_sq_r = cached_r
                A_r = Sigma_inv_r @ (S_Z2ir - Sigma_r) @ Sigma_inv_r
                g_r += 0.5 * np.trace(A_r @ dSigma_r)
                term1 = np.trace(A_r @ d2Sigma_r)
                term2 = -2 * np.trace(A_r @ dSigma_Sinv_dSigma_r)
                term3 = -tr_sinvds_sq_r
                h_r += 0.5 * (term1 + term2 + term3)
                valid_r += 1

            if valid_r > 0:
                if abs(h_r) < 1e-15 or not np.isfinite(h_r) or not np.isfinite(g_r):
                    update_step = -1e-5 * np.sign(g_r) * ell_r
                else:
                    update_step = -g_r / h_r
                
                max_change = 0.5 * ell_r
                update_step = np.clip(update_step, -max_change, max_change)
                ell_vec_new[r] = max(ell_r + update_step, ELL_MIN_THRESH * 10)

        ell_param_new = ell_vec_new

    return W1_new, W2_new, sigma2_new, ell_param_new, EZ1i_list, EZ2ij_list, Y_filled_list



def EM_algorithm_gp(
    Y_list, W1_init, W2_init, sigma2_init, ell_param_init,
    kernel_method, survey_times, participant_survey_indices,
    Y_obs, participant_original_indices,
    records_meta: Optional[Dict[str, Any]] = None,
    n_cpus=1, max_iter=100, tol=1e-4, eps=EPSILON_THRESH,
):
    """
    Run the EM algorithm for HPPCA with a temporal prior on Z2 (GP or IID).

    Parameters
    ----------
    Y_list : list[np.ndarray]
        Length n_eff list; for participant i, Y_list[i] has shape (p, J_i).
        Rows are features, columns are that participant's observed survey times.
    W1_init : np.ndarray
        Initial loading matrix for Z1 (participant-level), shape (p, d1).
    W2_init : np.ndarray
        Initial loading matrix for Z2 (participant×time), shape (p, d2).
    sigma2_init : float
        Initial observation noise variance σ² (> 0).
    ell_param_init : float | np.ndarray | None
        Kernel hyperparameter(s) for the Z2 temporal prior:
          - "gp_*_single_ell": float length-scale ℓ.
          - "gp_*_multi_ell" : array of shape (d2,) with one ℓ per Z2 dim.
          - "gp_iid"         : None (unused).
    kernel_method : str
        Chooses the Z2 temporal prior: e.g.
          "gp_rbf_single_ell", "gp_rbf_multi_ell",
          "gp_matern52_single_ell", "gp_matern52_multi_ell",
          "gp_iid" (independent over time; no ℓ updates).
    survey_times : np.ndarray
        All survey times for indices 0..J-1, shape (J,).
    participant_survey_indices : list[np.ndarray]
        Length n_eff; for participant i, an integer array of original survey
        indices mapping columns of Y_list[i] back to survey_times.
    Y_obs : np.ndarray | None
        Full observed tensor of shape (n_orig, J, p) with NaNs where missing.
        If None, final imputation is returned as a long DataFrame using `records_meta`.
    participant_original_indices : list[int]
        Length n_eff; maps Y_list indices back to 0..n_orig-1 rows in Y_obs.
    records_meta : dict | None
        Required when Y_obs is None. Metadata from `_records_to_lists` for row-wise
        DataFrame imputation output.
    n_cpus : int, default 1
        Number of worker processes for CPU E-step (1 = serial).
    max_iter : int, default 10000
        Maximum number of EM iterations.
    tol : float, default 1e-4
        Convergence tolerance on the maximum relative change across parameters.
    eps : float, default EPSILON_THRESH
        Small epsilon for relative-error denominators.

    Returns
    -------
    W1f : np.ndarray         # (p, d1)
    W2f : np.ndarray         # (p, d2)
    s2f : float              # final σ²
    ell_pf : float | np.ndarray | None
    EZ1i_list : list[np.ndarray]     # length n_eff, each (d1,)
    EZ2ij_list : list[np.ndarray]    # length n_eff, each (d2, J_i)
    Y_filled : np.ndarray | pd.DataFrame
        If Y_obs is provided, returns a filled tensor (n_orig, J, p).
        If Y_obs is None, returns a filled long DataFrame from original tabular rows.
    iteration_num : int
    converged : bool
    """

    d1 = W1_init.shape[1]
    d2 = W2_init.shape[1]

    n_eff = len(Y_list)

    W1, W2, sigma2 = W1_init.copy(), W2_init.copy(), sigma2_init
    if "single_ell" in kernel_method:
        ell_param = ell_param_init
    elif "multi_ell" in kernel_method:
        ell_param = ell_param_init.copy()
    elif "iid" in kernel_method:
        ell_param = ell_param_init  # typically None; ℓ is unused in IID
    else:
        raise ValueError(f"Unknown kernel_method for EM: {kernel_method}")

    index_mapping_list = construct_index_mapping(Y_list)
    converged, iteration_num = False, 0
    EZ1i_final_list = [None] * n_eff
    EZ2ij_final_list = [None] * n_eff

    print(f"\nStarting EM (Kernel: {kernel_method}). n_eff={n_eff}, MaxIter:{max_iter}, Tol:{tol}")

    for iteration in range(max_iter):
        print(f"Iter {iteration+1}/{max_iter}...", end=" ")
        start_iter_time = time.time()
        W1_old, W2_old, sigma2_old = W1.copy(), W2.copy(), sigma2
        ell_param_old = ell_param.copy() if isinstance(ell_param, np.ndarray) else ell_param

        W1_new, W2_new, sigma2_new, ell_param_new, EZ1i_curr, EZ2ij_curr, _ = onestep_em_gp(
            Y_list, W1, W2, sigma2, ell_param, kernel_method,
            survey_times, participant_survey_indices, index_mapping_list,
            Y_obs, participant_original_indices, n_cpus=n_cpus
        )

        W1, W2, sigma2, ell_param = W1_new, W2_new, sigma2_new, ell_param_new
        EZ1i_final_list = EZ1i_curr
        EZ2ij_final_list = EZ2ij_curr

        if d1 == 0:
            W1mre = 0.0
        else:
            W1mre = max_relative_error(W1_old @ W1_old.T, W1 @ W1.T, eps)
        if d2 == 0:
            W2mre = 0.0
        else:
            W2mre = max_relative_error(W2_old @ W2_old.T, W2 @ W2.T, eps)

        s2mre = max_relative_error(sigma2_old, sigma2, eps)
        if ("iid" in kernel_method) or (ell_param_old is None and ell_param is None):
            ell_mre = 0.0
        else:
            ell_mre = max_relative_error(ell_param_old, ell_param, eps)
        max_chg = max(W1mre, W2mre, s2mre, ell_mre) if iteration > 0 else np.inf

        iter_time = time.time() - start_iter_time
        print(f"MREs — W1: {W1mre:.3e} | W2: {W2mre:.3e} | s2: {s2mre:.3e} | ell: {ell_mre:.3e}")
        print(f"s2={sigma2:.4f}, ell={ell_param}, MaxRelErr={max_chg:.3e}")
        print(f"Time: {iter_time:.2f}s", end=" ")

        if max_chg < tol and iteration > 0:
            print(f"Converged: iter {iteration+1}, MaxRelErr {max_chg:.2e} < Tol {tol:.1e}")
            converged = True
            iteration_num = iteration + 1
            break
        if iteration == max_iter - 1:
            print(f"Max iters reached ({max_iter}). MaxRelErr {max_chg:.2e}")
            iteration_num = iteration + 1

    W1f, W2f, s2f, ell_pf = W1, W2, sigma2, ell_param

    # Final, one-shot imputation
    if Y_obs is not None:
        Y_filled = impute_full_Y_after_em(
            Y_obs=Y_obs,
            W1=W1f, W2=W2f,
            EZ1i_list=EZ1i_final_list,
            EZ2ij_list=EZ2ij_final_list,
            survey_times=survey_times,
            participant_survey_indices=participant_survey_indices,
            participant_original_indices=participant_original_indices,
            kernel_method=kernel_method,
            ell_param=ell_pf
        )
    else:
        if records_meta is None:
            raise ValueError("records_meta must be provided when Y_obs is None.")
        Y_filled = impute_rows_after_em_to_dataframe(
            records_meta=records_meta,
            W1=W1f,
            W2=W2f,
            EZ1i_list=EZ1i_final_list,
            EZ2ij_list=EZ2ij_final_list,
            survey_times=survey_times,
            participant_survey_indices=participant_survey_indices,
            participant_original_indices=participant_original_indices,
            kernel_method=kernel_method,
            ell_param=ell_pf,
        )

    return W1f, W2f, s2f, ell_pf, EZ1i_final_list, EZ2ij_final_list, Y_filled, iteration_num, converged


def build_lists_from_Y_obs(Y_obs: np.ndarray):
    """
    Convert a full (n_orig, J, p) array with NaNs into:
      - Y_list: list of (p, J_i) matrices (one per participant with any data),
      - participant_survey_indices: list of 1-D int arrays (original time indices),
      - participant_original_indices: list of original participant indices.
    """
    n_orig, J, p = Y_obs.shape
    Y_list, participant_survey_indices, participant_original_indices = [], [], []
    for i in range(n_orig):
        participant_data_i, original_indices_i = [], []
        for j in range(J):
            if np.any(~np.isnan(Y_obs[i, j, :])):
                participant_data_i.append(Y_obs[i, j, :])
                original_indices_i.append(j)
        if participant_data_i:
            Y_list.append(np.vstack(participant_data_i).T)                # (p, J_i)
            participant_survey_indices.append(np.array(original_indices_i))
            participant_original_indices.append(i)
    n_eff = len(Y_list)
    if n_eff < n_orig:
        print(f"Warning: {n_orig - n_eff} participants removed due to no observations.")
    return Y_list, participant_survey_indices, participant_original_indices


def _default_survey_times(J: int) -> np.ndarray:
    if J <= 0:
        raise ValueError("Number of survey waves J must be positive.")
    return np.linspace(10.0, 10.0 * J, J, dtype=float)


def _validate_or_default_survey_times(
    survey_times: Optional[np.ndarray],
    J: int,
) -> np.ndarray:
    if survey_times is None:
        return _default_survey_times(J)
    survey_times_arr = np.asarray(survey_times, dtype=float).reshape(-1)
    if survey_times_arr.size != J:
        raise ValueError(
            f"survey_times length ({survey_times_arr.size}) must equal number of waves J ({J})."
        )
    if not np.all(np.isfinite(survey_times_arr)):
        raise ValueError("survey_times must contain only finite values.")
    return survey_times_arr


def _resolve_df_column(
    df: "pd.DataFrame",
    col_spec: str | int | None,
    arg_name: str,
    required: bool,
) -> Any:
    if col_spec is None:
        if required:
            raise ValueError(f"{arg_name} is required for DataFrame input.")
        return None
    if isinstance(col_spec, str):
        if col_spec in df.columns:
            return col_spec
        if required:
            raise ValueError(f"Column '{col_spec}' from {arg_name} was not found in DataFrame.")
        return None
    if isinstance(col_spec, int):
        n_cols = len(df.columns)
        idx = col_spec + n_cols if col_spec < 0 else col_spec
        if idx < 0 or idx >= n_cols:
            raise ValueError(f"{arg_name}={col_spec} is out of bounds for DataFrame with {n_cols} columns.")
        return df.columns[idx]
    raise TypeError(f"{arg_name} must be a column name (str), column index (int), or None.")


def _tabular_numpy_to_dataframe(
    Y_tabular: np.ndarray,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
) -> tuple["pd.DataFrame", str, str | None, list[str]]:
    if pd is None:
        raise ImportError("pandas is required for tabular input. Please install pandas.")

    arr = np.asarray(Y_tabular)
    if arr.ndim != 2:
        raise ValueError("Tabular numpy input must be 2D with shape (n_rows, n_cols).")
    n_cols = arr.shape[1]

    def _normalize_col_idx(col_spec: int, name: str) -> int:
        idx = col_spec + n_cols if col_spec < 0 else col_spec
        if idx < 0 or idx >= n_cols:
            raise ValueError(f"{name}={col_spec} is out of bounds for array with {n_cols} columns.")
        return idx

    subject_idx = _normalize_col_idx(subject_id_col if isinstance(subject_id_col, int) else 0, "subject_id_col")

    if visit_time_col is None:
        visit_idx = None
    else:
        visit_idx = _normalize_col_idx(visit_time_col if isinstance(visit_time_col, int) else 1, "visit_time_col")

    if feature_cols is None:
        blocked = {subject_idx}
        if visit_idx is not None:
            blocked.add(visit_idx)
        feature_idx = [k for k in range(n_cols) if k not in blocked]
    else:
        feature_idx = []
        for c in feature_cols:
            if not isinstance(c, int):
                raise TypeError("For numpy tabular input, feature_cols must be a list of integer column indices.")
            feature_idx.append(_normalize_col_idx(c, "feature_cols"))

    if len(feature_idx) == 0:
        raise ValueError("No feature columns were identified from tabular numpy input.")

    if subject_idx in feature_idx or (visit_idx is not None and visit_idx in feature_idx):
        raise ValueError("subject_id_col/visit_time_col must not overlap with feature_cols.")

    feature_names = [f"feature_{k}" for k in range(len(feature_idx))]
    data = {"subject_id": arr[:, subject_idx]}
    if visit_idx is not None:
        data["visit_time"] = arr[:, visit_idx]
    for k, col_idx in enumerate(feature_idx):
        data[feature_names[k]] = arr[:, col_idx]

    return pd.DataFrame(data), "subject_id", ("visit_time" if visit_idx is not None else None), feature_names


def _time_values_to_indices(time_values: np.ndarray, survey_times: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    idx = np.full(time_values.shape[0], -1, dtype=np.int32)
    for j, t_val in enumerate(time_values):
        matches = np.where(np.isclose(survey_times, t_val, rtol=0.0, atol=tol))[0]
        if matches.size == 0:
            raise ValueError(
                f"visit_time value {t_val} was not found in provided survey_times (tol={tol})."
            )
        idx[j] = int(matches[0])
    return idx


def _records_to_tensor(
    Y_records: Any,
    survey_times: np.ndarray | None = None,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if pd is None:
        raise ImportError("pandas is required for tabular input. Please install pandas.")

    if isinstance(Y_records, np.ndarray):
        df, sid_col, time_col, feat_cols = _tabular_numpy_to_dataframe(
            Y_records,
            subject_id_col=subject_id_col,
            visit_time_col=visit_time_col,
            feature_cols=feature_cols,
        )
    elif isinstance(Y_records, pd.DataFrame):
        df = Y_records.copy()
        sid_col = _resolve_df_column(df, subject_id_col, "subject_id_col", required=True)
        time_col = _resolve_df_column(df, visit_time_col, "visit_time_col", required=False)
        if feature_cols is None:
            excluded = {sid_col}
            if time_col is not None:
                excluded.add(time_col)
            feat_cols = [c for c in df.columns if c not in excluded]
        else:
            feat_cols = [_resolve_df_column(df, c, "feature_cols", required=True) for c in feature_cols]
    else:
        raise TypeError(
            "Tabular input must be a pandas DataFrame or a 2D numpy array."
        )

    if len(df) == 0:
        raise ValueError("Input tabular data is empty.")
    if sid_col is None:
        raise ValueError("subject_id column is required for tabular input.")
    if len(feat_cols) == 0:
        raise ValueError("No feature columns were found for tabular input.")
    if sid_col in feat_cols or (time_col is not None and time_col in feat_cols):
        raise ValueError("subject_id and visit_time columns must be distinct from feature columns.")

    if df[sid_col].isna().any():
        raise ValueError("subject_id contains missing values, which are not allowed.")

    subject_ids = pd.unique(df[sid_col])
    subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    subject_idx = df[sid_col].map(subject_id_to_idx).to_numpy(dtype=np.int32)

    has_visit_time = time_col is not None and df[time_col].notna().any()
    if has_visit_time:
        visit_time_values = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(visit_time_values)):
            raise ValueError("visit_time column contains missing or non-numeric values.")

        if survey_times is None:
            survey_times_out = np.sort(np.unique(visit_time_values))
            visit_idx = np.searchsorted(survey_times_out, visit_time_values).astype(np.int32)
        else:
            survey_times_out = np.asarray(survey_times, dtype=float).reshape(-1)
            if survey_times_out.size == 0 or not np.all(np.isfinite(survey_times_out)):
                raise ValueError("survey_times must contain at least one finite value.")
            visit_idx = _time_values_to_indices(visit_time_values, survey_times_out)
    else:
        # No visit_time provided: use per-subject visit order and a shared default timeline.
        visit_idx = df.groupby(sid_col, sort=False).cumcount().to_numpy(dtype=np.int32)
        J = int(np.max(visit_idx)) + 1
        survey_times_out = _validate_or_default_survey_times(survey_times, J)

    dup_mask = pd.DataFrame({"subject_idx": subject_idx, "visit_idx": visit_idx}).duplicated(keep=False)
    if bool(dup_mask.any()):
        n_dup = int(dup_mask.sum())
        raise ValueError(
            f"Found {n_dup} duplicated subject/visit rows. "
            "Each (subject_id, visit_time) pair must appear at most once."
        )

    X = df.loc[:, feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)

    n_subjects = len(subject_ids)
    J = survey_times_out.shape[0]
    p = X.shape[1]
    Y_tensor = np.full((n_subjects, J, p), np.nan, dtype=float)
    Y_tensor[subject_idx, visit_idx, :] = X
    return Y_tensor, survey_times_out


def _records_to_lists(
    Y_records: Any,
    survey_times: np.ndarray | None = None,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int], np.ndarray, Dict[str, Any]]:
    """
    Build participant lists directly from long records without materializing a dense (n,J,p) tensor.
    """
    if pd is None:
        raise ImportError("pandas is required for tabular input. Please install pandas.")

    if isinstance(Y_records, np.ndarray):
        df, sid_col, time_col, feat_cols = _tabular_numpy_to_dataframe(
            Y_records,
            subject_id_col=subject_id_col,
            visit_time_col=visit_time_col,
            feature_cols=feature_cols,
        )
    elif isinstance(Y_records, pd.DataFrame):
        df = Y_records.copy()
        sid_col = _resolve_df_column(df, subject_id_col, "subject_id_col", required=True)
        time_col = _resolve_df_column(df, visit_time_col, "visit_time_col", required=False)
        if feature_cols is None:
            excluded = {sid_col}
            if time_col is not None:
                excluded.add(time_col)
            feat_cols = [c for c in df.columns if c not in excluded]
        else:
            feat_cols = [_resolve_df_column(df, c, "feature_cols", required=True) for c in feature_cols]
    else:
        raise TypeError("Tabular input must be a pandas DataFrame or a 2D numpy array.")

    if len(df) == 0:
        raise ValueError("Input tabular data is empty.")
    if sid_col is None:
        raise ValueError("subject_id column is required for tabular input.")
    if len(feat_cols) == 0:
        raise ValueError("No feature columns were found for tabular input.")
    if sid_col in feat_cols or (time_col is not None and time_col in feat_cols):
        raise ValueError("subject_id and visit_time columns must be distinct from feature columns.")
    if df[sid_col].isna().any():
        raise ValueError("subject_id contains missing values, which are not allowed.")

    subject_ids = pd.unique(df[sid_col])
    subject_id_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    subject_idx = df[sid_col].map(subject_id_to_idx).to_numpy(dtype=np.int32)

    has_visit_time = time_col is not None and df[time_col].notna().any()
    if has_visit_time:
        visit_time_values = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(visit_time_values)):
            raise ValueError("visit_time column contains missing or non-numeric values.")

        if survey_times is None:
            survey_times_out = np.sort(np.unique(visit_time_values))
            visit_idx = np.searchsorted(survey_times_out, visit_time_values).astype(np.int32)
        else:
            survey_times_out = np.asarray(survey_times, dtype=float).reshape(-1)
            if survey_times_out.size == 0 or not np.all(np.isfinite(survey_times_out)):
                raise ValueError("survey_times must contain at least one finite value.")
            visit_idx = _time_values_to_indices(visit_time_values, survey_times_out)
    else:
        visit_idx = df.groupby(sid_col, sort=False).cumcount().to_numpy(dtype=np.int32)
        J = int(np.max(visit_idx)) + 1
        survey_times_out = _validate_or_default_survey_times(survey_times, J)

    dup_mask = pd.DataFrame({"subject_idx": subject_idx, "visit_idx": visit_idx}).duplicated(keep=False)
    if bool(dup_mask.any()):
        n_dup = int(dup_mask.sum())
        raise ValueError(
            f"Found {n_dup} duplicated subject/visit rows. "
            "Each (subject_id, visit_time) pair must appear at most once."
        )

    X_rows = df.loc[:, feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)

    Y_list: list[np.ndarray] = []
    participant_survey_indices: list[np.ndarray] = []
    participant_original_indices: list[int] = []

    n_subjects = len(subject_ids)
    for subj_i in range(n_subjects):
        row_mask = subject_idx == subj_i
        if not np.any(row_mask):
            continue

        subj_visits = visit_idx[row_mask]
        subj_values = X_rows[row_mask, :]
        order = np.argsort(subj_visits, kind="mergesort")
        subj_visits = subj_visits[order]
        subj_values = subj_values[order, :]

        # Keep only visits with at least one observed feature for EM updates.
        keep = np.any(~np.isnan(subj_values), axis=1)
        if not np.any(keep):
            continue

        Y_list.append(subj_values[keep, :].T.copy())
        participant_survey_indices.append(subj_visits[keep].astype(np.int32, copy=True))
        participant_original_indices.append(subj_i)

    if len(Y_list) < n_subjects:
        print(f"Warning: {n_subjects - len(Y_list)} participants removed due to no observations.")

    records_meta: Dict[str, Any] = {
        "subject_ids": np.asarray(subject_ids),
        "feature_names": [str(c) for c in feat_cols],
        "row_subject_idx": subject_idx.astype(np.int32, copy=True),
        "row_visit_idx": visit_idx.astype(np.int32, copy=True),
        "row_values": X_rows.copy(),
    }
    return Y_list, participant_survey_indices, participant_original_indices, survey_times_out, records_meta


def prepare_hppca_input(
    Y_obs: Any,
    survey_times: np.ndarray | None = None,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalize user input into a dense tensor `Y_obs` and a global `survey_times` vector.

    Supported inputs
    ----------------
    1) 3D numpy array (n, J, p): existing behavior.
    2) pandas DataFrame: one row per visit with columns
       [subject_id, visit_time(optional), feature_1, ..., feature_p].
    3) 2D numpy array in the same long format as (2), using column indices.

    Notes
    -----
    - DataFrame input is preferred because named columns avoid index ambiguity.
    - If tabular input has no visit_time column (or it's entirely missing), visits are
      ordered within each subject and mapped to a shared timeline.
    """
    if isinstance(Y_obs, np.ndarray) and Y_obs.ndim == 3:
        Y_arr = np.asarray(Y_obs, dtype=float)
        survey_times_out = _validate_or_default_survey_times(survey_times, Y_arr.shape[1])
        return Y_arr, survey_times_out

    if (isinstance(Y_obs, np.ndarray) and Y_obs.ndim == 2) or (
        pd is not None and isinstance(Y_obs, pd.DataFrame)
    ):
        return _records_to_tensor(
            Y_obs,
            survey_times=survey_times,
            subject_id_col=subject_id_col,
            visit_time_col=visit_time_col,
            feature_cols=feature_cols,
        )

    raise TypeError(
        "Y_obs must be a 3D numpy array, a pandas DataFrame, or a 2D tabular numpy array."
    )


def filled_tensor_to_dataframe(Y_filled: np.ndarray, survey_times: np.ndarray) -> "pd.DataFrame":
    """
    Convert a filled tensor (n, J, p) into long-form DataFrame columns:
      subject_id, visit_time, feature_0..feature_{p-1}.
    """
    if pd is None:
        raise ImportError("pandas is required to return filled data as a DataFrame.")
    if Y_filled.ndim != 3:
        raise ValueError("Y_filled must be a 3D tensor with shape (n, J, p).")

    n, J, p = Y_filled.shape
    survey_times_arr = np.asarray(survey_times, dtype=float).reshape(-1)
    if survey_times_arr.size != J:
        raise ValueError(
            f"survey_times length ({survey_times_arr.size}) must equal Y_filled second dimension J ({J})."
        )

    y2 = Y_filled.reshape(n * J, p)
    out = {
        "subject_id": np.repeat(np.arange(n, dtype=np.int64), J),
        "visit_time": np.tile(survey_times_arr, n),
    }
    for k in range(p):
        out[f"feature_{k}"] = y2[:, k]
    return pd.DataFrame(out)


def build_latent_long_dataframe(
    EZ1i_list: list,
    EZ2ij_list: list,
    participant_survey_indices: list,
    participant_original_indices: list,
    survey_times: np.ndarray,
    subject_ids: np.ndarray | None = None,
    subject_col_name: str = "subject_id",
    visit_col_name: str = "visit_time",
) -> "pd.DataFrame":
    """
    Build long latent-factor table with columns:
      subject_id, visit_time, z1_dim_1..z1_dim_d1, z2_dim_1..z2_dim_d2.
    One row corresponds to one observed visit used in EM for each effective participant.
    """
    if pd is None:
        raise ImportError("pandas is required to return latent factors as a DataFrame.")

    n_eff = len(participant_original_indices)
    if not (len(EZ1i_list) == len(EZ2ij_list) == len(participant_survey_indices) == n_eff):
        raise ValueError(
            "Inconsistent latent list lengths for EZ1i_list, EZ2ij_list, "
            "participant_survey_indices, participant_original_indices."
        )

    survey_times_arr = np.asarray(survey_times, dtype=float).reshape(-1)

    d1 = int(np.asarray(EZ1i_list[0]).size) if n_eff > 0 else 0
    d2 = 0
    for ez2 in EZ2ij_list:
        if ez2 is None:
            continue
        arr = np.asarray(ez2)
        if arr.ndim == 2:
            d2 = int(arr.shape[0])
            break

    blocks = []
    for eff_i, subj_idx in enumerate(participant_original_indices):
        visit_idx = np.asarray(participant_survey_indices[eff_i], dtype=np.int32).reshape(-1)
        if visit_idx.size == 0:
            continue

        if np.any(visit_idx < 0) or np.any(visit_idx >= survey_times_arr.size):
            raise ValueError("participant_survey_indices contains invalid visit index.")

        subj_value = subject_ids[int(subj_idx)] if subject_ids is not None else int(subj_idx)
        ez1_i = np.asarray(EZ1i_list[eff_i], dtype=float).reshape(-1)
        if d1 > 0 and ez1_i.size != d1:
            raise ValueError(f"EZ1i_list[{eff_i}] has size {ez1_i.size}, expected {d1}.")

        if d2 > 0:
            ez2_i_raw = EZ2ij_list[eff_i]
            if ez2_i_raw is None:
                ez2_i = np.zeros((d2, visit_idx.size), dtype=float)
            else:
                ez2_i = np.asarray(ez2_i_raw, dtype=float)
                if ez2_i.ndim != 2:
                    raise ValueError(f"EZ2ij_list[{eff_i}] must be 2D; got ndim={ez2_i.ndim}.")
                if ez2_i.shape[1] != visit_idx.size and ez2_i.shape[0] == visit_idx.size:
                    ez2_i = ez2_i.T
                if ez2_i.shape[0] != d2 or ez2_i.shape[1] != visit_idx.size:
                    raise ValueError(
                        f"EZ2ij_list[{eff_i}] shape {ez2_i.shape} incompatible with "
                        f"(d2={d2}, Ji={visit_idx.size})."
                    )
        else:
            ez2_i = np.zeros((0, visit_idx.size), dtype=float)

        block = {
            subject_col_name: np.repeat(subj_value, visit_idx.size),
            visit_col_name: survey_times_arr[visit_idx],
        }
        for j in range(d1):
            block[f"z1_dim_{j+1}"] = np.repeat(ez1_i[j], visit_idx.size)
        for k in range(d2):
            block[f"z2_dim_{k+1}"] = ez2_i[k, :]
        blocks.append(pd.DataFrame(block))

    if not blocks:
        cols = [subject_col_name, visit_col_name]
        cols += [f"z1_dim_{j+1}" for j in range(d1)]
        cols += [f"z2_dim_{k+1}" for k in range(d2)]
        return pd.DataFrame(columns=cols)
    return pd.concat(blocks, ignore_index=True)


def impute_rows_after_em_to_dataframe(
    records_meta: Dict[str, Any],
    W1: np.ndarray,
    W2: np.ndarray,
    EZ1i_list: list,
    EZ2ij_list: list,
    survey_times: np.ndarray,
    participant_survey_indices: list,
    participant_original_indices: list,
    kernel_method: str,
    ell_param,
) -> "pd.DataFrame":
    """
    Fill missing feature values row-wise for the original tabular records only.
    This avoids constructing a dense (n, J, p) tensor for irregular timelines.
    """
    if pd is None:
        raise ImportError("pandas is required to return filled data as a DataFrame.")

    row_subject_idx = np.asarray(records_meta["row_subject_idx"], dtype=np.int32)
    row_visit_idx = np.asarray(records_meta["row_visit_idx"], dtype=np.int32)
    row_values = np.asarray(records_meta["row_values"], dtype=float).copy()
    subject_ids = np.asarray(records_meta["subject_ids"])
    feature_names = list(records_meta["feature_names"])

    survey_times_arr = np.asarray(survey_times, dtype=float).reshape(-1)
    if row_values.ndim != 2:
        raise ValueError("records_meta['row_values'] must be 2D.")
    p = row_values.shape[1]
    d1 = W1.shape[1]
    d2 = W2.shape[1]

    use_time_gp = ("iid" not in kernel_method) and (d2 > 0)
    kernel_fun = matern52_kernel if ("matern52" in kernel_method) else rbf_kernel
    orig_to_eff = {orig_i: eff_i for eff_i, orig_i in enumerate(participant_original_indices)}

    for subj_idx in np.unique(row_subject_idx):
        row_ids = np.where(row_subject_idx == subj_idx)[0]
        if row_ids.size == 0:
            continue

        if subj_idx in orig_to_eff:
            eff_i = orig_to_eff[subj_idx]
            EZ1_i = EZ1i_list[eff_i] if (d1 > 0 and EZ1i_list[eff_i] is not None) else (np.zeros(d1) if d1 > 0 else None)
            EZ2_obs = EZ2ij_list[eff_i] if (d2 > 0 and EZ2ij_list[eff_i] is not None) else (np.zeros((d2, 0)) if d2 > 0 else None)
            idx_obs = np.asarray(participant_survey_indices[eff_i], dtype=np.int32)
        else:
            EZ1_i = np.zeros(d1) if d1 > 0 else None
            EZ2_obs = np.zeros((d2, 0)) if d2 > 0 else None
            idx_obs = np.array([], dtype=np.int32)

        req_idx = row_visit_idx[row_ids]
        unique_req, inv_req = np.unique(req_idx, return_inverse=True)

        if d2 == 0:
            EZ2_req = None
        else:
            EZ2_req = np.zeros((d2, unique_req.size))
            obs_lookup = {int(idx_obs[k]): k for k in range(idx_obs.size)}
            for col_u, idx_u in enumerate(unique_req):
                k_obs = obs_lookup.get(int(idx_u))
                if k_obs is not None and EZ2_obs is not None and k_obs < EZ2_obs.shape[1]:
                    EZ2_req[:, col_u] = EZ2_obs[:, k_obs]

            req_miss_mask = np.array([int(u) not in obs_lookup for u in unique_req], dtype=bool)
            req_miss_idx = unique_req[req_miss_mask]
            if use_time_gp and req_miss_idx.size > 0 and idx_obs.size > 0:
                t_obs = survey_times_arr[idx_obs]
                t_req = survey_times_arr[req_miss_idx]

                if "single_ell" in kernel_method:
                    ell = float(ell_param)
                    K_oo, _ = kernel_fun(t_obs, ell)
                    try:
                        K_oo_reg = K_oo.copy()
                        K_oo_reg[np.diag_indices_from(K_oo_reg)] += 1e-9
                        K_oo_inv_EZ2 = np.linalg.solve(K_oo_reg, EZ2_obs.T).T
                    except np.linalg.LinAlgError:
                        K_oo_inv = np.linalg.pinv(K_oo)
                        K_oo_inv_EZ2 = (K_oo_inv @ EZ2_obs.T).T

                    if "rbf" in kernel_method:
                        K_ro = np.exp(-0.5 * ((t_req.reshape(-1, 1) - t_obs.reshape(1, -1)) ** 2) / (ell**2))
                    else:
                        from scipy.spatial.distance import cdist
                        r = cdist(t_req.reshape(-1, 1), t_obs.reshape(-1, 1), metric="euclidean")
                        term1_val = SQRT5 * r / ell
                        term2_val = 5.0 * r**2 / (3.0 * ell**2)
                        K_ro = (1.0 + term1_val + term2_val) * np.exp(-term1_val)
                    EZ2_req[:, req_miss_mask] = (K_ro @ K_oo_inv_EZ2.T).T

                elif "multi_ell" in kernel_method:
                    ell_vec = np.asarray(ell_param)
                    for rdim in range(d2):
                        ell_r = float(ell_vec[rdim])
                        K_oo, _ = kernel_fun(t_obs, ell_r)
                        try:
                            K_oo_reg = K_oo.copy()
                            K_oo_reg[np.diag_indices_from(K_oo_reg)] += 1e-9
                            K_oo_inv_m = np.linalg.solve(K_oo_reg, np.eye(len(t_obs)))
                        except np.linalg.LinAlgError:
                            K_oo_inv_m = np.linalg.pinv(K_oo)

                        if "rbf" in kernel_method:
                            K_ro = np.exp(-0.5 * ((t_req.reshape(-1, 1) - t_obs.reshape(1, -1)) ** 2) / (ell_r**2))
                        else:
                            from scipy.spatial.distance import cdist
                            r = cdist(t_req.reshape(-1, 1), t_obs.reshape(-1, 1), metric="euclidean")
                            term1_val = SQRT5 * r / ell_r
                            term2_val = 5.0 * r**2 / (3.0 * ell_r**2)
                            K_ro = (1.0 + term1_val + term2_val) * np.exp(-term1_val)
                        EZ2_req[rdim, req_miss_mask] = K_ro @ (K_oo_inv_m @ EZ2_obs[rdim, :])

        for local_k, row_id in enumerate(row_ids):
            y_vec = row_values[row_id, :]
            miss_mask = np.isnan(y_vec)
            if not np.any(miss_mask):
                continue
            pred = np.zeros(p)
            if d1 > 0 and EZ1_i is not None:
                pred += W1 @ EZ1_i
            if d2 > 0 and EZ2_req is not None:
                pred += W2 @ EZ2_req[:, inv_req[local_k]]
            y_vec[miss_mask] = pred[miss_mask]
            row_values[row_id, :] = y_vec

    out = {
        "subject_id": subject_ids[row_subject_idx],
        "visit_time": survey_times_arr[row_visit_idx],
    }
    for k, feat_name in enumerate(feature_names):
        out[feat_name] = row_values[:, k]
    return pd.DataFrame(out)

# ---------------------------
# 0) Rough Imputation (then centering)
# ---------------------------

def impute_and_center_Y(
    Y_obs: np.ndarray,
    strategy: str = "feature_mean",
) -> np.ndarray:
    """
    Impute missing entries in Y_obs and center the result over (participants, surveys).
    Y_obs: (n, J, p) with NaNs for missing.
    strategy:
      - "feature_mean": legacy name; fills each missing cell by that participant's
        across-visit mean for the same feature, with a cohort-wide feature mean fallback
        if the participant has no observed value for that feature.
    Returns
    -------
    Y_imp_centered: (n, J, p)
    """
    if Y_obs.ndim != 3:
        raise ValueError("Y_obs must be a 3D array (n, J, p).")

    Y = np.array(Y_obs, dtype=float, copy=True)
    n, J, p = Y.shape
    missing = ~np.isfinite(Y)

    if strategy == "feature_mean":
        with np.errstate(invalid="ignore"):
            mu_p = np.nanmean(Y, axis=(0, 1))  # (p,)
            mu_ip = np.nanmean(Y, axis=1)      # (n, p)
        # fall back to cohort feature means when a participant/feature is entirely missing
        mu_p = np.where(np.isfinite(mu_p), mu_p, 0.0)
        mu_ip = np.where(np.isfinite(mu_ip), mu_ip, mu_p[None, :])
        Y = np.where(missing, mu_ip[:, None, :], Y)
    else:
        raise ValueError(f"Unknown imputation strategy: {strategy}")

    # Center across all participants and surveys (per paper’s convention)
    Y -= Y.mean(axis=(0, 1), keepdims=True)
    return Y


# ---------------------------
# 1) Rough Σ̂ estimation and projection to Assumption 1
# ---------------------------

def estimate_time_cov_from_residuals(Y_imp: np.ndarray) -> np.ndarray:
    """
    Estimate a rough time covariance Σ̂ (J x J) from per-subject residuals after
    removing each subject's time-average (reduces between-subject W1 effect).
    Aggregates across features to stabilize.
    """
    n, J, p = Y_imp.shape
    subj_mean = Y_imp.mean(axis=1, keepdims=True)        # (n, 1, p)
    R = Y_imp - subj_mean                                 # (n, J, p)
    X = np.transpose(R, (0, 2, 1)).reshape(n * p, J)      # (n*p, J): rows are subject-feature pairs
    Sig = (X.T @ X) / max(n * p, 1)
    return (Sig + Sig.T) / 2.0


def project_to_assumption1(
    Sigma_hat: np.ndarray,
    eps: float = 1e-8,
    max_iter: int = 10,
    tol: float = 1e-7,
) -> np.ndarray:
    """
    Project Σ̂ onto correlation matrices that satisfy Assumption 1 (1 is an eigenvector).
    Ensures:
      * symmetry
      * PSD with eigenvalues >= eps
      * unit diagonal (correlation matrix)
      * Σ 1 ∥ 1 (Assumption 1)
    """
    if Sigma_hat.ndim != 2 or Sigma_hat.shape[0] != Sigma_hat.shape[1]:
        raise ValueError("Sigma_hat must be a square matrix.")

    J = Sigma_hat.shape[0]
    u1 = np.ones(J) / np.sqrt(J)
    P1 = np.outer(u1, u1)
    Pperp = np.eye(J) - P1

    def _symmetrize(M: np.ndarray) -> np.ndarray:
        return (M + M.T) / 2.0

    def _psd_clip(M: np.ndarray) -> np.ndarray:
        w, U = np.linalg.eigh(_symmetrize(M))
        w = np.maximum(w, eps)
        return U @ np.diag(w) @ U.T

    def _project_assumption(M: np.ndarray) -> np.ndarray:
        lam1 = float(u1 @ M @ u1)
        M_proj = lam1 * P1 + Pperp @ M @ Pperp
        return _symmetrize(M_proj)

    Sigma = _psd_clip(Sigma_hat)
    for _ in range(max_iter):
        Sigma_prev = Sigma

        d = np.sqrt(np.clip(np.diag(Sigma), eps, None))
        Sigma = Sigma / np.outer(d, d)
        np.fill_diagonal(Sigma, 1.0)

        Sigma = _project_assumption(Sigma)
        Sigma = _psd_clip(Sigma)

        diff = np.linalg.norm(Sigma - Sigma_prev, ord="fro")
        baseline = max(np.linalg.norm(Sigma_prev, ord="fro"), 1.0)
        if diff / baseline < tol and np.allclose(np.diag(Sigma), 1.0, atol=1e-6):
            break

    # Final tidy: enforce symmetry, PSD, and unit diagonal
    Sigma = _psd_clip(Sigma)
    d = np.sqrt(np.clip(np.diag(Sigma), eps, None))
    Sigma = Sigma / np.outer(d, d)
    np.fill_diagonal(Sigma, 1.0)
    Sigma = _project_assumption(Sigma)
    Sigma = _psd_clip(Sigma)
    np.fill_diagonal(Sigma, 1.0)
    return _symmetrize(Sigma)


# ---------------------------
# 2) RBF length-scale initializers
# ---------------------------

def _pairwise_time_distances(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times).ravel()
    return np.abs(t[:, None] - t[None, :])


def fit_rbf_lengthscale_from_cov(
    Sigma: np.ndarray,
    times: np.ndarray,
    grid_size: int = 120
) -> float:
    """
    Choose ℓ to make K_RBF(t,t';ℓ) ≈ Corr(Sigma) in least-squares (off-diagonals).
    """
    J = Sigma.shape[0]
    D = _pairwise_time_distances(times)
    d = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    Corr = Sigma / (d[:, None] * d[None, :])
    np.fill_diagonal(Corr, 1.0)

    iu = np.triu_indices(J, k=1)
    r = D[iu]
    s = Corr[iu]
    s = np.clip(s, 1e-6, 0.999)  # stabilize

    r_pos = r[r > 0]
    if r_pos.size == 0:
        return 1.0
    rmin = float(np.min(r_pos))
    rmax = float(np.max(r_pos))

    ls_grid = np.logspace(np.log10(rmin / 10.0), np.log10(max(rmax * 10.0, rmin*1.1)), grid_size)
    best, best_val = None, np.inf
    for ell in ls_grid:
        pred = np.exp(-(r ** 2) / (2.0 * ell ** 2))
        val = np.mean((pred - s) ** 2)
        if val < best_val:
            best_val, best = val, float(ell)

    # Fallback by matching single correlation at median distance
    if best is None or not np.isfinite(best):
        s_bar = float(np.median(s))
        s_bar = np.clip(s_bar, 1e-6, 0.999)
        r_med = float(np.median(r_pos))
        best = r_med / np.sqrt(-2.0 * np.log(s_bar))
    return float(best)


def tau2_to_rbf_lengthscale(tau2: float, times: np.ndarray) -> float:
    """
    Convert CS parameter τ^2 (off-diagonal correlation) to an RBF ℓ by matching
    exp(-r_med^2/(2ℓ^2)) = τ^2 at the median nonzero separation r_med.
    """
    D = _pairwise_time_distances(times)
    r = D[np.triu_indices(D.shape[0], k=1)]
    r_pos = r[r > 0]
    if r_pos.size == 0:
        return 1.0
    r_med = float(np.median(r_pos))
    t = float(np.clip(tau2, 1e-6, 1 - 1e-6))
    return float(r_med / np.sqrt(-2.0 * np.log(t)))


# -------- Matérn 5/2 correlation helpers (unit variance) --------

def _pairwise_time_distances(times: np.ndarray) -> np.ndarray:
    t = np.asarray(times, dtype=float).ravel()
    return np.abs(t[:, None] - t[None, :])

def matern52_corr_matrix(times: np.ndarray, ell: float) -> np.ndarray:
    """Full JxJ Matérn-5/2 correlation matrix (diag=1)."""
    D = _pairwise_time_distances(times)
    r = D / max(ell, 1e-12)
    return (1.0 + np.sqrt(5.0)*r + 5.0*r*r/3.0) * np.exp(-np.sqrt(5.0)*r)

def _matern52_corr_at_r(r: np.ndarray | float, ell: float) -> np.ndarray | float:
    r = np.asarray(r, dtype=float)
    z = r / max(ell, 1e-12)
    return (1.0 + np.sqrt(5.0)*z + 5.0*z*z/3.0) * np.exp(-np.sqrt(5.0)*z)

def _solve_ell_for_target_corr_m52(r: float, target: float,
                                   ell_lo: float = 1e-6,
                                   ell_hi: float = 1e6,
                                   max_iter: int = 60, tol: float = 1e-10) -> float:
    """
    Solve for ell so that matern52_corr(r, ell) == target using bisection.
    For fixed r>0, correlation increases monotonically with ell in (0, ∞).
    """
    target = float(np.clip(target, 1e-8, 1 - 1e-8))
    lo, hi = max(ell_lo, 1e-12), max(ell_hi, 10.0*r + 1.0)
    f_lo = _matern52_corr_at_r(r, lo) - target
    f_hi = _matern52_corr_at_r(r, hi) - target
    # Ensure bracketing (expand hi if needed)
    k = 0
    while f_lo * f_hi > 0 and k < 20:
        hi *= 10.0
        f_hi = _matern52_corr_at_r(r, hi) - target
        k += 1
    # Bisection
    for _ in range(max_iter):
        mid = 0.5*(lo + hi)
        fm = _matern52_corr_at_r(r, mid) - target
        if abs(fm) < tol or (hi - lo) < 1e-12*max(1.0, mid):
            return float(mid)
        if f_lo * fm <= 0:
            hi, f_hi = mid, fm
        else:
            lo, f_lo = mid, fm
    return float(0.5*(lo + hi))

def fit_matern52_lengthscale_from_cov(Sigma: np.ndarray,
                                      times: np.ndarray,
                                      grid_size: int = 120) -> float:
    """
    Pick ell by least-squares fit of Matérn-5/2 correlations to Corr(Sigma) over off-diagonals.
    """
    J = Sigma.shape[0]
    D = _pairwise_time_distances(times)
    sd = np.sqrt(np.clip(np.diag(Sigma), 1e-12, None))
    Corr = Sigma / (sd[:, None] * sd[None, :])
    np.fill_diagonal(Corr, 1.0)

    iu = np.triu_indices(J, k=1)
    r_all = D[iu]
    s_all = np.clip(Corr[iu], 1e-6, 0.999)
    # Use only positive distances (ignore exact duplicates)
    mask = r_all > 0
    r, s = r_all[mask], s_all[mask]
    if r.size == 0:
        return 1.0

    rmin, rmax = float(np.min(r)), float(np.max(r))
    ls_grid = np.logspace(np.log10(rmin/10.0), np.log10(max(rmax*10.0, rmin*1.1)), grid_size)

    best, best_val = None, np.inf
    for ell in ls_grid:
        pred = _matern52_corr_at_r(r, ell)
        val = float(np.mean((pred - s)**2))
        if val < best_val:
            best_val, best = val, float(ell)

    # Fallback: match single correlation at median distance using bisection
    if best is None or not np.isfinite(best):
        r_med = float(np.median(r))
        s_med = float(np.median(s))
        best = _solve_ell_for_target_corr_m52(r_med, s_med)
    return float(best)

def tau2_to_matern52_lengthscale(tau2: float, times: np.ndarray) -> float:
    """
    Convert CS parameter tau2 to an Matern-5/2 length-scale by matching
    corr(r_med; ell) = tau2 at the median nonzero time gap.
    """
    D = _pairwise_time_distances(times)
    r = D[np.triu_indices(D.shape[0], k=1)]
    r_pos = r[r > 0]
    if r_pos.size == 0:
        return 1.0
    r_med = float(np.median(r_pos))
    return _solve_ell_for_target_corr_m52(r_med, float(np.clip(tau2, 1e-8, 1-1e-8)))


# -------- Your initializer switch: add Matérn-5/2 branch --------

def _ell_init_by_kernel(kernel_method: str, times: np.ndarray, Sigma: Optional[np.ndarray], d2: int,
                        tau2: Optional[float] = None) -> Optional[float | np.ndarray]:
    km = (kernel_method or "").lower()

    # --- RBF ---
    if "gp_rbf" in km or ("rbf" in km and "gp" in km):
        if tau2 is not None:
            ell = tau2_to_rbf_lengthscale(tau2, times)  # you already have this
        else:
            if Sigma is None:
                return None
            ell = fit_rbf_lengthscale_from_cov(Sigma, times)  # you already have this
        return np.repeat(ell, d2) if "multi_ell" in km else float(ell)

    # --- Matérn 5/2 ---
    if "matern52" in km or "m52" in km or "gp_matern52" in km:
        if tau2 is not None:
            ell = tau2_to_matern52_lengthscale(tau2, times)
        else:
            if Sigma is None:
                return None
            ell = fit_matern52_lengthscale_from_cov(Sigma, times)
        return np.repeat(ell, d2) if "multi_ell" in km else float(ell)

    # --- IID / others ---
    if "iid" in km:
        return None

    # Future kernels: add branches here
    return None


import numpy as np
from scipy.spatial.distance import cdist

def _kernel_family_from_method(kernel_method: str) -> str:
    km = (kernel_method or "").lower()
    if "iid" in km:
        return "iid"
    if "matern52" in km or "m52" in km:
        return "matern52"
    # default to RBF
    return "rbf"

def _is_multi_ell(kernel_method: str) -> bool:
    return "multi_ell" in (kernel_method or "").lower()


def _pairwise_diff_stats(Y):  # Y: (n, J, p) with NaNs
    n,J,p = Y.shape
    num = np.zeros((J,J))
    den = np.zeros((J,J))
    for i in range(n):
        M = np.isfinite(Y[i])          # (J,p)
        for t in range(J):
            for s in range(t+1, J):
                both = M[t] & M[s]
                m = int(both.sum())
                if m == 0: continue
                d = Y[i,t,both] - Y[i,s,both]
                val = float(np.dot(d, d))  # sum of squares over available features
                num[t,s] += val; num[s,t] += val
                den[t,s] += m;   den[s,t] += m
    D = np.divide(num, den, out=np.zeros_like(num), where=den>0)
    return D, den  # D_hat and weights


def _pairwise_diff_stats_from_lists(
    Y_list: list[np.ndarray],
    participant_survey_indices: list[np.ndarray],
    J: int,
):
    """
    Pairwise difference statistics from ragged participant lists without dense tensor materialization.
    """
    num = np.zeros((J, J))
    den = np.zeros((J, J))
    for Yi, idx_i in zip(Y_list, participant_survey_indices):
        if Yi.ndim != 2:
            continue
        Ji = Yi.shape[1]
        if Ji <= 1:
            continue
        idx_i = np.asarray(idx_i, dtype=int)
        for a in range(Ji):
            t = int(idx_i[a])
            ya = Yi[:, a]
            for b in range(a + 1, Ji):
                s = int(idx_i[b])
                yb = Yi[:, b]
                both = np.isfinite(ya) & np.isfinite(yb)
                m = int(np.sum(both))
                if m == 0:
                    continue
                d = ya[both] - yb[both]
                val = float(np.dot(d, d))
                num[t, s] += val
                num[s, t] += val
                den[t, s] += m
                den[s, t] += m
    D = np.divide(num, den, out=np.zeros_like(num), where=den > 0)
    return D, den

def _corr_rbf(times, ell):
    t = np.asarray(times).ravel()[:, None]
    D = cdist(t, t, "euclidean")
    K = np.exp(-0.5 * (D / max(ell, 1e-12))**2)
    np.fill_diagonal(K, 1.0)
    return K

def _corr_m52(times, ell):
    t = np.asarray(times).ravel()[:, None]
    r = cdist(t, t, "euclidean") / max(ell, 1e-12)
    K = (1.0 + np.sqrt(5.0)*r + 5.0*r*r/3.0) * np.exp(-np.sqrt(5.0)*r)
    np.fill_diagonal(K, 1.0)
    return K

def estimate_sigma_from_diffs_with_method(
    Y_obs: np.ndarray | None,
    times: np.ndarray,
    kernel_method: str,
    ells: np.ndarray | None = None,
    Y_list: list[np.ndarray] | None = None,
    participant_survey_indices: list[np.ndarray] | None = None,
    J_from_lists: int | None = None,
):
    """
    Uses CLI-style kernel_method to pick the family: 
      gp_rbf_*        -> RBF
      gp_matern52_*   -> Matérn 5/2
      gp_iid          -> IID (Σ = I, ℓ = None)
    Returns: (Sigma0, meta)
      meta includes: 'ell' (or None), 'b', 'sigma2', and diagnostics.
    """
    fam = _kernel_family_from_method(kernel_method)
    if Y_obs is not None:
        Dhat, W = _pairwise_diff_stats(Y_obs)
    else:
        if Y_list is None or participant_survey_indices is None or J_from_lists is None:
            raise ValueError(
                "When Y_obs is None, you must provide Y_list, participant_survey_indices, and J_from_lists."
            )
        Dhat, W = _pairwise_diff_stats_from_lists(Y_list, participant_survey_indices, int(J_from_lists))
    J = Dhat.shape[0]
    tri = np.triu_indices(J, 1)
    d = Dhat[tri]; w = W[tri]

    # IID shortcut: Σ = I; ℓ is undefined
    if fam == "iid":
        Sigma0 = np.eye(J)
        sigma2_hat = float(np.median(d[d > 0]) / 2.0) if d.size else 0.0
        return Sigma0, {"ell": None, "b": 0.0, "sigma2": sigma2_hat, "Dhat": Dhat, "weights": W}

    if ells is None or len(ells) == 0:
        uniq = np.unique(np.asarray(times, float))
        gaps = np.diff(np.sort(uniq))
        dmin = float(np.min(gaps[gaps > 0])) if gaps.size else 1.0
        dmax = float(np.max(uniq) - np.min(uniq)) if uniq.size > 1 else 1.0
        ells = np.geomspace(dmin / 20.0, max(dmax * 5.0, dmin * 1.5), 60)

    corr = _corr_rbf if fam == "rbf" else _corr_m52

    best = None
    for ell in ells:
        K = corr(times, ell)
        x1 = 2.0 * (1.0 - K[tri])  # 2b (1 - K)
        x2 = 2.0 * np.ones_like(x1) # 2σ²
        X = np.vstack([x1, x2]).T

        Wsqrt = np.sqrt(np.clip(w, 1e-12, None))
        Xw = X * Wsqrt[:, None]
        dw = d * Wsqrt
        try:
            beta, *_ = np.linalg.lstsq(Xw, dw, rcond=None)
        except np.linalg.LinAlgError:
            continue
        b_hat = float(max(beta[0], 0.0))
        s2_hat = float(max(beta[1], 0.0))
        resid = dw - Xw @ np.array([b_hat, s2_hat])
        sse = float(np.dot(resid, resid))
        if (best is None) or (sse < best[0]):
            best = (sse, float(ell), b_hat, s2_hat, K)

    sse, ell_hat, b_hat, sigma2_hat, K = best
    Sigma0 = K.copy()
    return Sigma0, {"ell": ell_hat, "b": b_hat, "sigma2": sigma2_hat, "Dhat": Dhat, "weights": W}

# ---------------------------
# 3) Initialization #1 — via Algorithm 1 (estimate Σ̂ then project to Assumption 1)
# ---------------------------

def initialize_via_algo1(
    Y_obs: Any,
    d1: int,
    d2: int,
    survey_times: Optional[np.ndarray] = None,
    kernel_method: str = "gp_rbf_single_ell",
    seed: Optional[int] = None,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
    Y_list_for_init: list[np.ndarray] | None = None,
    participant_survey_indices_for_init: list[np.ndarray] | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[float | np.ndarray], Dict[str, np.ndarray]]:
    """
    Steps:
      (a) Impute Y and center.
      (b) Estimate rough Σ̂ from time-demeaned residuals; project to Assumption 1 (Σ 1 ∥ 1).
      (c) Run Algorithm 1 with Σ = Σ̂_proj.
      (d) Initialize kernel parameter(s), e.g., RBF ℓ by fitting Corr(Σ̂_proj).

    Returns
    -------
    W1_init, W2_init, sigma2_init, ell_param_init, extras (dict with 'Sigma_init')
    """

    rng = np.random.default_rng(seed)
    is_tensor = isinstance(Y_obs, np.ndarray) and Y_obs.ndim == 3

    if is_tensor:
        if survey_times is None:
            survey_times = _default_survey_times(Y_obs.shape[1])
        Y_imp = impute_and_center_Y(Y_obs, strategy="feature_mean")

        Sigma0, pars = estimate_sigma_from_diffs_with_method(
            Y_obs=Y_imp,
            times=survey_times,
            kernel_method=kernel_method,
        )
        Sigma_proj = project_to_assumption1(Sigma0)

        # ---- Algo 1 fit (Part 2: Lemma 1; Theorem 1; Theorem 2; Proposition 1) ----
        fit1 = fit_hppca_alg1(Y_imp, Sigma_proj, d1=d1, d2=d2, rng=rng)
    else:
        if Y_list_for_init is None or participant_survey_indices_for_init is None:
            Y_list_for_init, participant_survey_indices_for_init, _, survey_times_out, _ = _records_to_lists(
                Y_records=Y_obs,
                survey_times=survey_times,
                subject_id_col=subject_id_col,
                visit_time_col=visit_time_col,
                feature_cols=feature_cols,
            )
            if survey_times is None:
                survey_times = survey_times_out
        if survey_times is None:
            raise ValueError("survey_times could not be resolved for tabular algo1 initialization.")

        Sigma0, pars = estimate_sigma_from_diffs_with_method(
            Y_obs=None,
            times=survey_times,
            kernel_method=kernel_method,
            Y_list=Y_list_for_init,
            participant_survey_indices=participant_survey_indices_for_init,
            J_from_lists=int(np.asarray(survey_times).shape[0]),
        )
        Sigma_proj = project_to_assumption1(Sigma0)

        fit1 = fit_hppca_alg1(
            Y_obs,
            Sigma_proj,
            d1=d1,
            d2=d2,
            survey_times=survey_times,
            subject_id_col=subject_id_col,
            visit_time_col=visit_time_col,
            feature_cols=feature_cols,
            rng=rng,
        )

    if "iid" in kernel_method:
        ell_init = None
    elif _is_multi_ell(kernel_method):
        ell_init = np.repeat(pars["ell"], d2)
    else:
        ell_init = float(pars["ell"])

    return fit1["W1"], fit1["W2"], float(fit1["sigma2"]), ell_init, {"Sigma_init": Sigma_proj}


# ---------------------------
# 4) Initialization #2 — via Algorithm 2 (assume compound symmetry)
# ---------------------------

def initialize_via_algo2_cs(
    Y_obs: Any,
    d1: int,
    d2: int,
    survey_times: Optional[np.ndarray] = None,
    kernel_method: str = "gp_rbf_single_ell",
    seed: Optional[int] = None,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
) -> Tuple[np.ndarray, np.ndarray, float, Optional[float | np.ndarray], Dict[str, float]]:
    """
    Steps:
      (a) Impute Y and center.
      (b) Run Algorithm 2 (compound symmetry Σ(τ^2)).
      (c) Convert τ^2 to an RBF ℓ by matching correlation at median time separation.

    Returns
    -------
    W1_init, W2_init, sigma2_init, ell_param_init, extras (dict with 'tau2_init')
    """

    rng = np.random.default_rng(seed)
    is_tensor = isinstance(Y_obs, np.ndarray) and Y_obs.ndim == 3

    if is_tensor:
        if survey_times is None:
            survey_times = _default_survey_times(Y_obs.shape[1])
        Y_imp = impute_and_center_Y(Y_obs, strategy="feature_mean")
        # ---- Algo 2 fit (Part 2: Corollary 1 & Algorithm 2) ----
        fit2 = fit_hppca_alg2_cs(Y_imp, d1=d1, d2=d2, rng=rng)
    else:
        if survey_times is None:
            _, _, _, survey_times_out, _ = _records_to_lists(
                Y_records=Y_obs,
                survey_times=None,
                subject_id_col=subject_id_col,
                visit_time_col=visit_time_col,
                feature_cols=feature_cols,
            )
            survey_times = survey_times_out
        fit2 = fit_hppca_alg2_cs(
            Y_obs,
            d1=d1,
            d2=d2,
            survey_times=survey_times,
            subject_id_col=subject_id_col,
            visit_time_col=visit_time_col,
            feature_cols=feature_cols,
            rng=rng,
        )

    # RBF ℓ (or None for IID); here we use τ^2 to pick ℓ
    ell_init = _ell_init_by_kernel(kernel_method, survey_times, Sigma=None, d2=d2, tau2=float(fit2["tau2"]))

    return fit2["W1"], fit2["W2"], float(fit2["sigma2"]), ell_init, {"tau2_init": float(fit2["tau2"])}

def fit_hppca(
    Y_obs: Any,
    d1: int,
    d2: int,
    kernel_method: str = "gp_rbf_single_ell",
    survey_times: np.ndarray | None = None,
    W1_init: np.ndarray | None = None,
    W2_init: np.ndarray | None = None,
    sigma2_init: float | None = None,
    ell_param_init: float | np.ndarray | None = None,
    init_method: str = "algo2_cs",
    n_cpus: int = 1,
    max_iter: int = 10000,
    tol: float = 1e-4,
    seed: int | None = None,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
    return_filled_dataframe: bool | None = None,
    return_latent_dataframe: bool = False,
):
    """
    Convenience wrapper:
      - accepts either the original 3D tensor input or tabular visit records,
      - builds participant lists directly (tabular path avoids dense tensor materialization),
      - initializes params if not provided,
      - runs EM_algorithm_gp,
      - returns the same outputs as EM_algorithm_gp plus the prep artifacts.

    Returns
    -------
    (W1f, W2f, s2f, ell_pf, EZ1i_list, EZ2ij_list, Y_filled, iteration_num, converged,
     Y_list, participant_survey_indices, participant_original_indices, survey_times)
    If `return_latent_dataframe=True`, one additional output is appended:
      latent_X_df (long latent table with z1/z2 columns).

    Notes
    -----
    - If `return_filled_dataframe` is None (default), `Y_filled` is:
        * a DataFrame for tabular input (2D numpy / DataFrame),
        * a tensor for 3D tensor input.
    - Set `return_filled_dataframe` explicitly to override this behavior.
    """
    if seed is not None:
        np.random.seed(seed)

    input_is_tensor = isinstance(Y_obs, np.ndarray) and Y_obs.ndim == 3
    if return_filled_dataframe is None:
        return_filled_dataframe = not input_is_tensor

    records_meta: Optional[Dict[str, Any]] = None
    if input_is_tensor:
        Y_obs_tensor, survey_times = prepare_hppca_input(
            Y_obs=Y_obs,
            survey_times=survey_times,
            subject_id_col=subject_id_col,
            visit_time_col=visit_time_col,
            feature_cols=feature_cols,
        )
        n_orig, J, p = Y_obs_tensor.shape
        Y_list, participant_survey_indices, participant_original_indices = build_lists_from_Y_obs(Y_obs_tensor)
    else:
        Y_list, participant_survey_indices, participant_original_indices, survey_times, records_meta = _records_to_lists(
            Y_records=Y_obs,
            survey_times=survey_times,
            subject_id_col=subject_id_col,
            visit_time_col=visit_time_col,
            feature_cols=feature_cols,
        )
        n_orig = int(np.asarray(records_meta["subject_ids"]).shape[0])
        p = int(np.asarray(records_meta["row_values"]).shape[1])
        J = int(np.asarray(survey_times).shape[0])
        Y_obs_tensor = None

    # 2) Choose initializer
    if init_method in ("algo1", "algo2_cs"):
        init_input = Y_obs_tensor if input_is_tensor else Y_obs
        try:
            if init_method == "algo1":
                W1_init, W2_init, sigma2_init, ell_param_init, _ = initialize_via_algo1(
                    init_input,
                    d1,
                    d2,
                    survey_times=survey_times,
                    kernel_method=kernel_method,
                    seed=seed,
                    subject_id_col=subject_id_col,
                    visit_time_col=visit_time_col,
                    feature_cols=feature_cols,
                    Y_list_for_init=(None if input_is_tensor else Y_list),
                    participant_survey_indices_for_init=(None if input_is_tensor else participant_survey_indices),
                )
            else:
                W1_init, W2_init, sigma2_init, ell_param_init, _ = initialize_via_algo2_cs(
                    init_input,
                    d1,
                    d2,
                    survey_times=survey_times,
                    kernel_method=kernel_method,
                    seed=seed,
                    subject_id_col=subject_id_col,
                    visit_time_col=visit_time_col,
                    feature_cols=feature_cols,
                )
        except Exception as exc:
            warnings.warn(
                f"Initialization via '{init_method}' failed "
                f"({type(exc).__name__}: {exc}). Falling back to random initialization."
            )
            W1_init, W2_init, sigma2_init, ell_param_init = initialize_parameters(
                p, d1, d2, J, survey_times, kernel_method
            )
    else:
        W1_init, W2_init, sigma2_init, ell_param_init = initialize_parameters(
            p, d1, d2, J, survey_times, kernel_method
        )

    # 3) Run EM
    (
        W1f, W2f, s2f, ell_pf,
        EZ1i_list, EZ2ij_list, Y_filled_raw,
        iteration_num, converged
    ) = EM_algorithm_gp(
        Y_list, W1_init, W2_init, sigma2_init, ell_param_init,
        kernel_method, survey_times, participant_survey_indices,
        Y_obs_tensor, participant_original_indices,
        records_meta=records_meta,
        n_cpus=n_cpus, max_iter=max_iter, tol=tol
    )

    if return_filled_dataframe:
        if pd is not None and isinstance(Y_filled_raw, pd.DataFrame):
            Y_filled_out = Y_filled_raw
        else:
            Y_filled_out = filled_tensor_to_dataframe(Y_filled_raw, survey_times)
    else:
        if pd is not None and isinstance(Y_filled_raw, pd.DataFrame):
            raise ValueError(
                "Tabular mode produced a DataFrame output. "
                "Set return_filled_dataframe=True (or leave it as None)."
            )
        Y_filled_out = Y_filled_raw

    base_out = (
        W1f, W2f, s2f, ell_pf,
        EZ1i_list, EZ2ij_list, Y_filled_out,
        iteration_num, converged,
        W1_init, W2_init, sigma2_init, ell_param_init,
        Y_list, participant_survey_indices, participant_original_indices, survey_times
    )
    if not return_latent_dataframe:
        return base_out

    latent_subject_ids = None
    if records_meta is not None:
        latent_subject_ids = np.asarray(records_meta["subject_ids"])
    latent_subject_col = subject_id_col if isinstance(subject_id_col, str) else "subject_id"
    latent_X_df = build_latent_long_dataframe(
        EZ1i_list=EZ1i_list,
        EZ2ij_list=EZ2ij_list,
        participant_survey_indices=participant_survey_indices,
        participant_original_indices=participant_original_indices,
        survey_times=survey_times,
        subject_ids=latent_subject_ids,
        subject_col_name=latent_subject_col,
        visit_col_name="visit_time",
    )
    return (*base_out, latent_X_df)

def impute_full_Y_after_em(
    Y_obs: np.ndarray,
    W1: np.ndarray,
    W2: np.ndarray,
    EZ1i_list: list,
    EZ2ij_list: list,
    survey_times: np.ndarray,
    participant_survey_indices: list,
    participant_original_indices: list,
    kernel_method: str,
    ell_param,
    return_dataframe: bool = False,
):
    """
    Impute all missing entries in `Y_obs` once after EM converges, including:
      - survey waves with no observed features for a participant, and
      - participants omitted from the E-step because they had no observations at all.

    Approach
    --------
    For each participant i that contributed to the E-step, we have E[Z1_i] and E[Z2_i(t_obs)].
    We extend E[Z2_i] to all survey times via GP conditioning:
        E[Z2_i(t_miss)] = K_mo K_oo^{-1} E[Z2_i(t_obs)]
    where K depends on `ell_param` per dimension (multi_ell) or shared (single_ell).
    For IID, unobserved times get E[Z2]=0.

    Then fill missing feature values with E[Y_ij] = W1 E[Z1_i] + W2 E[Z2_i(t_j)].
    Participants with no observations are filled using the model prior means (zeros).

    Parameters
    ----------
    return_dataframe : bool, default False
        If True, return a long-form DataFrame with columns:
        `subject_id`, `visit_time`, and `feature_k`.
    """
    n_orig, J, p = Y_obs.shape
    d1 = W1.shape[1]
    d2 = W2.shape[1]

    # Start from original observed tensor; fill NaNs only
    Y_filled = Y_obs.copy()

    # Precompute a list of all time indices for complement sets
    all_idx = np.arange(J)

    # Choose kernel function for GP
    kernel_fun = matern52_kernel if ("matern52" in kernel_method) else rbf_kernel
    use_time_gp = ("iid" not in kernel_method) and (d2 > 0)

    # Map from original participant index to its position in EZ lists
    orig_to_eff = {orig_i: eff_i for eff_i, orig_i in enumerate(participant_original_indices)}

    for orig_i in range(n_orig):
        # Determine if this participant was part of the E-step
        if orig_i in orig_to_eff:
            i_eff = orig_to_eff[orig_i]
            EZ1_i = EZ1i_list[i_eff] if (d1 > 0 and EZ1i_list[i_eff] is not None) else (np.zeros(d1) if d1 > 0 else None)
            EZ2_obs = EZ2ij_list[i_eff] if (d2 > 0 and EZ2ij_list[i_eff] is not None) else (np.zeros((d2, 0)) if d2 > 0 else None)
            idx_obs = participant_survey_indices[i_eff]
        else:
            # Participant had no observations at all
            EZ1_i = np.zeros(d1) if d1 > 0 else None
            EZ2_obs = np.zeros((d2, 0)) if d2 > 0 else None
            idx_obs = np.array([], dtype=int)

        # Build E[Z2] over all J times for this participant
        if d2 == 0:
            EZ2_full = None
        else:
            EZ2_full = np.zeros((d2, J))
            # Place observed-time posterior means directly
            if EZ2_obs is not None and idx_obs.size > 0:
                EZ2_full[:, idx_obs] = EZ2_obs

            # If GP across time is used, propagate to missing survey times
            idx_miss = np.setdiff1d(all_idx, idx_obs, assume_unique=True)
            if use_time_gp and idx_miss.size > 0 and idx_obs.size > 0:
                t_obs = survey_times[idx_obs]
                t_miss = survey_times[idx_miss]

                if "single_ell" in kernel_method:
                    ell = float(ell_param)
                    K_oo, _ = kernel_fun(t_obs, ell)
                    try:
                        K_oo_reg = K_oo.copy()
                        K_oo_reg[np.diag_indices_from(K_oo_reg)] += 1e-9
                        # Solve K_oo x = m instead of explicit inverse
                        K_oo_inv_EZ2 = np.linalg.solve(K_oo_reg, EZ2_obs.T).T  # shape (d2, |obs|)
                    except np.linalg.LinAlgError:
                        K_oo_inv = np.linalg.pinv(K_oo)
                        K_oo_inv_EZ2 = (K_oo_inv @ EZ2_obs.T).T
                    # Cross-cov between miss and obs
                    if "rbf" in kernel_method:
                        K_mo = np.exp(-0.5 * ((t_miss.reshape(-1,1) - t_obs.reshape(1,-1))**2) / (ell**2))
                    else:
                        from scipy.spatial.distance import cdist
                        r = cdist(t_miss.reshape(-1,1), t_obs.reshape(-1,1), metric='euclidean')
                        term1_val = SQRT5 * r / ell
                        term2_val = 5.0 * r**2 / (3.0 * ell**2)
                        K_mo = (1.0 + term1_val + term2_val) * np.exp(-term1_val)
                    EZ2_full[:, idx_miss] = (K_mo @ K_oo_inv_EZ2.T).T
                elif "multi_ell" in kernel_method:
                    ell_vec = np.asarray(ell_param)
                    for r in range(d2):
                        ell_r = float(ell_vec[r])
                        K_oo, _ = kernel_fun(t_obs, ell_r)
                        try:
                            K_oo_reg = K_oo.copy()
                            K_oo_reg[np.diag_indices_from(K_oo_reg)] += 1e-9
                            K_oo_inv_m = np.linalg.solve(K_oo_reg, np.eye(len(t_obs)))
                        except np.linalg.LinAlgError:
                            K_oo_inv_m = np.linalg.pinv(K_oo)
                        if "rbf" in kernel_method:
                            K_mo = np.exp(-0.5 * ((t_miss.reshape(-1,1) - t_obs.reshape(1,-1))**2) / (ell_r**2))
                        else:
                            from scipy.spatial.distance import cdist
                            r = cdist(t_miss.reshape(-1,1), t_obs.reshape(-1,1), metric='euclidean')
                            term1_val = SQRT5 * r / ell_r
                            term2_val = 5.0 * r**2 / (3.0 * ell_r**2)
                            K_mo = (1.0 + term1_val + term2_val) * np.exp(-term1_val)
                        EZ2_full[r, idx_miss] = K_mo @ (K_oo_inv_m @ EZ2_obs[r, :])
            # If IID or no observed times, EZ2 at missing times remains 0 (prior mean)

        # Now fill missing entries in Y with E[Y]
        for j in range(J):
            y_vec = Y_obs[orig_i, j, :]
            miss_mask = np.isnan(y_vec)
            if not np.any(miss_mask):
                continue
            # Expected Y at this time
            pred = np.zeros(p)
            if d1 > 0 and EZ1_i is not None:
                pred += W1 @ EZ1_i
            if d2 > 0 and EZ2_full is not None:
                pred += W2 @ EZ2_full[:, j]
            Y_filled[orig_i, j, miss_mask] = pred[miss_mask]

    if return_dataframe:
        return filled_tensor_to_dataframe(Y_filled, survey_times)
    return Y_filled
