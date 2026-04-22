# em_worker.py
import numpy as np
from scipy.linalg import cholesky
from scipy.spatial.distance import cdist  # For kernel functions

# pybind11 bindings
from .hppca_bindings import outer_prod_dbl, add_Gij
from .hppca_bindings import mat_inverse, add_identity_dbl

from .hppca_bindings import set_cblas_threads
set_cblas_threads(1)

# --- Constants ---
EPSILON_THRESH = 1e-15
ELL_MIN_THRESH = 1e-15
SQRT5 = np.sqrt(5.0)

G1_index_init, G2_index_init = [], []
is_observed_feature_mask, indices_for_Yp_values = [], []
is_missing_feature_mask,  indices_for_EYm_rows  = [], []

# Per-process cache: prior precision blocks for repeated (time-pattern, ell) pairs.
_prior_prec_cache = {}


def _ell_to_cache_key(ell_param, kernel_method):
    if "iid" in kernel_method:
        return ("iid",)
    if isinstance(ell_param, np.ndarray):
        return tuple(np.asarray(ell_param, dtype=float).tolist())
    return (float(ell_param),)


def _stable_inverse_from_kernel(K):
    n = K.shape[0]
    if n == 0:
        return np.zeros((0, 0))

    for jitter in (1e-9, 1e-8, 1e-7, 1e-6, 1e-5):
        try:
            K_reg = K.copy()
            add_identity_dbl(K_reg, jitter)
            L = cholesky(K_reg, lower=True, check_finite=False)
            Linv = mat_inverse(L)
            return Linv.T @ Linv
        except np.linalg.LinAlgError:
            continue

    return np.linalg.pinv(K)


def _get_prior_prec_Z2_cached(survey_idx_i, survey_times, ell_param, kernel_method, kernel_fun, d2):
    J_i = len(survey_idx_i)
    if J_i == 0 or d2 == 0:
        return np.zeros((J_i * d2, J_i * d2))

    if "iid" in kernel_method:
        return np.eye(J_i * d2)

    survey_idx_key = tuple(np.asarray(survey_idx_i, dtype=int).tolist())
    t_i = survey_times[survey_idx_i]
    t_key = tuple(np.asarray(t_i, dtype=float).tolist())
    cache_key = (
        kernel_method,
        d2,
        survey_idx_key,
        t_key,
        _ell_to_cache_key(ell_param, kernel_method),
    )
    cached = _prior_prec_cache.get(cache_key)
    if cached is not None:
        return cached

    prior_prec_Z2 = np.zeros((J_i * d2, J_i * d2))

    if "single_ell" in kernel_method:
        K, _ = kernel_fun(t_i, float(ell_param))
        K_inv = _stable_inverse_from_kernel(K)
        for r in range(d2):
            prior_prec_Z2[r::d2, r::d2] = K_inv
    elif "multi_ell" in kernel_method:
        ell_vec = np.asarray(ell_param, dtype=float).reshape(-1)
        for r in range(d2):
            K_r, _ = kernel_fun(t_i, float(ell_vec[r]))
            K_inv_r = _stable_inverse_from_kernel(K_r)
            prior_prec_Z2[r::d2, r::d2] = K_inv_r
    else:
        raise ValueError(f"Unknown kernel_method: {kernel_method}")

    if len(_prior_prec_cache) > 512:
        _prior_prec_cache.clear()
    _prior_prec_cache[cache_key] = prior_prec_Z2
    return prior_prec_Z2

# --- Kernel Functions  ---
def rbf_kernel(t, ell):
    t = np.asarray(t).reshape(-1, 1)
    sqdist = cdist(t, t, metric='sqeuclidean')
    if ell <= ELL_MIN_THRESH:
        return np.eye(len(t)), sqdist
    K = np.exp(-sqdist / (2 * ell**2))
    return K, sqdist

def rbf_kernel_grad(t, ell, K, sqdist):
    if ell <= ELL_MIN_THRESH:
        return np.zeros_like(K)
    K_prime = K * sqdist / (ell**3)
    return K_prime

def matern52_kernel(t, ell):
    t = np.asarray(t).reshape(-1, 1)
    r = cdist(t, t, metric='euclidean')
    if ell <= ELL_MIN_THRESH:
        return np.eye(len(t)), r
    term1_val = SQRT5 * r / ell
    term2_val = 5.0 * r**2 / (3.0 * ell**2)
    exp_val = np.exp(-term1_val)
    K = (1.0 + term1_val + term2_val) * exp_val
    return K, r

def matern52_kernel_grad1(t, ell, K_unused, r):
    if ell <= ELL_MIN_THRESH:
        return np.zeros_like(r)
    exp_term = np.exp(-SQRT5 * r / ell)
    factor1 = (5.0 * r**2) / (3.0 * ell**4)
    factor2 = ell + SQRT5 * r
    grad1 = exp_term * factor1 * factor2
    return grad1

def matern52_kernel_grad2(t, ell, K_unused, r):
    if ell <= ELL_MIN_THRESH:
        return np.zeros_like(r)
    if ell**6 == 0: return np.zeros_like(r)
    exp_term = np.exp(-SQRT5 * r / ell)
    term_in_paren = -5.0 * ell**2 - 5.0 * SQRT5 * r * ell + (25.0/3.0) * r**2
    grad2 = exp_term * (r**2 / ell**6) * term_in_paren
    return grad2

# --- G-Construction Functions (vectorized versions) ---
def add_Gij_vectorized(i_w, j_l, Yp_ij_g,
                       EZ1i_g, EZ2ij_g, EYmZ1T_g, EYmZ2T_g, sG1, sG2):
    global is_observed_feature_mask, indices_for_Yp_values
    global is_missing_feature_mask,  indices_for_EYm_rows
    add_Gij(Yp_ij_g, EZ1i_g, EZ2ij_g, EYmZ1T_g, EYmZ2T_g,
            is_observed_feature_mask[i_w][j_l],
            indices_for_Yp_values[i_w][j_l],
            is_missing_feature_mask[i_w][j_l],
            indices_for_EYm_rows[i_w][j_l],
            sG1, sG2)

def construct_Gij_vectorized(p_g, i_w, j_l, d1_g, d2_g, Yp_ij_g,
                             EZ1i_g, EZ2ij_g, EYmZ1T_g, EYmZ2T_g):
    G1 = np.zeros((p_g, d1_g))
    G2 = np.zeros((p_g, d2_g))
    if indices_for_Yp_values[i_w][j_l].size > 0:
        G1[is_observed_feature_mask[i_w][j_l], :] = outer_prod_dbl(Yp_ij_g[indices_for_Yp_values[i_w][j_l]], EZ1i_g)
        G2[is_observed_feature_mask[i_w][j_l], :] = outer_prod_dbl(Yp_ij_g[indices_for_Yp_values[i_w][j_l]], EZ2ij_g)
    if indices_for_EYm_rows[i_w][j_l].size > 0:
        if EYmZ1T_g.shape[0] > 0:
            G1[is_missing_feature_mask[i_w][j_l], :] = EYmZ1T_g[indices_for_EYm_rows[i_w][j_l], :]
        if EYmZ2T_g.shape[0] > 0:
            G2[is_missing_feature_mask[i_w][j_l], :] = EYmZ2T_g[indices_for_EYm_rows[i_w][j_l], :]
    return G1, G2

# --- Worker Initialization and Data Store ---
WORKER_SHARED_DATA = {}

def init_pool_worker(y_list_arg, w1_arg, w2_arg, sigma2_arg, ell_param_arg, kernel_method_arg,
                     survey_times_arg, participant_survey_indices_arg, 
                     index_mapping_list_arg, y_obs_full_arg, participant_original_indices_arg):
    """Initializer for each worker process in the pool."""
    WORKER_SHARED_DATA['Y_list'] = y_list_arg
    WORKER_SHARED_DATA['W1'] = w1_arg
    WORKER_SHARED_DATA['W2'] = w2_arg
    WORKER_SHARED_DATA['sigma2'] = sigma2_arg
    WORKER_SHARED_DATA['ell_param'] = ell_param_arg
    WORKER_SHARED_DATA['kernel_method'] = kernel_method_arg
    WORKER_SHARED_DATA['survey_times_full'] = survey_times_arg  # Full list of all possible survey times
    WORKER_SHARED_DATA['participant_survey_indices_list'] = participant_survey_indices_arg # List of arrays of indices
    WORKER_SHARED_DATA['index_mapping_list_all'] = index_mapping_list_arg # List of lists of map_mat
    WORKER_SHARED_DATA['Y_obs_full'] = y_obs_full_arg # The complete Y_obs (n_orig, J, p)
    WORKER_SHARED_DATA['participant_original_indices_list'] = participant_original_indices_arg # List of original participant indices

# --- E-step Core Logic (called by the pool worker) ---
def _execute_e_step_logic(
    i_part,
    Y_i,                       # (p, J_i) participant-specific matrix (NaNs allowed)
    orig_idx,                  # original participant index in Y_obs_full (0..n_orig-1)
    W1, W2,                    # (p,d1), (p,d2)
    sigma2,                    # scalar
    ell_param,                 # float or (d2,)
    kernel_method,             # str: "...single_ell"/"...multi_ell"/"...iid" (+ rbf/matern52)
    survey_times,              # (J,)
    survey_idx_i,              # (J_i,) indices into survey_times for this participant
    index_maps_i,              # list length J_i of (p,3) mapping matrices
    Y_obs_full,                # kept for signature compatibility; unused
    p, J_i, d1, d2             # dimensions for this participant
):
    """
    Compute E-step quantities for a single participant i_part.

    Returns
    -------
    (i_part, eyy_s2_i, den_s2_i,
     sG1_i, sG2_i,
     E_Z1Z1_i, sZ1Z2T_i, sZ2Z2T_i,
     S_Z2_i, Yfilled_i, EZ1_i, EZ2_i)

    Shapes:
      sG1_i: (p, d1)
      sG2_i: (p, d2)
      E_Z1Z1_i: (d1,d1)
      sZ1Z2T_i: (d1,d2)
      sZ2Z2T_i: (d2,d2)
      S_Z2_i:
        - single_ell: (J_i, J_i)
        - multi_ell: list of length d2, each (J_i, J_i)
        - iid: None
      Yfilled_i: None (legacy slot kept for compatibility)
      EZ1_i: (d1,)
      EZ2_i: (d2, J_i)
    """

    # We no longer materialize per-participant full panels here to avoid dense memory use.
    Yfilled_i = None
    # accumulators
    sG1_i = np.zeros((p, d1))
    sG2_i = np.zeros((p, d2))
    sZ1Z2T_i = np.zeros((d1, d2))
    sZ2Z2T_i = np.zeros((d2, d2))
    S_Z2_i = None

    EZ1_i = np.zeros(d1)
    EZ2_i = np.zeros((d2, J_i))

    # choose kernel (if any)
    kernel_fun = matern52_kernel if ("matern52" in kernel_method) else rbf_kernel

    # Build prior precision of Z2
    try:
        prior_prec_Z2 = _get_prior_prec_Z2_cached(
            survey_idx_i, survey_times, ell_param, kernel_method, kernel_fun, d2
        )
    except Exception:
        # kernel failed – return safe zeros so caller can skip
        return (
            i_part, 0.0, 0.0,
            np.zeros((p, d1)), np.zeros((p, d2)),
            np.zeros((d1, d1)), np.zeros((d1, d2)), np.zeros((d2, d2)),
            None, Yfilled_i, np.zeros(d1), np.zeros((d2, J_i))
        )

    # Likelihood pieces (block precision of [Z1; Z2_stack])
    OmA_lik = np.zeros((d1, d1))
    OmB_lik = np.zeros((d1, J_i * d2))
    OmD_lik = np.zeros((J_i * d2, J_i * d2))
    b1 = np.zeros(d1)
    b2_stack = np.zeros(J_i * d2)

    for j_col in range(J_i):
        Y_ij = Y_i[:, j_col]
        obs = ~np.isnan(Y_ij)

        Yp = Y_ij[obs]
        W1p = W1[obs, :]
        W2p = W2[obs, :]

        W1pT_W1p = W1p.T @ W1p if Yp.size > 0 else np.zeros((d1, d1))
        W1pT_W2p = W1p.T @ W2p if Yp.size > 0 else np.zeros((d1, d2))
        W2pT_W2p = W2p.T @ W2p if Yp.size > 0 else np.zeros((d2, d2))

        OmA_lik += W1pT_W1p
        if Yp.size > 0:
            b1 += W1p.T @ Yp

        s, e = j_col * d2, (j_col + 1) * d2
        OmB_lik[:, s:e] = W1pT_W2p
        OmD_lik[s:e, s:e] = W2pT_W2p
        if Yp.size > 0:
            b2_stack[s:e] = W2p.T @ Yp

    OmA_prior = np.eye(d1)
    OmA = OmA_prior + OmA_lik / sigma2
    OmD = OmD_lik / sigma2 + prior_prec_Z2
    OmB = OmB_lik / sigma2
    OmC = OmB.T

    try:
        Omega = np.block([[OmA, OmB], [OmC, OmD]])
    except ValueError:
        # shape mismatch (e.g., J_i*d2 == 0)
        return (
            i_part, 0.0, 0.0,
            np.zeros((p, d1)), np.zeros((p, d2)),
            np.zeros((d1, d1)), np.zeros((d1, d2)), np.zeros((d2, d2)),
            None, Yfilled_i, np.zeros(d1), np.zeros((d2, J_i))
        )

    b_stack = np.concatenate([b1, b2_stack]) / sigma2

    # Posterior moments of [Z1; Z2_stack]
    try:
        Om_copy = Omega.copy()
        add_identity_dbl(Om_copy, 1e-9)
        L = cholesky(Om_copy, lower=True, check_finite=False)
        Linv = mat_inverse(L)
        Sigma = Linv.T @ Linv
        Mu = Sigma @ b_stack
    except np.linalg.LinAlgError:
        try:
            Omega_pinv = np.linalg.pinv(Omega)
            Sigma, Mu = Omega_pinv, Omega_pinv @ b_stack
        except np.linalg.LinAlgError:
            return (
                i_part, 0.0, 0.0,
                np.zeros((p, d1)), np.zeros((p, d2)),
                np.zeros((d1, d1)), np.zeros((d1, d2)), np.zeros((d2, d2)),
                None, Yfilled_i, np.zeros(d1), np.zeros((d2, J_i))
            )

    EZ = Mu
    EZZT = Sigma + outer_prod_dbl(Mu, Mu)

    EZ1_i = EZ[:d1]
    E_Z1Z1_i = EZZT[:d1, :d1]
    E_Z2Z2 = EZZT[d1:, d1:]  # (J_i*d2, J_i*d2)

    # Build S_Z2_i for ell update if needed
    if ("single_ell" in kernel_method or "multi_ell" in kernel_method) and J_i > 0 and d2 > 0:
        E_Z2Z2_4d = E_Z2Z2.reshape(J_i, d2, J_i, d2)
        if "single_ell" in kernel_method:
            S_Z2_i = np.trace(E_Z2Z2_4d, axis1=1, axis2=3)
        else:
            S_Z2_i = [E_Z2Z2_4d[:, r, :, r].copy() for r in range(d2)]
    # for "iid" we leave S_Z2_i = None (not used downstream)

    # E[Y_ij^T Y_ij] terms for the sigma2 M-step. The W-dependent
    # reconstruction terms are evaluated after the new loadings are solved.
    sYpYp = sTrEYmYmT = 0.0

    Sigma_Z1Z1 = Sigma[:d1, :d1]

    for j in range(J_i):
        orig_j = survey_idx_i[j]
        y_col = Y_i[:, j]
        miss = np.isnan(y_col)
        obs = ~miss

        Yp = y_col[obs]
        W1p, W1m = W1[obs, :], W1[miss, :]
        W2p, W2m = W2[obs, :], W2[miss, :]

        s, e = d1 + j * d2, d1 + (j + 1) * d2
        EZ2_ij = EZ[s:e]
        EZ2_i[:, j] = EZ2_ij

        EZ1Z2T = EZZT[:d1, s:e]
        EZ2Z1T = EZZT[s:e, :d1]
        EZ2Z2 = EZZT[s:e, s:e]

        sZ1Z2T_i += EZ1Z2T
        sZ2Z2T_i += EZ2Z2

        EY_m = W1m @ EZ1_i + W2m @ EZ2_ij

        if np.any(miss):
            # E[Y_m Z^T] blocks
            EYmZ1T = W1m @ E_Z1Z1_i + W2m @ EZ2Z1T
            EYmZ2T = W1m @ EZ1Z2T + W2m @ EZ2Z2

            # Var[Y_m]
            Sigma_Z1Z2 = Sigma[:d1, s:e]
            Sigma_Z2Z1 = Sigma_Z1Z2.T
            Sigma_Z2Z2 = Sigma[s:e, s:e]

            VarYm = W2m @ Sigma_Z2Z2 @ W2m.T

            if d1 > 0:
                VarYm += W1m @ Sigma_Z1Z1 @ W1m.T
                VarYm += W1m @ Sigma_Z1Z2 @ W2m.T + W2m @ Sigma_Z2Z1 @ W1m.T

            add_identity_dbl(VarYm, sigma2)
            EYmYmT = VarYm + outer_prod_dbl(EY_m, EY_m)
        else:
            EYmZ1T = np.zeros((0, d1))
            EYmZ2T = np.zeros((0, d2))
            EYmYmT = np.zeros((0, 0))

        if Yp.size > 0:
            sYpYp += Yp.T @ Yp

        sTrEYmYmT += np.trace(EYmYmT)

        # accumulate G terms with your vectorized helper (uses globals set by construct_index_mapping)
        #idx_map_j = index_maps_i[j]
        #obs_mask_loc, yp_idx_loc, miss_mask_loc, eym_idx_loc = _masks_from_index_map(idx_map_j)
        #add_Gij(
            #Yp,            # observed responses (len = sum(obs))
            #EZ1_i,         # (d1,)   — length 0 if d1==0
            #EZ2_ij,        # (d2,)
            #EYmZ1T,        # (num_missing, d1)
            #EYmZ2T,        # (num_missing, d2)
            #obs_mask_loc,  # (p,) bool
            #yp_idx_loc,    # (sum(obs),) int
            #miss_mask_loc, # (p,) bool
            #eym_idx_loc,   # (sum(miss),) int
            #sG1_i,         # (p, d1)
            #sG2_i,         # (p, d2)
        #)
        add_Gij_vectorized(i_part, j, Yp, EZ1_i, EZ2_ij, EYmZ1T, EYmZ2T, sG1_i, sG2_i)
        #sG1_i, sG2_i = construct_Gij_vectorized(p, i_part, j, d1, d2, Yp, EZ1_i, EZ2_ij, EYmZ1T, EYmZ2T)

    eyy_s2_i = sYpYp + sTrEYmYmT
    den_s2_i = p * J_i if p * J_i > 0 else 1.0

    return (
        i_part, eyy_s2_i, den_s2_i,
        sG1_i, sG2_i,
        E_Z1Z1_i, sZ1Z2T_i, sZ2Z2T_i,
        S_Z2_i, Yfilled_i, EZ1_i, EZ2_i
    )


# --- Pool Worker Function (what pool.map calls) ---
def _e_step_worker_for_pool(i_part):
    """
    Wrapper for multiprocessing: pulls shared data for participant `i_part`,
    then calls _execute_e_step_logic with clean names.
    """
    Y_list = WORKER_SHARED_DATA['Y_list']
    W1 = WORKER_SHARED_DATA['W1']
    W2 = WORKER_SHARED_DATA['W2']
    sigma2 = WORKER_SHARED_DATA['sigma2']
    ell_param = WORKER_SHARED_DATA['ell_param']
    kernel_method = WORKER_SHARED_DATA['kernel_method']
    survey_times = WORKER_SHARED_DATA['survey_times_full']
    survey_idx_list = WORKER_SHARED_DATA['participant_survey_indices_list']
    index_maps_list = WORKER_SHARED_DATA['index_mapping_list_all']
    Y_obs_full = WORKER_SHARED_DATA['Y_obs_full']
    orig_idx_list = WORKER_SHARED_DATA['participant_original_indices_list']

    Y_i = Y_list[i_part]
    orig_idx = orig_idx_list[i_part]

    p = W1.shape[0]
    J_i = Y_i.shape[1] if (Y_i.ndim == 2 and Y_i.shape[1] > 0) else 0

    # no observed waves -> fast return
    if J_i == 0:
        d1 = W1.shape[1]
        d2 = W2.shape[1]
        return (
            i_part, 0.0, 0.0,
            np.zeros((p, d1)), np.zeros((p, d2)),
            np.zeros((d1, d1)), np.zeros((d1, d2)), np.zeros((d2, d2)),
            None, None, np.zeros(d1), np.zeros((d2, 0))
        )

    survey_idx_i = survey_idx_list[i_part]
    index_maps_i = index_maps_list[i_part]
    d1 = W1.shape[1]
    d2 = W2.shape[1]

    return _execute_e_step_logic(
        i_part, Y_i, orig_idx, W1, W2, sigma2, ell_param, kernel_method,
        survey_times, survey_idx_i, index_maps_i, Y_obs_full, p, J_i, d1, d2
    )
