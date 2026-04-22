"""
HPPCA — Algorithm 1 (known Sigma under Assumption 1) and
Algorithm 2 / 3 in derivation PDF (compound-symmetry Sigma(tau^2)).

Input Y can be:
  - dense tensor: (n_participants, n_surveys, n_features), or
  - long/tabular records: pandas DataFrame / 2D numpy array with
      subject_id, visit_time(optional), and feature columns.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.linalg import eigh, svd, norm, qr
import os

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from .hppca_bindings import mat_prod, trace_prod_dbl, set_cblas_threads
    _HAVE_BINDINGS = True
except Exception:  # pragma: no cover
    _HAVE_BINDINGS = False

if _HAVE_BINDINGS:
    _threads = os.getenv("HPPCA_ALGO_BLAS_THREADS")
    if _threads:
        try:
            set_cblas_threads(max(1, int(_threads)))
        except Exception:
            pass


# -----------------------------
# Utilities
# -----------------------------

def _as_c_f64(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(a, dtype=np.float64))


def _gemm(m0: np.ndarray, m1: np.ndarray, t0: bool = False, t1: bool = False) -> np.ndarray:
    """
    Matrix multiplication with optional pybind11/OpenBLAS implementation.
    Falls back to NumPy matmul if bindings are unavailable.
    """
    if not _HAVE_BINDINGS:
        A = m0.T if t0 else m0
        B = m1.T if t1 else m1
        return A @ B

    A = _as_c_f64(m0)
    B = _as_c_f64(m1)
    m = A.shape[1] if t0 else A.shape[0]
    n = B.shape[0] if t1 else B.shape[1]
    out = np.empty((m, n), dtype=np.float64)
    mat_prod(A, B, out, t0, t1, 1.0, 0.0)
    return out


def _trace_ab(m0: np.ndarray, m1: np.ndarray) -> float:
    """
    Compute trace(m0 @ m1) with optional pybind11 helper.
    """
    if _HAVE_BINDINGS:
        return float(trace_prod_dbl(_as_c_f64(m0), _as_c_f64(m1)))
    return float(np.trace(m0 @ m1))


def _symmetrize(M: np.ndarray) -> np.ndarray:
    return (M + M.T) / 2.0

def _orthonormal_complement(u1: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Given u1 (J,), return U_perp (J, J-1) with orthonormal columns spanning u1^perp.
    """
    J = u1.shape[0]
    if rng is None:
        rng = np.random.default_rng(0)
    R = rng.standard_normal((J, J - 1))
    R = R - np.outer(u1, u1 @ R)
    Q, _ = qr(R, mode="reduced")
    return Q


def _householder_ut(J: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """
    Build UT whose first column is 1/sqrt(J) * 1 and remaining columns are an orthonormal complement.
    """
    u1 = np.ones(J) / np.sqrt(J)
    Uperp = _orthonormal_complement(u1, rng=rng)
    return np.column_stack([u1, Uperp])


def _sigma_eigendecomp_under_assumption1(Sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Diagonalize Sigma in a basis whose first column is exactly 1/sqrt(J) * 1.
    This is required under Assumption 1, and avoids the gp_iid/I_J degeneracy where a raw
    eigendecomposition may return an arbitrary basis that does not isolate the mean direction.
    """
    Sigma = _symmetrize(np.asarray(Sigma, dtype=float))
    J = Sigma.shape[0]
    u1 = np.ones(J) / np.sqrt(J)
    lam1 = float(u1 @ Sigma @ u1)
    if J == 1:
        return np.array([lam1], dtype=float), u1[:, None]

    Uperp0 = _orthonormal_complement(u1, rng=np.random.default_rng(0))
    Sigma_perp = _symmetrize(Uperp0.T @ Sigma @ Uperp0)
    w_perp, V_perp = eigh(Sigma_perp)
    idx = np.argsort(w_perp)[::-1]
    Uperp = Uperp0 @ V_perp[:, idx]
    UT = np.column_stack([u1, Uperp])
    lambdas = np.concatenate(([lam1], w_perp[idx]))
    return lambdas, UT


def _rotate_time_and_covs(Y: np.ndarray, UT: np.ndarray):
    """
    Rotate time dimension of Y with UT (J x J).
    Returns:
      - St_list: list of p x p empirical covariances for rotated columns t=0..J-1
      - S1: St_list[0]
      - Sc: average over t=1..J-1
    """
    n, J, p = Y.shape
    Ye = np.matmul(np.transpose(Y, (0, 2, 1)), UT)  # (n, p, J)
    St_stack = np.einsum("npt,nqt->tpq", Ye, Ye, optimize=True) / max(float(n), 1.0)
    S1 = St_stack[0]
    Sc = St_stack[1:].mean(axis=0) if J > 1 else np.zeros((p, p))
    return [St_stack[t] for t in range(J)], S1, Sc


def _prepare_long_records(
    Y: Any,
    survey_times: np.ndarray | None = None,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
):
    """
    Convert long/tabular data into compact arrays used by covariance builders.
    Returns dict with:
      n_subjects, J, p, survey_times, row_subject_idx, row_visit_idx, row_values.
    """
    if pd is None:
        raise ImportError("pandas is required for tabular Algorithm 1/2 input.")

    if isinstance(Y, np.ndarray):
        arr = np.asarray(Y)
        if arr.ndim != 2:
            raise ValueError("Tabular numpy input for Algorithm 1/2 must be 2D.")
        n_cols = arr.shape[1]

        sid_idx = subject_id_col if isinstance(subject_id_col, int) else 0
        sid_idx = sid_idx + n_cols if sid_idx < 0 else sid_idx
        if sid_idx < 0 or sid_idx >= n_cols:
            raise ValueError("subject_id_col index is out of bounds.")

        if visit_time_col is None:
            vt_idx = None
        else:
            vt_idx = visit_time_col if isinstance(visit_time_col, int) else 1
            vt_idx = vt_idx + n_cols if vt_idx < 0 else vt_idx
            if vt_idx < 0 or vt_idx >= n_cols:
                raise ValueError("visit_time_col index is out of bounds.")

        if feature_cols is None:
            blocked = {sid_idx}
            if vt_idx is not None:
                blocked.add(vt_idx)
            feat_idx = [k for k in range(n_cols) if k not in blocked]
        else:
            feat_idx = []
            for c in feature_cols:
                if not isinstance(c, int):
                    raise TypeError("feature_cols for 2D numpy input must be integer indices.")
                cc = c + n_cols if c < 0 else c
                if cc < 0 or cc >= n_cols:
                    raise ValueError("A feature_cols index is out of bounds.")
                feat_idx.append(cc)
        if len(feat_idx) == 0:
            raise ValueError("No feature columns found for tabular input.")

        data = {"subject_id": arr[:, sid_idx]}
        if vt_idx is not None:
            data["visit_time"] = arr[:, vt_idx]
        for k, cidx in enumerate(feat_idx):
            data[f"feature_{k}"] = arr[:, cidx]
        df = pd.DataFrame(data)
        sid_col = "subject_id"
        time_col = "visit_time" if vt_idx is not None else None
        feat_cols = [f"feature_{k}" for k in range(len(feat_idx))]
    elif isinstance(Y, pd.DataFrame):
        df = Y.copy()
        sid_col = subject_id_col
        time_col = visit_time_col
        if isinstance(sid_col, int):
            sid_col = df.columns[sid_col]
        if isinstance(time_col, int):
            time_col = df.columns[time_col]
        if sid_col not in df.columns:
            raise ValueError("subject_id column not found in DataFrame.")
        if time_col is not None and time_col not in df.columns:
            raise ValueError("visit_time column not found in DataFrame.")
        if feature_cols is None:
            excluded = {sid_col}
            if time_col is not None:
                excluded.add(time_col)
            feat_cols = [c for c in df.columns if c not in excluded]
        else:
            feat_cols = []
            for c in feature_cols:
                col = df.columns[c] if isinstance(c, int) else c
                if col not in df.columns:
                    raise ValueError(f"feature column '{c}' not found in DataFrame.")
                feat_cols.append(col)
    else:
        raise TypeError("Y must be a 3D tensor, DataFrame, or 2D tabular numpy array.")

    if len(df) == 0:
        raise ValueError("Tabular input is empty.")
    if df[sid_col].isna().any():
        raise ValueError("subject_id contains missing values.")
    if len(feat_cols) == 0:
        raise ValueError("No feature columns were selected.")

    subject_ids = pd.unique(df[sid_col])
    sid_to_idx = {sid: i for i, sid in enumerate(subject_ids)}
    row_subject_idx = df[sid_col].map(sid_to_idx).to_numpy(dtype=np.int32)

    if time_col is not None and df[time_col].notna().any():
        time_vals = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isfinite(time_vals)):
            raise ValueError("visit_time contains non-numeric or missing values.")
        if survey_times is None:
            survey_times_arr = np.sort(np.unique(time_vals))
            row_visit_idx = np.searchsorted(survey_times_arr, time_vals).astype(np.int32)
            if not np.all(np.isclose(survey_times_arr[row_visit_idx], time_vals, rtol=0.0, atol=1e-10)):
                raise ValueError("visit_time values must match inferred survey_times (within tolerance).")
        else:
            survey_times_arr = np.asarray(survey_times, dtype=float).reshape(-1)
            if survey_times_arr.size == 0:
                raise ValueError("survey_times must be non-empty when provided.")
            row_visit_idx = np.full(time_vals.shape[0], -1, dtype=np.int32)
            for m, tv in enumerate(time_vals):
                matches = np.where(np.isclose(survey_times_arr, tv, rtol=0.0, atol=1e-10))[0]
                if matches.size == 0:
                    raise ValueError("visit_time values must match provided survey_times (within tolerance).")
                row_visit_idx[m] = int(matches[0])
    else:
        row_visit_idx = df.groupby(sid_col, sort=False).cumcount().to_numpy(dtype=np.int32)
        J = int(np.max(row_visit_idx)) + 1
        if survey_times is None:
            survey_times_arr = np.linspace(10.0, 10.0 * J, J, dtype=float)
        else:
            survey_times_arr = np.asarray(survey_times, dtype=float).reshape(-1)
            if survey_times_arr.size != J:
                raise ValueError(
                    f"survey_times length ({survey_times_arr.size}) must equal implied J ({J}) when visit_time is absent."
                )

    dup_mask = pd.DataFrame({"subject_idx": row_subject_idx, "visit_idx": row_visit_idx}).duplicated(keep=False)
    if bool(dup_mask.any()):
        raise ValueError("Each (subject_id, visit_time) pair must appear at most once.")

    row_values = df.loc[:, feat_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float, copy=True)
    return {
        "n_subjects": int(len(subject_ids)),
        "J": int(len(survey_times_arr)),
        "p": int(len(feat_cols)),
        "survey_times": np.asarray(survey_times_arr, dtype=float),
        "row_subject_idx": row_subject_idx,
        "row_visit_idx": row_visit_idx.astype(np.int32, copy=False),
        "row_values": row_values,
    }


def _rotate_time_and_covs_from_long(records: dict, UT: np.ndarray):
    """
    Build rotated time covariances directly from long records without dense n x J x p tensor storage.
    This matches the dense initializer path where every subject is first completed onto the
    shared J-visit grid by participant-feature means (with cohort-feature fallback when needed),
    then globally centered before the time rotation is applied.
    """
    n = int(records["n_subjects"])
    J = int(records["J"])
    p = int(records["p"])
    row_subject_idx = np.asarray(records["row_subject_idx"], dtype=np.int32)
    row_visit_idx = np.asarray(records["row_visit_idx"], dtype=np.int32)
    row_values = np.asarray(records["row_values"], dtype=float)

    finite_mask = np.isfinite(row_values)
    vals0 = np.where(finite_mask, row_values, 0.0)
    cnt0 = finite_mask.astype(np.int64)

    sum_p = vals0.sum(axis=0)
    cnt_p = cnt0.sum(axis=0)
    mu_p = np.divide(sum_p, cnt_p, out=np.zeros(p), where=cnt_p > 0)

    sum_ip = np.zeros((n, p), dtype=float)
    cnt_ip = np.zeros((n, p), dtype=np.int64)
    np.add.at(sum_ip, row_subject_idx, vals0)
    np.add.at(cnt_ip, row_subject_idx, cnt0)
    mu_ip = np.divide(
        sum_ip,
        cnt_ip,
        out=np.broadcast_to(mu_p, (n, p)).copy(),
        where=cnt_ip > 0,
    )

    total_ip = sum_ip + (J - cnt_ip) * mu_ip
    global_mean = total_ip.sum(axis=0) / max(float(n * J), 1.0)

    St_stack = np.zeros((J, p, p), dtype=float)
    ones_rot = np.sum(UT, axis=0)  # shape (J,)

    order = np.argsort(row_subject_idx, kind="mergesort")
    subj_ord = row_subject_idx[order]
    visit_ord = row_visit_idx[order]
    vals_ord = row_values[order, :]

    if order.size > 0:
        starts = np.r_[0, np.flatnonzero(np.diff(subj_ord)) + 1]
        ends = np.r_[starts[1:], order.size]
        uniq_subj = subj_ord[starts]
    else:
        starts = np.array([], dtype=int)
        ends = np.array([], dtype=int)
        uniq_subj = np.array([], dtype=np.int32)

    for subj, st, ed in zip(uniq_subj, starts, ends):
        subj = int(subj)
        base = mu_ip[subj] - global_mean
        Zi = np.outer(base, ones_rot)

        vv = visit_ord[st:ed]
        rows = vals_ord[st:ed, :]
        rows_filled = np.where(np.isfinite(rows), rows, mu_ip[subj][None, :])
        delta = rows_filled - mu_ip[subj][None, :]
        Zi += _gemm(delta.T, UT[vv, :])

        St_stack += np.einsum("pt,qt->tpq", Zi, Zi, optimize=True)

    St_stack /= max(float(n), 1.0)
    S1 = St_stack[0]
    Sc = St_stack[1:].mean(axis=0) if J > 1 else np.zeros((p, p))
    return [St_stack[t] for t in range(J)], S1, Sc


def _project_top_positive(S: np.ndarray, d: int) -> np.ndarray:
    """
    Positive-part projection Pi_+^{(d)}(S): keep up to d positive eigenvalues.
    """
    w, U = eigh(_symmetrize(S))
    idx = np.argsort(w)[::-1]
    w = w[idx]
    U = U[:, idx]
    keep = (w > 0)
    keep_idx = np.where(keep)[0][:d]
    nu = np.zeros_like(w)
    nu[keep_idx] = w[keep_idx]
    return (U * nu) @ U.T


def _build_B_from_qb(Q: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _symmetrize(_gemm(Q * b[None, :], Q, t1=True))


def _profile_A_from_mean_block(
    S1: np.ndarray,
    B: np.ndarray,
    sigma2: float,
    lambda1: float,
    J: int,
    d1: int,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Theorem 1 profile update:
      A* = J^{-1} H^{1/2} Pi_+^{(d1)}(H^{-1/2} S1 H^{-1/2} - I) H^{1/2},
    where H = lambda1 * B + sigma2 * I.
    """
    p = S1.shape[0]
    H = _symmetrize(lambda1 * B + sigma2 * np.eye(p))
    w_h, U_h = eigh(H)
    w_h = np.maximum(w_h, eps)
    H_half = (U_h * np.sqrt(w_h)) @ U_h.T
    H_mhalf = (U_h * (1.0 / np.sqrt(w_h))) @ U_h.T
    T = _symmetrize(H_mhalf @ _symmetrize(S1) @ H_mhalf)
    M_star = _project_top_positive(T - np.eye(p), d1)
    return _symmetrize((H_half @ M_star @ H_half) / max(float(J), 1.0))


def _mean_direction_objective(
    S1: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    sigma2: float,
    lambda1: float,
    J: int,
    eps: float = 1e-12,
) -> float:
    p = S1.shape[0]
    M1 = _symmetrize(J * A + lambda1 * B + sigma2 * np.eye(p))
    w1, U1 = eigh(M1)
    w1 = np.maximum(w1, eps)
    logdet1 = float(np.sum(np.log(w1)))
    Minv = (U1 * (1.0 / w1)) @ U1.T
    return float(logdet1 + _trace_ab(_symmetrize(S1), Minv))


def _principal_angle_deg(W_true: np.ndarray, W_est: np.ndarray) -> float:
    if W_true.size == 0 or W_est.size == 0:
        return np.nan
    U1, _, _ = svd(W_true, full_matrices=False)
    U2, _, _ = svd(W_est, full_matrices=False)
    s = svd(U1.T @ U2, compute_uv=False)
    cmin = float(np.clip(s.min(), -1.0, 1.0))
    return float(np.degrees(np.arccos(cmin)))


def _relative_frobenius_loss(A: np.ndarray, B: np.ndarray) -> float:
    return float(norm(A - B, "fro") / max(norm(B, "fro"), 1e-12))


def _cov_y_full(A: np.ndarray, B: np.ndarray, Sigma: np.ndarray, sigma2: float) -> np.ndarray:
    J = Sigma.shape[0]
    p = A.shape[0]
    return np.kron(np.ones((J, J)), A) + np.kron(Sigma, B) + sigma2 * np.eye(p * J)


# -----------------------------
# Algorithm 1 (known Sigma under Assumption 1)
# -----------------------------

def fit_hppca_alg1(
    Y: Any,
    Sigma: np.ndarray,
    d1: int,
    d2: int,
    survey_times: np.ndarray | None = None,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
    max_outer: int = 30,
    tol_outer: float = 1e-8,
    max_irls: int = 20,
    tol_irls: float = 1e-8,
    max_q: int = 200,
    tol_q: float = 1e-8,
    step_eta0: float = 1.0,
    bt_c: float = 1e-4,
    bt_beta: float = 0.5,
    eps: float = 1e-9,
    rng: np.random.Generator | None = None,
    verbose: bool = False,
):
    """
    Block-coordinate MLE for HPPCA when Sigma is known and satisfies Assumption 1.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if isinstance(Y, np.ndarray) and Y.ndim == 3:
        _, J, p = Y.shape
        is_long = False
    else:
        records = _prepare_long_records(
            Y,
            survey_times=survey_times,
            subject_id_col=subject_id_col,
            visit_time_col=visit_time_col,
            feature_cols=feature_cols,
        )
        J = records["J"]
        p = records["p"]
        is_long = True

    def _log(msg: str):
        if verbose:
            print(msg)

    def _warn(msg: str):
        print(msg)

    def _truncate_b_inplace(vec: np.ndarray) -> np.ndarray:
        if d2 <= 0:
            vec[:] = 0.0
            return vec
        if d2 >= vec.shape[0]:
            return vec
        idx_keep = np.argpartition(vec, -d2)[-d2:]
        mask = np.zeros_like(vec, dtype=bool)
        mask[idx_keep] = True
        vec[~mask] = 0.0
        return vec

    def contrast_objective_from_sdiag(Sdiag: np.ndarray, b: np.ndarray, sigma2: float) -> float:
        if J <= 1:
            return 0.0
        a = np.maximum(lambdas[1:, None] * b[None, :] + sigma2, eps)
        return float(np.sum(np.log(a) + Sdiag / a))

    lambdas, UT = _sigma_eigendecomp_under_assumption1(Sigma)

    if is_long:
        St_list, S1, Sc = _rotate_time_and_covs_from_long(records, UT)
    else:
        St_list, S1, Sc = _rotate_time_and_covs(Y, UT)

    St_stack = np.stack(St_list, axis=0)
    St_tail = St_stack[1:] if J > 1 else np.zeros((0, p, p))

    w_c, Q = eigh((Sc + Sc.T) / 2.0)
    idx = np.argsort(w_c)[::-1]
    w_c = w_c[idx]
    Q = Q[:, idx]

    lam_c_bar = float(np.mean(lambdas[1:])) if J > 1 else 1.0
    sigma2 = float(np.mean(w_c[d2:])) if d2 < p else 1e-6
    b = np.zeros(p)
    lam_c_bar = lam_c_bar if lam_c_bar > 0 else 1.0
    for k in range(min(d2, p)):
        b[k] = max((w_c[k] - sigma2) / lam_c_bar, 0.0)
    _truncate_b_inplace(b)

    def irls_update(b, sigma2, outer_iter_idx=None):
        for irls_iter in range(max_irls):
            Sdiag = (
                np.einsum("ki,tkl,li->ti", Q, St_tail, Q, optimize=True)
                if J > 1 else np.zeros((0, p))
            )

            a = lambdas[1:, None] * b[None, :] + sigma2
            a = np.maximum(a, eps)
            w = 1.0 / (a * a)

            num = np.sum(lambdas[1:, None] * w * (Sdiag - sigma2), axis=0)
            den = np.sum((lambdas[1:, None] ** 2) * w, axis=0) + eps
            b_new = np.maximum(num / den, 0.0)
            _truncate_b_inplace(b_new)

            num_s = np.sum(w * (Sdiag - lambdas[1:, None] * b_new[None, :]))
            den_s = np.sum(w) + eps
            sigma2_new = max(num_s / den_s, eps)

            b_prev = b.copy()
            sigma2_prev = float(sigma2)
            old_obj = contrast_objective_from_sdiag(Sdiag, b_prev, sigma2_prev)
            step = 1.0
            accepted = False
            for _ in range(50):
                b_try = np.maximum(b_prev + step * (b_new - b_prev), 0.0)
                _truncate_b_inplace(b_try)
                sigma2_try = max(sigma2_prev + step * (sigma2_new - sigma2_prev), eps)
                new_obj = contrast_objective_from_sdiag(Sdiag, b_try, sigma2_try)
                if new_obj <= old_obj + 1e-12:
                    b_new = b_try
                    sigma2_new = sigma2_try
                    accepted = True
                    break
                step *= bt_beta
            if not accepted:
                b_new = b_prev
                sigma2_new = sigma2_prev

            ch = max(
                abs(sigma2_new - sigma2) / (abs(sigma2) + 1e-12),
                norm(b_new - b) / (norm(b) + 1e-12),
            )
            b, sigma2 = b_new, sigma2_new
            if verbose:
                prefix = f"[Alg1][outer {outer_iter_idx}] " if outer_iter_idx is not None else "[Alg1] "
                _log(f"{prefix}IRLS iter {irls_iter}: change={ch:.3e}, sigma2={sigma2:.6g}")
            if ch < tol_irls:
                break
        return b, sigma2

    def q_step(Q, b, sigma2, outer_iter_idx=None):
        inv_a = (
            1.0 / np.maximum(lambdas[1:, None] * b[None, :] + sigma2, eps)
            if J > 1 else np.zeros((0, p))
        )

        def project_to_stiefel(Yt: np.ndarray, q_iter_idx: int, eta: float) -> np.ndarray:
            prefix = "[Alg1]"
            if outer_iter_idx is not None:
                prefix += f"[outer {outer_iter_idx}]"
            prefix += f"[Q-step {q_iter_idx}]"

            if not np.isfinite(Yt).all():
                max_abs = float(np.nanmax(np.abs(Yt)))
                raise FloatingPointError(
                    f"{prefix} projection input contains non-finite values "
                    f"(eta={eta:.3e}, max_abs={max_abs:.3e})."
                )

            last_err: np.linalg.LinAlgError | None = None
            for attempt in range(3):
                try:
                    U, _, Vt = svd(np.asarray(Yt, dtype=np.float64), full_matrices=False)
                    if attempt > 0:
                        _warn(
                            f"{prefix} SVD succeeded on retry {attempt + 1}/3 "
                            f"(eta={eta:.3e})."
                        )
                    return U @ Vt
                except np.linalg.LinAlgError as err:
                    last_err = err
                    if attempt < 2:
                        _warn(
                            f"{prefix} SVD failed on attempt {attempt + 1}/3 "
                            f"(eta={eta:.3e}): {err}. Retrying."
                        )

            gram = _symmetrize(Yt.T @ Yt)
            try:
                eigvals, eigvecs = eigh(gram)
            except np.linalg.LinAlgError as err:
                raise np.linalg.LinAlgError(
                    f"{prefix} SVD failed after 3 attempts and fallback eigendecomposition "
                    f"also failed (eta={eta:.3e})."
                ) from err

            if not np.isfinite(eigvals).all():
                raise np.linalg.LinAlgError(
                    f"{prefix} SVD failed after 3 attempts and fallback eigendecomposition "
                    f"produced non-finite eigenvalues (eta={eta:.3e})."
                ) from last_err

            scale = max(float(np.max(np.abs(eigvals))), 1.0)
            ridge = max(eps, 1e-12 * scale)
            eigvals_floor = np.maximum(eigvals, ridge)
            inv_sqrt = eigvecs @ ((1.0 / np.sqrt(eigvals_floor))[:, None] * eigvecs.T)
            Q_try = Yt @ inv_sqrt
            if not np.isfinite(Q_try).all():
                raise np.linalg.LinAlgError(
                    f"{prefix} fallback polar projection produced non-finite values "
                    f"(eta={eta:.3e})."
                ) from last_err

            ortho_err = norm(
                Q_try.T @ Q_try - np.eye(Q_try.shape[1], dtype=Q_try.dtype),
                "fro",
            )
            _warn(
                f"{prefix} SVD failed after 3 attempts (eta={eta:.3e}): {last_err}. "
                f"Using Gram-based polar fallback with ridge={ridge:.3e}, "
                f"ortho_err={ortho_err:.3e}."
            )
            return Q_try

        def f_of_Q(Q):
            if J <= 1:
                return 0.0
            Dt_diag = np.einsum("ki,tkl,li->ti", Q, St_tail, Q, optimize=True)
            return float(np.sum(Dt_diag * inv_a))

        def grad(Q):
            G = np.zeros_like(Q)
            if J > 1:
                MQ = np.einsum("tij,jk->tik", St_tail, Q, optimize=True)
                G = np.einsum("tik,tk->ik", MQ, inv_a, optimize=True)
            QtG = Q.T @ G
            return G - Q @ ((QtG + QtG.T) / 2.0)

        cur_val = f_of_Q(Q)
        for q_iter in range(max_q):
            R = grad(Q)
            gnorm = norm(R, "fro")
            if verbose:
                prefix = f"[Alg1][outer {outer_iter_idx}] " if outer_iter_idx is not None else "[Alg1] "
                _log(f"{prefix}Q-step iter {q_iter}: grad_norm={gnorm:.3e}, obj={cur_val:.6e}")
            if gnorm < tol_q:
                break
            eta = step_eta0
            for _ in range(50):
                Yt = Q - eta * R
                Q_try = project_to_stiefel(Yt, q_iter, eta)
                new_val = f_of_Q(Q_try)
                if new_val <= cur_val - bt_c * eta * (gnorm ** 2):
                    Q = Q_try
                    cur_val = new_val
                    break
                eta *= bt_beta
            else:
                break
        return Q

    obj_hist = []
    for outer_iter in range(max_outer):
        b, sigma2 = irls_update(b, sigma2, outer_iter_idx=outer_iter)
        Q = q_step(Q, b, sigma2, outer_iter_idx=outer_iter)
        B = _build_B_from_qb(Q, b)
        A = _profile_A_from_mean_block(S1, B, sigma2, lambdas[0], J, d1, eps=eps)

        if J > 1:
            at = lambdas[1:, None] * b[None, :] + sigma2
            sdiag = np.einsum("ki,tkl,li->ti", Q, St_tail, Q, optimize=True)
            phi = float(np.sum(np.log(np.maximum(at, eps)) + sdiag / np.maximum(at, eps)))
        else:
            phi = 0.0
        obj_hist.append(float(phi))
        if verbose:
            _log(
                f"[Alg1] outer iter {outer_iter}: objective={phi:.6e}, sigma2={sigma2:.6g}, "
                f"max(b)={(b.max() if b.size else 0):.6g}"
            )
        if len(obj_hist) >= 2:
            rel = abs(obj_hist[-2] - obj_hist[-1]) / (abs(obj_hist[-2]) + 1e-12)
            if rel < tol_outer:
                break

    idx_pos = np.where(b > 1e-12)[0][:d2]
    W2 = Q[:, idx_pos] * np.sqrt(b[idx_pos])[None, :] if len(idx_pos) > 0 else np.zeros((p, 0))
    wa, Ua = eigh((A + A.T) / 2.0)
    idxa = np.argsort(wa)[::-1]
    wa = np.maximum(wa[idxa][:d1], 0.0)
    Ua = Ua[:, idxa][:, :d1]
    W1 = Ua * np.sqrt(wa)[None, :] if d1 > 0 else np.zeros((p, 0))

    return {
        "A": A, "B": B, "sigma2": float(sigma2),
        "Q": Q, "b": b, "W1": W1, "W2": W2,
        "objective_history": obj_hist, "lambdas": lambdas, "UT": UT,
    }


# -----------------------------
# Algorithm 2 / 3 (Compound Symmetry Sigma(tau^2))
# -----------------------------

def fit_hppca_alg2_cs(
    Y: Any,
    d1: int,
    d2: int,
    survey_times: np.ndarray | None = None,
    subject_id_col: str | int = "subject_id",
    visit_time_col: str | int | None = "visit_time",
    feature_cols: list[str] | list[int] | None = None,
    tau2_init: float = 0.2,
    max_outer: int = 30,
    tol_tau: float = 1e-5,
    rng: np.random.Generator | None = None,
):
    """
    MLE when Sigma(tau^2) = (1-tau^2) I + tau^2 11^T (compound symmetry).
    """
    if rng is None:
        rng = np.random.default_rng(0)

    if isinstance(Y, np.ndarray) and Y.ndim == 3:
        _, J, p = Y.shape
        is_long = False
    else:
        records = _prepare_long_records(
            Y,
            survey_times=survey_times,
            subject_id_col=subject_id_col,
            visit_time_col=visit_time_col,
            feature_cols=feature_cols,
        )
        J = records["J"]
        p = records["p"]
        is_long = True

    UT = _householder_ut(J, rng=rng)
    if is_long:
        _, S1, Sc = _rotate_time_and_covs_from_long(records, UT)
    else:
        _, S1, Sc = _rotate_time_and_covs(Y, UT)

    wc, Q = eigh((Sc + Sc.T) / 2.0)
    idx = np.argsort(wc)[::-1]
    wc = wc[idx]
    Q = Q[:, idx]

    sigma2_hat = float(np.mean(wc[d2:])) if d2 < p else float(np.mean(wc))
    sigma2_hat = max(sigma2_hat, 1e-9)

    def profile_A(B, sigma2, tau2):
        lam1 = 1.0 + (J - 1) * tau2
        return _profile_A_from_mean_block(S1, B, sigma2, lam1, J, d1)

    def profiled_mean_objective(tau2):
        tau2 = float(np.clip(tau2, 0.0, 1.0 - 1e-6))
        lam1 = 1.0 + (J - 1) * tau2
        lamc = max(1.0 - tau2, 1e-6)
        b = np.zeros(p)
        r = min(d2, p)
        if r > 0:
            b[:r] = np.maximum((wc[:r] - sigma2_hat) / lamc, 0.0)
        B = _build_B_from_qb(Q, b)
        A = profile_A(B, sigma2_hat, tau2)
        val = _mean_direction_objective(S1, A, B, sigma2_hat, lam1, J)
        return float(val), b, A, B

    tau_lo = 0.0
    tau_hi = 1.0 - 1e-6
    grid_size = max(17, min(65, max_outer + 11))
    refine_rounds = max(2, min(max_outer, 5))
    tau_grid = np.linspace(tau_lo, tau_hi, grid_size)
    tau_seed = float(np.clip(tau2_init, tau_lo, tau_hi))
    tau_grid = np.unique(np.r_[tau_grid, tau_seed])

    best_val = np.inf
    tau2 = tau_seed
    b = np.zeros(p)
    A = np.zeros((p, p))
    B = np.zeros((p, p))
    for _ in range(refine_rounds):
        vals = []
        for tau in tau_grid:
            val, b_tau, A_tau, B_tau = profiled_mean_objective(float(tau))
            vals.append((val, float(tau), b_tau, A_tau, B_tau))
            if val < best_val:
                best_val = val
                tau2 = float(tau)
                b = b_tau
                A = A_tau
                B = B_tau

        best_idx = int(np.argmin([item[0] for item in vals]))
        if best_idx == 0:
            left = float(tau_grid[0])
            right = float(tau_grid[min(1, tau_grid.size - 1)])
        elif best_idx == tau_grid.size - 1:
            left = float(tau_grid[max(tau_grid.size - 2, 0)])
            right = float(tau_grid[-1])
        else:
            left = float(tau_grid[best_idx - 1])
            right = float(tau_grid[best_idx + 1])

        if right - left <= tol_tau * (1.0 + abs(tau2)):
            break
        tau_grid = np.linspace(left, right, grid_size)

    sigma2 = sigma2_hat

    idx_pos = np.where(b > 1e-12)[0][:d2]
    W2 = Q[:, idx_pos] * np.sqrt(b[idx_pos])[None, :] if len(idx_pos) > 0 else np.zeros((p, 0))
    wa, Ua = eigh((A + A.T) / 2.0)
    idxa = np.argsort(wa)[::-1]
    wa = np.maximum(wa[idxa][:d1], 0.0)
    Ua = Ua[:, idxa][:, :d1]
    W1 = Ua * np.sqrt(wa)[None, :] if d1 > 0 else np.zeros((p, 0))

    return {
        "A": A, "B": B, "sigma2": float(sigma2),
        "W1": W1, "W2": W2,
        "tau2": float(tau2), "Q": Q, "b": b,
        "S1": S1, "Sc": Sc, "UT": UT,
    }


# -----------------------------
# Simulation helpers
# -----------------------------

def simulate_hppca(
    n: int, J: int, p: int, d1: int, d2: int, sigma2: float, Sigma: np.ndarray,
    rng: np.random.Generator | None = None,
):
    """
    Simulate Y[n, J, p] from:
      Y_ij = W1 Z1_i + W2 Z2_ij + sigma eps_ij
    """
    if rng is None:
        rng = np.random.default_rng(123)
    W1 = rng.standard_normal((p, d1)) / np.sqrt(p) if d1 > 0 else np.zeros((p, 0))
    W2 = rng.standard_normal((p, d2)) / np.sqrt(p) if d2 > 0 else np.zeros((p, 0))
    Z1 = rng.standard_normal((n, d1)) if d1 > 0 else np.zeros((n, 0))

    if d2 > 0:
        w_S, U_S = eigh((Sigma + Sigma.T) / 2.0)
        w_S = np.maximum(w_S, 1e-12)
        L_S = U_S @ np.diag(np.sqrt(w_S))
        Z2 = np.empty((n, J, d2))
        for r in range(d2):
            Zstd = rng.standard_normal((n, J))
            Z2[:, :, r] = Zstd @ L_S.T
    else:
        Z2 = np.zeros((n, J, 0))

    Y = np.empty((n, J, p))
    for i in range(n):
        for j in range(J):
            mean = np.zeros(p)
            if d1 > 0:
                mean += W1 @ Z1[i]
            if d2 > 0:
                mean += W2 @ Z2[i, j]
            noise = rng.standard_normal(p) * np.sqrt(sigma2)
            Y[i, j] = mean + noise

    Y -= Y.mean(axis=(0, 1), keepdims=True)
    A_true = W1 @ W1.T
    B_true = W2 @ W2.T
    return Y, {"W1": W1, "W2": W2, "A": A_true, "B": B_true, "sigma2": sigma2, "Sigma": Sigma}


def cs_cov(J: int, tau2: float) -> np.ndarray:
    """Compound symmetry Sigma(tau^2) = (1-tau^2) I + tau^2 11^T."""
    return (1.0 - tau2) * np.eye(J) + tau2 * np.ones((J, J))
