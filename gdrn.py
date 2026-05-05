# -*- coding: utf-8 -*-
"""
GDRN: Global Decay Reservoir Network (paper Sec. 2.1).

Training-free continuous-time reservoir for asynchronous multivariate
irregular time series. The forward pass combines three mechanisms:

    (1) per-channel time decay between events
            h_j <- exp(-lambda (t - t_last(j))) h_j
    (2) cross-channel global pooling
            g_t = (1/b) sum_j h_j
    (3) event-driven masked update
            h_j <- tanh(W g_t + w_in x_{t,j})  iff mask[t, j] = 1

The per-channel ridge readout (paper Sec. 2.2) is also implemented here so
the file is end-to-end usable on its own; the four-component RSS feature
[W_out | mu | sigma | h_end] (paper Sec. 2.3) lives in rss.py.

Inner loops are compiled with numba.
"""

import numpy as np
from numba import njit, prange
from scipy.sparse import random as sparse_random
from scipy.linalg import eigvals


def make_reservoir(n=20, spectral_radius=0.9, sparsity=0.5, input_scale=1.0,
                   seed=42):
    """Random sparse reservoir with target spectral radius rho < 1.

    Returns (W, W_in) with shapes (n, n) and (1, n), float64 contiguous.
    """
    rng = np.random.RandomState(seed)
    W = sparse_random(n, n, density=sparsity, random_state=rng,
                      data_rvs=lambda s: rng.uniform(-1, 1, s)).toarray()
    sr = np.max(np.abs(eigvals(W)))
    if sr > 0:
        W = W * (spectral_radius / sr)
    W_in = rng.uniform(-input_scale, input_scale, (1, n))
    return (np.ascontiguousarray(W, dtype=np.float64),
            np.ascontiguousarray(W_in, dtype=np.float64))


@njit(cache=True)
def _gdrn_update(X, times, mask, W, W_in, decay):
    """Event-driven reservoir update with per-channel time decay.

    X     : (T, b) channel observations (value is ignored where mask==0)
    times : (T,)   timestamps
    mask  : (T, b) binary observation mask
    W     : (n, n) reservoir matrix
    W_in  : (1, n) input-to-reservoir vector
    decay : alpha in exp(-alpha * dt)

    Returns H: (T, b, n) per-channel reservoir state trajectories.
    """
    T, b = X.shape
    n = W.shape[0]
    H = np.zeros((T, b, n))
    h_last = np.zeros((b, n))
    t_last = np.zeros(b)
    w_in = W_in[0]

    for i in range(T):
        m_i = mask[i]
        if m_i.sum() < 0.5:
            continue

        # Decay + cross-channel pooling
        df = np.exp(-decay * (times[i] - t_last))
        g = (df.reshape(-1, 1) * h_last).sum(axis=0) / b
        gW = g @ W

        # Masked selective update: only observed channels step forward
        for j in range(b):
            if m_i[j] > 0.5:
                h_new = np.tanh(gW + X[i, j] * w_in)
                H[i, j] = h_new
                h_last[j] = h_new
                t_last[j] = times[i]

    return H


@njit(cache=True)
def _readout_per_channel(H, teacher, mask, ridge, n):
    """Per-channel ridge readout (paper Sec. 2.2).

    For each channel j independently, solve the input-reconstruction ridge
    using the sample-normalized form:
        W_out^j = (S_j^T S_j / T_j + gamma I)^{-1} (S_j^T y_j / T_j)
    where S_j stacks reservoir states at channel j's observed events and
    y_j the corresponding observed values. The 1/T_j scaling makes the
    effective regularizer scale-invariant in T_j and is equivalent to the
    paper's (S^T S + gamma' I)^{-1} S^T y form with gamma' = gamma * T_j.

    Lemma 2.1 shows this readout becomes rank-deficient when T_j < n; the
    inversion-free RSS components in rss.py complement it in that regime.

    Returns W_out flat: (b * n,).
    """
    T, b, _ = H.shape
    W_out = np.zeros(b * n)

    for j in range(b):
        cnt = 0
        for t in range(T):
            if mask[t, j] > 0.5:
                cnt += 1
        if cnt < 1:
            continue

        S = np.empty((cnt, n))
        y = np.empty(cnt)
        idx = 0
        for t in range(T):
            if mask[t, j] > 0.5:
                S[idx] = H[t, j]
                y[idx] = teacher[t, j]
                idx += 1

        StS = (S.T @ S) / cnt + ridge * np.eye(n)
        Sty = (S.T @ y) / cnt
        W_out[j * n : (j + 1) * n] = np.linalg.solve(StS, Sty)

    return W_out


@njit(parallel=True, cache=True)
def _gdrn_readout_batch(X_all, times_all, mask_all, W, W_in, decay, ridge):
    """Parallel batch readout extraction. Returns (N, b*n)."""
    N = X_all.shape[0]
    n = W.shape[0]
    b = X_all.shape[2]
    out = np.zeros((N, b * n))
    for i in prange(N):
        H = _gdrn_update(X_all[i], times_all[i], mask_all[i], W, W_in, decay)
        out[i] = _readout_per_channel(H, X_all[i], mask_all[i], ridge, n)
    return out


def gdrn_extract_batch(X_all, times_all, mask_all, W, W_in, decay, ridge=1e-4):
    """Fitted-readout-only feature for a batch of sequences (the W-only
    baseline used in the per-component ablation, paper Sec. 2.2).

    Returns (N, b*n) matrix of W_out^j vectors. For the full RSS feature
    phi_RSS = [W_out | mu | sigma | h_end] (paper Sec. 2.3) use
    rss.extract_rss_batch.
    """
    N, T, b = X_all.shape
    if times_all.ndim == 1:
        times_all = np.tile(times_all, (N, 1))
    return _gdrn_readout_batch(
        np.ascontiguousarray(X_all,     dtype=np.float64),
        np.ascontiguousarray(times_all, dtype=np.float64),
        np.ascontiguousarray(mask_all,  dtype=np.float64),
        np.ascontiguousarray(W,         dtype=np.float64),
        np.ascontiguousarray(W_in,      dtype=np.float64),
        decay, ridge)
