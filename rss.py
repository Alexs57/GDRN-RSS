# -*- coding: utf-8 -*-
"""
RSS: Reservoir State Statistics (paper Sec. 2.3).

Given the GDRN hidden-state trajectory H : (T, b, n), RSS concatenates four
per-channel views for each channel j:

    W_out^j        in R^n   : per-channel ridge readout (paper Sec. 2.2)
    mu_j           in R^n   : trajectory centroid
    sigma_j        in R^n   : trajectory standard deviation
    h^j_{end}      in R^n   : reservoir state at j's last observation

The per-sequence feature is

    phi_RSS = [ W_out^1 ; mu_1 ; sigma_1 ; h_end^1 ;
                ... ;
                W_out^b ; mu_b ; sigma_b ; h_end^b ]   in R^{4 b n}.

mu_j, sigma_j, h^j_{end} are bounded (tanh forces |h| <= 1) and well-defined
for any T_j >= 1, which is what makes RSS stable in the sparse-observation
regime where W_out^j alone is confined to the row space of the observed
states (paper Lemma 2.1).

All statistics are computed over channel j's observed events
I_j = { i : mask[i, j] = 1 }, not over the global timeline T:
    mu_j     = (1 / T_j) * sum_{i in I_j} h_i^j
    sigma_j  = sqrt( (1 / T_j) * sum_{i in I_j} (h_i^j - mu_j)^2 )   # population
    h_end^j  = h^j_{ max(I_j) }
where T_j = |I_j|. Channels with T_j = 0 contribute zeros (the never-observed
channel default discussed in the paper's limitations).
"""

import numpy as np
from numba import njit, prange
from gdrn import _gdrn_update, _readout_per_channel


@njit(cache=True)
def _extract_rss_one(X, times, mask, W, W_in, decay, ridge):
    """RSS feature for a single sequence, shape (4*b*n,)."""
    H = _gdrn_update(X, times, mask, W, W_in, decay)
    T, b, n = H.shape

    W_out = _readout_per_channel(H, X, mask, ridge, n)
    mu_H = np.zeros(b * n)
    sigma_H = np.zeros(b * n)
    h_end = np.zeros(b * n)

    for j in range(b):
        cnt = 0
        last_t = 0
        for t in range(T):
            if mask[t, j] > 0.5:
                cnt += 1
                last_t = t
        if cnt < 1:
            continue

        Sj = np.empty((cnt, n))
        idx = 0
        for t in range(T):
            if mask[t, j] > 0.5:
                Sj[idx] = H[t, j]
                idx += 1

        mu = Sj.sum(axis=0) / cnt
        var = ((Sj - mu) ** 2).sum(axis=0) / cnt
        mu_H[j * n : (j + 1) * n] = mu
        sigma_H[j * n : (j + 1) * n] = np.sqrt(var)
        h_end[j * n : (j + 1) * n] = H[last_t, j]

    out = np.empty(4 * b * n)
    out[          : b * n    ] = W_out
    out[b * n     : 2 * b * n] = mu_H
    out[2 * b * n : 3 * b * n] = sigma_H
    out[3 * b * n :          ] = h_end
    return out


@njit(parallel=True, cache=True)
def _extract_rss_batch(X_all, times_all, mask_all, W, W_in, decay, ridge):
    N = X_all.shape[0]
    b = X_all.shape[2]
    n = W.shape[0]
    out = np.zeros((N, 4 * b * n))
    for i in prange(N):
        out[i] = _extract_rss_one(
            X_all[i], times_all[i], mask_all[i], W, W_in, decay, ridge)
    return out


def extract_rss_batch(X_all, times_all, mask_all, W, W_in, decay, ridge=1e-4):
    """Compute phi_RSS for a batch of sequences.

    Args:
        X_all     : (N, T, b) observations (missing entries ignored via mask)
        times_all : (N, T) or (T,) timestamps
        mask_all  : (N, T, b) binary observation mask
        W, W_in   : reservoir weights from gdrn.make_reservoir
        decay     : alpha in exp(-alpha * dt)
        ridge     : readout ridge penalty (paper default 1e-4)

    Returns:
        phi : (N, 4 b n) RSS feature matrix, ready for any downstream classifier.
    """
    N, T, b = X_all.shape
    if times_all.ndim == 1:
        times_all = np.tile(times_all, (N, 1))
    return _extract_rss_batch(
        np.ascontiguousarray(X_all,     dtype=np.float64),
        np.ascontiguousarray(times_all, dtype=np.float64),
        np.ascontiguousarray(mask_all,  dtype=np.float64),
        np.ascontiguousarray(W,         dtype=np.float64),
        np.ascontiguousarray(W_in,      dtype=np.float64),
        decay, ridge)


def extract_ablation_batch(X_all, times_all, mask_all, W, W_in, decay,
                           ridge=1e-4,
                           use_wout=True, use_mu=True, use_sigma=True,
                           use_hend=True):
    """Ablation variant: concatenate any non-empty subset of the four RSS
    components (W_out, mu, sigma, h_end). Reproduces the per-component
    breakdown reported in the paper's ablation section.
    """
    assert use_wout or use_mu or use_sigma or use_hend, \
        "select at least one component"
    full = extract_rss_batch(X_all, times_all, mask_all, W, W_in, decay, ridge)
    bn = full.shape[1] // 4

    parts = []
    if use_wout:  parts.append(full[:, 0 * bn : 1 * bn])
    if use_mu:    parts.append(full[:, 1 * bn : 2 * bn])
    if use_sigma: parts.append(full[:, 2 * bn : 3 * bn])
    if use_hend:  parts.append(full[:, 3 * bn : 4 * bn])
    return np.hstack(parts)
