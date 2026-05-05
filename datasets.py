# -*- coding: utf-8 -*-
"""
Synthetic generators, irregular masking, and few-shot utilities.

This file provides the minimum needed to reproduce the synthetic-dynamics
experiments of the paper without external data:

  * generate_kuramoto, generate_cml, generate_lorenz96 : three families of
    coupled-oscillator / chaotic dynamics parameterized so that the class
    label is determined by the dynamical regime.
  * mask_mcar, mask_burst, mask_event_triggered : the three artificial
    masking mechanisms evaluated in the paper's main results table.
  * make_k_shot_indices, zscore_fit, zscore_apply : few-shot split +
    mask-aware z-score normalization.
"""

import numpy as np
from numba import njit


def generate_kuramoto(N=100, T=500, F=16, n_classes=3, dt=0.05,
                      transient=100, seed=42):
    """Kuramoto coupled oscillators on a ring.

        d theta_i / dt = omega_i + (K / F) * sum_j sin(theta_j - theta_i)

    Classes differ by coupling strength K in {0.5, 2.0, 5.0} and natural-
    frequency spread in {2.0, 1.0, 0.5}, producing qualitatively different
    synchronization regimes (incoherent / partially synced / fully synced).

    Returns X: (N*n_classes, T, F), y: (N*n_classes,).
    """
    @njit(cache=True)
    def _kura_one(T, F, K, dt, transient, theta0, omega):
        theta = theta0.copy()
        out = np.zeros((T, F))
        for _ in range(transient):
            c = np.zeros(F)
            for i in range(F):
                s = 0.0
                for j in range(F):
                    s += np.sin(theta[j] - theta[i])
                c[i] = s * K / F
            for i in range(F):
                theta[i] += dt * (omega[i] + c[i])
        for t in range(T):
            for i in range(F):
                out[t, i] = np.sin(theta[i])
            c = np.zeros(F)
            for i in range(F):
                s = 0.0
                for j in range(F):
                    s += np.sin(theta[j] - theta[i])
                c[i] = s * K / F
            for i in range(F):
                theta[i] += dt * (omega[i] + c[i])
        return out

    rng = np.random.RandomState(seed)
    total = N * n_classes
    X = np.zeros((total, T, F), dtype=np.float32)
    y = np.zeros(total, dtype=np.int64)

    K_values = [0.5, 2.0, 5.0][:n_classes]
    omega_spreads = [2.0, 1.0, 0.5][:n_classes]

    # Warmup
    _kura_one(5, F, 1.0, dt, 2,
              rng.uniform(0, 2 * np.pi, F), rng.normal(0, 1, F))

    for c_idx in range(n_classes):
        K = K_values[c_idx]
        spread = omega_spreads[c_idx]
        for n_i in range(N):
            idx = c_idx * N + n_i
            y[idx] = c_idx
            theta0 = rng.uniform(0, 2 * np.pi, F)
            omega = rng.normal(0, spread, F)
            X[idx] = _kura_one(T, F, K, dt, transient,
                               theta0, omega).astype(np.float32)

    perm = rng.permutation(total)
    return X[perm], y[perm]


def generate_cml(N=100, T=500, F=20, n_classes=3, transient=200, seed=42):
    """Coupled Map Lattice: coupled logistic maps on a ring.

        x_i(t+1) = (1-c) f(x_i) + c/2 (f(x_{i-1}) + f(x_{i+1})),
        f(x) = r x (1 - x)

    Classes differ by (r, c) regime (period / edge-of-chaos / chaos).
    """
    @njit(cache=True)
    def _cml_one(T, F, coupling, r, transient, x0):
        x = x0.copy()
        out = np.zeros((T, F))
        for _ in range(transient):
            x_new = np.zeros(F)
            for i in range(F):
                fi = r * x[i] * (1 - x[i])
                fl = r * x[(i - 1) % F] * (1 - x[(i - 1) % F])
                fr = r * x[(i + 1) % F] * (1 - x[(i + 1) % F])
                x_new[i] = (1 - coupling) * fi + coupling / 2 * (fl + fr)
            for i in range(F):
                x[i] = min(max(x_new[i], 0.001), 0.999)
        for t in range(T):
            for i in range(F):
                out[t, i] = x[i]
            x_new = np.zeros(F)
            for i in range(F):
                fi = r * x[i] * (1 - x[i])
                fl = r * x[(i - 1) % F] * (1 - x[(i - 1) % F])
                fr = r * x[(i + 1) % F] * (1 - x[(i + 1) % F])
                x_new[i] = (1 - coupling) * fi + coupling / 2 * (fl + fr)
            for i in range(F):
                x[i] = min(max(x_new[i], 0.001), 0.999)
        return out

    rng = np.random.RandomState(seed)
    total = N * n_classes
    class_params = [(3.5, 0.1), (3.8, 0.3), (3.9, 0.7),
                    (3.56, 0.5), (3.7, 0.05)][:n_classes]
    X = np.zeros((total, T, F), dtype=np.float32)
    y = np.zeros(total, dtype=np.int64)

    _cml_one(5, F, 0.3, 3.8, 2, rng.rand(F) * 0.5 + 0.25)  # warmup

    for c_idx in range(n_classes):
        r_c, coupling_c = class_params[c_idx]
        for n_i in range(N):
            idx = c_idx * N + n_i
            y[idx] = c_idx
            x0 = rng.rand(F) * 0.5 + 0.25
            X[idx] = _cml_one(T, F, coupling_c, r_c, transient,
                              x0).astype(np.float32)

    perm = rng.permutation(total)
    return X[perm], y[perm]


def generate_lorenz96(N_per_class=100, b=16, T_horizon=20.0, dt=0.01,
                      forcing_list=(4, 6, 8), noise_std=0.5, T_out=500,
                      seed=42):
    """Lorenz-96 chaotic system.

        dx_i / dt = (x_{i+1} - x_{i-2}) x_{i-1} - x_i + F

    Classes are defined by forcing level F.
    """
    @njit(cache=True)
    def _rk4(x, F_val, dt, noise):
        bb = x.shape[0]

        def deriv(xin, Fv):
            d = np.zeros(bb)
            for i in range(bb):
                ip1 = (i + 1) % bb
                im1 = (i - 1) % bb
                im2 = (i - 2) % bb
                d[i] = (xin[ip1] - xin[im2]) * xin[im1] - xin[i] + Fv
            return d

        k1 = deriv(x, F_val)
        k2 = deriv(x + 0.5 * dt * k1, F_val)
        k3 = deriv(x + 0.5 * dt * k2, F_val)
        k4 = deriv(x + dt * k3, F_val)
        return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4) + noise

    @njit(cache=True)
    def _simulate(bb, F_val, n_steps, dt, noise_std, x0, noise):
        traj = np.zeros((n_steps, bb))
        x = x0.copy()
        for t in range(n_steps):
            traj[t] = x
            x = _rk4(x, F_val, dt, noise[t] * noise_std)
        return traj

    rng = np.random.RandomState(seed)
    n_classes = len(forcing_list)
    n_steps = int(T_horizon / dt)
    total = N_per_class * n_classes

    _simulate(b, 8.0, 10, dt, 0.0,
              rng.randn(b).astype(np.float64),
              rng.randn(10, b).astype(np.float64))

    stride = max(1, n_steps // T_out)
    T_actual = min(n_steps // stride, T_out)
    X = np.zeros((total, T_actual, b), dtype=np.float32)
    y = np.zeros(total, dtype=np.int64)

    for c_idx, F_val in enumerate(forcing_list):
        for n_i in range(N_per_class):
            idx = c_idx * N_per_class + n_i
            y[idx] = c_idx
            x0 = rng.randn(b).astype(np.float64)
            noise = rng.randn(n_steps, b).astype(np.float64)
            traj = _simulate(b, float(F_val), n_steps, dt, noise_std, x0, noise)
            for t in range(T_actual):
                X[idx, t] = traj[t * stride].astype(np.float32)

    perm = rng.permutation(total)
    return X[perm], y[perm]



def mask_mcar(X, missing_rate=0.5, seed=42):
    """MCAR: each (sequence, time, channel) entry dropped iid."""
    rng = np.random.RandomState(seed)
    mask = (rng.rand(*X.shape) > missing_rate).astype(np.float32)
    return X * mask, mask


def mask_burst(X, burst_prob=0.02, burst_len_mean=20, seed=42):
    """Burst dropouts: per-channel outages of geometric length."""
    rng = np.random.RandomState(seed)
    N, T, F = X.shape
    mask = np.ones((N, T, F), dtype=np.float32)
    for i in range(N):
        for j in range(F):
            t = 0
            while t < T:
                if rng.rand() < burst_prob:
                    gap = rng.geometric(1.0 / burst_len_mean)
                    mask[i, t:t + gap, j] = 0
                    t += gap
                else:
                    t += 1
    return X * mask, mask


def mask_event_triggered(X, delta_ratio=0.1, seed=42):
    """Send-on-delta: a channel is observed only when its value changes
    by more than delta_ratio * range since the last report."""
    N, T, F = X.shape
    mask = np.zeros((N, T, F), dtype=np.float32)
    for j in range(F):
        vr = np.max(X[:, :, j]) - np.min(X[:, :, j])
        delta = delta_ratio * vr if vr > 0 else 0.01
        for i in range(N):
            last = X[i, 0, j]
            mask[i, 0, j] = 1
            for t in range(1, T):
                if abs(X[i, t, j] - last) > delta:
                    mask[i, t, j] = 1
                    last = X[i, t, j]
    return X * mask, mask



def make_k_shot_indices(y, k=10, seed=42):
    """k samples per class for train, remainder for test."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    tr, te = [], []
    for c in classes:
        idx_c = np.where(y == c)[0]
        rng.shuffle(idx_c)
        k_take = max(1, min(k, len(idx_c) - 1))
        tr.extend(idx_c[:k_take])
        te.extend(idx_c[k_take:])
    return np.array(tr), np.array(te)


def zscore_fit(X, mask=None):
    """Per-channel mean/std from observed entries only."""
    N, T, F = X.shape
    Xr = X.reshape(-1, F)
    if mask is not None:
        Mr = mask.reshape(-1, F).astype(bool)
    else:
        Mr = (Xr != 0)

    means = np.zeros(F, dtype=np.float32)
    stds = np.ones(F, dtype=np.float32)
    for j in range(F):
        obs = Xr[Mr[:, j], j]
        if obs.size > 0:
            s = obs.std()
            means[j] = obs.mean()
            stds[j] = s if s > 1e-8 else 1.0
    return means, stds


def zscore_apply(X, means, stds, mask=None):
    """Apply z-score while preserving zeros for missing entries."""
    X_norm = (X - means) / stds
    if mask is not None:
        X_norm = X_norm * mask
    else:
        X_norm = np.where(X != 0, X_norm, 0)
    return X_norm.astype(np.float32)
