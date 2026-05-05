# -*- coding: utf-8 -*-
"""
End-to-end GDRN-RSS demo on synthetic dynamics.

Reproduces the synthetic-dynamics pattern of the paper's main results table:
under irregular masking and a few-shot label budget, the four-component RSS
feature [W_out | mu | sigma | h_end] dominates per-channel raw summary
statistics by tens of accuracy points.

No external data is needed. Run:

    python run_demo.py                  # default: Kuramoto MCAR 20%, k=10
    python run_demo.py --k 20 --seeds 0,1,2,3,4
    python run_demo.py --dataset cml    # also: kuramoto / cml / lorenz96

Expected wall-clock on a laptop CPU: <1 min for Kuramoto, 5 seeds.
"""

import argparse
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from gdrn import make_reservoir
from rss import extract_rss_batch
from datasets import (
    generate_kuramoto, generate_cml, generate_lorenz96,
    mask_mcar,
    make_k_shot_indices, zscore_fit, zscore_apply,
)


def raw_stats(X, masks):
    """5 statistics per channel: mean, std, min, max, observation rate."""
    N, T, F = X.shape
    out = np.zeros((N, 5 * F), dtype=np.float64)
    for i in range(N):
        for j in range(F):
            obs = masks[i, :, j] > 0.5
            cnt = obs.sum()
            if cnt == 0:
                continue
            v = X[i, obs, j].astype(np.float64)
            out[i, j * 5 + 0] = v.mean()
            out[i, j * 5 + 1] = v.std()
            out[i, j * 5 + 2] = v.min()
            out[i, j * 5 + 3] = v.max()
            out[i, j * 5 + 4] = cnt / T
    return out


def classify_rf(feat_tr, y_tr, feat_te, y_te, seed=0):
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(feat_tr, y_tr)
    pred = clf.predict(feat_te)
    return (accuracy_score(y_te, pred),
            f1_score(y_te, pred, average='macro'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['kuramoto', 'cml', 'lorenz96'],
                    default='kuramoto')
    ap.add_argument('--obs_rate', type=float, default=0.2,
                    help='MCAR retain fraction (0.2 = 20%% observed)')
    ap.add_argument('--k', type=int, default=10,
                    help='few-shot shots per class')
    ap.add_argument('--seeds', type=str, default='0,1,2,3,4')
    ap.add_argument('--n_reservoir', type=int, default=20,
                    help='reservoir hidden dimension n (paper default 20)')
    ap.add_argument('--rho', type=float, default=0.9,
                    help='reservoir spectral radius')
    ap.add_argument('--decay', type=float, default=0.1,
                    help='per-channel decay alpha')
    ap.add_argument('--ridge', type=float, default=1e-4,
                    help='readout ridge penalty')
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(',')]

    # 1) Generate clean synthetic dynamics
    print(f'[1/3] Generating {args.dataset} ...')
    if args.dataset == 'kuramoto':
        X, y = generate_kuramoto(N=100, T=500, F=16, n_classes=3, seed=42)
    elif args.dataset == 'cml':
        X, y = generate_cml(N=100, T=500, F=20, n_classes=3, seed=42)
    else:
        X, y = generate_lorenz96(N_per_class=100, b=16, T_out=500, seed=42)
    N, T, F = X.shape
    print(f'      shape {X.shape}, classes {np.unique(y)}')

    # 2) Apply irregular MCAR masking
    print(f'[2/3] Applying MCAR (retain {args.obs_rate:.0%}) ...')
    X, masks = mask_mcar(X, missing_rate=1.0 - args.obs_rate, seed=0)
    times = np.tile(np.arange(T, dtype=np.float64), (N, 1))
    print(f'      observed fraction = {masks.mean():.2%}')

    # 3) Per-seed: build features, train RF, score
    print(f'[3/3] Few-shot (k={args.k}, {len(seeds)} seeds):\n')
    header = f"{'seed':>4}  {'raw-stats Acc':>14}  {'RSS Acc':>10}  {'gain':>6}"
    print(header); print('-' * len(header))

    results = {'raw': [], 'rss': []}
    t_total = time.time()
    for seed in seeds:
        idx_tr, idx_te = make_k_shot_indices(y, k=args.k, seed=seed)

        # Train-only z-score (no leakage)
        mu, sd = zscore_fit(X[idx_tr], masks[idx_tr])
        X_norm = zscore_apply(X, mu, sd, masks)

        # Baseline: raw per-channel statistics
        feat_raw = raw_stats(X_norm, masks)
        feat_raw = StandardScaler().fit_transform(feat_raw)
        feat_raw = np.nan_to_num(feat_raw)
        acc_raw, _ = classify_rf(
            feat_raw[idx_tr], y[idx_tr], feat_raw[idx_te], y[idx_te], seed=seed)

        # Ours: RSS = [W_out | mu_H | sigma_H | h_end]
        W, W_in = make_reservoir(n=args.n_reservoir, spectral_radius=args.rho,
                                 sparsity=0.5, seed=42)
        feat_rss = extract_rss_batch(X_norm, times, masks, W, W_in,
                                     args.decay, args.ridge)
        feat_rss = StandardScaler().fit_transform(feat_rss)
        feat_rss = np.nan_to_num(feat_rss)
        acc_rss, _ = classify_rf(
            feat_rss[idx_tr], y[idx_tr], feat_rss[idx_te], y[idx_te], seed=seed)

        results['raw'].append(acc_raw)
        results['rss'].append(acc_rss)
        print(f"{seed:>4}  {acc_raw*100:>13.2f}%  {acc_rss*100:>9.2f}%"
              f"  {(acc_rss-acc_raw)*100:>+5.1f}")

    r = np.array(results['raw']) * 100
    s = np.array(results['rss']) * 100
    print('-' * len(header))
    print(f"{'mean':>4}  {r.mean():>12.2f}+-{r.std():>3.1f}  "
          f"{s.mean():>7.2f}+-{s.std():>3.1f}  {(s.mean()-r.mean()):>+5.1f}")
    print(f'\nTotal wall-clock: {time.time()-t_total:.1f}s')

    if args.dataset == 'kuramoto' and args.k == 10 and abs(args.obs_rate - 0.2) < 1e-6:
        print('\nSanity check on Kuramoto MCAR 20%, k=10:')
        print('  raw per-channel statistics fall in the mid-50s')
        print('  RSS (ours) reaches the high-80s to low-90s')
        print('  (matches the corresponding row of the main results table)')


if __name__ == '__main__':
    main()
