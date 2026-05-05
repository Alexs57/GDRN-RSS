# GDRN-RSS

Reference implementation for the paper **"Beyond the Readout: Reservoir
State Statistics for Model-Space Learning under Sparse Observations"**.


## Contents

| File              | Paper reference                                                |
| ----------------- | -------------------------------------------------------------- |
| `gdrn.py`         | GDRN forward pass and per-channel ridge readout (Sec. 2.1–2.2) |
| `rss.py`          | Four-component RSS feature `[W_out, mu, sigma, h_end]` (Sec. 2.3) |
| `datasets.py`     | Synthetic dynamics + three irregular-masking schemes           |
| `run_demo.py`     | End-to-end entry point (Kuramoto / CML / Lorenz-96)            |
| `requirements.txt`| Pinned dependencies                                            |

## Quick start

```bash
pip install -r requirements.txt
python run_demo.py                              # Kuramoto MCAR 20%, k=10
python run_demo.py --dataset cml --k 20
python run_demo.py --dataset lorenz96 --obs_rate 0.5
```

On Kuramoto MCAR 20%, k=10 (default), raw per-channel statistics score
in the mid-50s and the RSS feature in the high-80s to low-90s.

Wall-clock is under one minute on a laptop CPU. No GPU, no external data.

## Hyperparameters

A **single fixed configuration** is used across all experiments in the
paper (no per-dataset tuning):

| Name             | Symbol  | Default | `run_demo.py` flag |
| ---------------- | ------- | ------- | ------------------ |
| Reservoir size   | `n`     | 20      | `--n_reservoir`    |
| Spectral radius  | `rho`   | 0.9     | `--rho`            |
| Decay            | `lambda`| 0.1     | `--decay`          |
| Readout ridge    | `gamma` | 1e-4    | `--ridge`          |
| Sparsity         | —       | 0.5     | (in `make_reservoir`) |
| Shots per class  | `k`     | 10      | `--k`              |


## Details

### Real datasets

| Dataset             | Domain                    | Original source                                                                     |
| ------------------- | ------------------------- | ----------------------------------------------------------------------------------- |
| **P12**             | ICU mortality             | PhysioNet Challenge 2012: <https://physionet.org/content/challenge-2012/1.0.0/>     |
| **P19**             | ICU sepsis                | PhysioNet Challenge 2019: <https://physionet.org/content/challenge-2019/1.0.0/>     |
| **PAM**             | Wearable activity         | UCI PAMAP2: <https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring> |
| **Person Activity** | UCI RFID activity         | <https://archive.ics.uci.edu/dataset/196/localization+data+for+person+activity>     |
| **TEP**             | Tennessee Eastman process | <https://github.com/camaramm/tennessee-eastman-profBraatz>                          |
| **Volcano CAU**     | Seismic event waveforms   | Multi-station Chilean volcano dataset: <https://zenodo.org/records/15384923>        |

For **P12, P19, and PAM** the preprocessed release published with the
Raindrop paper is used, together with the same five official
train/val/test splits. Direct figshare mirrors:

- P12: <https://doi.org/10.6084/m9.figshare.19514341.v1>
- P19: <https://doi.org/10.6084/m9.figshare.19514338.v1>
- PAM: <https://doi.org/10.6084/m9.figshare.19514347.v1>

### Synthetic datasets 

| Dataset    | Generator                    |
| ---------- | ---------------------------- |
| Lorenz-96  | `datasets.generate_lorenz96` |
| Kuramoto   | `datasets.generate_kuramoto` |
| CML        | `datasets.generate_cml`      |

### Baselines

**Deep / gradient-trained baselines:**

| Baseline     | Code                                                    |
| ------------ | ------------------------------------------------------- |
| GRU-D        | <https://github.com/zhiyongc/GRU-D>                     |
| Latent ODE   | <https://github.com/YuliaRubanova/latent_ode>           |
| Raindrop     | <https://github.com/mims-harvard/Raindrop>              |
| mTAND        | <https://github.com/reml-lab/mTAN>                      |
| ATENet       | <https://github.com/shlee-labs/ATENet>                  |
| Warpformer   | <https://github.com/imJiawen/Warpformer>                |
| Hi-Patch     | <https://github.com/qianlima-lab/Hi-Patch>              |
| PrimeNet     | <https://github.com/ranakroychowdhury/PrimeNet>         |
| ViTST        | <https://github.com/Leezekun/ViTST>                     |

**Training-free / non-DL baselines:**

| Baseline     | Features                                                                  | Source                                                |
| ------------ | ------------------------------------------------------------------------- | ----------------------------------------------------- |
| RF-stats     | 5 per-channel stats (mean / std / min / max / obs\_rate) + Random Forest  | `scikit-learn` (`run_demo.raw_stats` implements this) |
| StatsGap     | RF-stats + 2 per-channel time-gap stats (mean / std of inter-event gaps)  | `scikit-learn` (trivial extension of RF-stats)        |
| catch22      | 22 canonical time-series features per channel                             | `pycatch22`: `pip install pycatch22`                  |
| MiniROCKET   | random convolutional kernels + ridge                                      | `aeon-toolkit`: `pip install aeon`                    |
| MultiROCKET  | extended MiniROCKET with additional pooling operators + ridge             | `aeon-toolkit`: `pip install aeon`                    |



## License

MIT. See `LICENSE`.
