# SDFormer

**SDFormer: Multi-Granularity Time Series Modeling for Computing Power Network Supply-Demand Forecasting**

---

## Overview

Supply-demand series in computing power networks combine **regular low-frequency periodic components** with **irregular high-frequency burst components**, fully aliased in the time domain. This poses two challenges for classical time-series forecasting: a single fixed window cannot serve both regimes, and mean/variance-based instance normalization is highly sensitive to sparse extreme values.

SDFormer tackles this from a behavior-decoupling perspective with four core components:

| Component | Function |
|-----------|----------|
| **Multi-Granularity Independent Encoding** | Three parallel Transformer encoders model high-frequency abrupt changes, transition bands, and low-frequency trends separately, eliminating cross-granularity gradient interference. |
| **ATSR** (Adaptive Temporal Salience Recalibration) | Introduces learnable point-wise weights before local window projection, automatically amplifying irregular burst signals so peaks are not diluted. |
| **AGF** (Adaptive Granularity Gating Fusion) | Dynamically generates per-granularity contribution weights from global context; leans fine-grained during high-burst periods and coarse-grained during stable periods. |
| **Anomaly-Aware Normalization** | Uses kurtosis and tail probability as features to adaptively interpolate convexly between median/MAD and mean/std, protecting the feature space from the input side. |

---

## Main Results

Compared with PatchTST, iTransformer, Autoformer, and FEDformer on five benchmark datasets (ETTh1, ETTh2, ETTm1, ETTm2, Weather):

- ETTh1 96-step MSE: **7.6% lower** than PatchTST, **7.5% lower** than iTransformer
- ETTh1 192-step MSE: **14.9% lower** than PatchTST
- Best across all prediction horizons on ETTh2
- Competitive on ETTm / Weather medium- and long-term forecasts (≥ 336 steps)

---

## Requirements

```
Python   3.12.3
PyTorch  2.5.1
CUDA     12.1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Data Preparation

Download the datasets from [Time-Series-Library](https://github.com/thuml/Time-Series-Library) and place them under `./dataset/`:

```
dataset/
├── ETT-small/
│   ├── ETTh1.csv
│   ├── ETTh2.csv
│   ├── ETTm1.csv
│   └── ETTm2.csv
└── weather/
    └── weather.csv
```

---

## Quick Start

**Training**

```bash
python run.py \
  --model SDFormer \
  --data ETTh1 \
  --seq_len 512 \
  --pred_len 96 \
  --patch_sizes 16 32 64 \
  --d_model 512 \
  --n_heads 8 \
  --e_layers 1 \
  --d_ff 2048 \
  --dropout 0.1 \
  --batch_size 32 \
  --learning_rate 1e-4 \
  --train_epochs 20
```

**Reproducing paper results**

```bash
bash scripts/run_all.sh
```

---

## Project Structure

```
SDFormer/
├── models/
│   └── SDFormer.py          # Main model
├── layers/
│   ├── anomaly_norm.py      # Anomaly-aware normalization
│   ├── atsr.py              # Adaptive Temporal Salience Recalibration
│   └── agf.py               # Adaptive Granularity Gating Fusion
├── data_provider/           # Data loading
├── exp/                     # Training / evaluation pipeline
├── scripts/                 # Reproduction scripts
├── dataset/                 # Datasets (download separately)
├── run.py
└── requirements.txt
```

---

## Citation

If this work is helpful, please cite:

```bibtex
@article{sdformer2026,
  title     = {SDFormer: Multi-Granularity Time Series Modeling for Computing Power Network Supply-Demand Forecasting},
  year      = {2026}
}
```

---

## Acknowledgements

This project is built on the [Time-Series-Library](https://github.com/thuml/Time-Series-Library) framework. We thank the open-source work of PatchTST and iTransformer.

## Repository

https://github.com/Wingspeg/sdformer

## CPN Scheduling Simulation

The `cpn_simulation/` directory provides an M-site × K-resource supply-demand simulator for computing power networks, end-to-end running the system model (supply / demand / gap / cost C) from Section 3.1 of the paper.

```
cpn_simulation/
├── cpn_simulator.py    # Multi-site demand simulator (periodic + i.i.d. heavy-tailed burst)
├── cpn_runner.py       # Training + scheduling evaluation runner (SDFormer/PatchTST/iTransformer/WPMixer + Reactive/MA/Perfect)
└── results.json        # 4-horizon × 4-model evaluation results
```

**Generate data**

```bash
python cpn_simulation/cpn_simulator.py --out_dir dataset/CPN-sim
```

**Run simulation** (a few minutes on CPU)

```bash
python cpn_simulation/cpn_runner.py \
  --horizons 24 48 96 168 \
  --epochs 8 \
  --models SDFormer PatchTST iTransformer WPMixer \
  --no_cuda
```

Simulation results overwrite `cpn_simulation/results.json`.

---

## License

MIT — see [LICENSE](LICENSE).
