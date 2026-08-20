"""
Train and evaluate forecasting models (SDFormer, PatchTST, iTransformer)
on the synthetic CPN demand, and report the scheduling cost C of Eq. 4
under several reservation strategies.

Runs entirely on CPU. Designed to complete within a few minutes per
horizon so it can be used as the simulation experiment that
complements the ETT/Weather results in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# Silence noisy warnings
warnings.filterwarnings("ignore")

# Make sure the project root is on sys.path so we can import the models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from cpn_simulation.cpn_simulator import (  # noqa: E402
    CPNDemandSimulator, Scheduler, SimConfig, evaluate_reservation,
)


# ---------------------------------------------------------------------------
# Windowed dataset (custom, no internal split)
# ---------------------------------------------------------------------------
class CPNSlidingWindow(Dataset):
    """(T, N) demand series -> (samples, L, N) supervised learning windows."""
    def __init__(self, data: np.ndarray, L: int, T_pred: int):
        # data: (T, N), float32
        self.data = torch.from_numpy(data.astype(np.float32))
        self.L = L
        self.T_pred = T_pred
        # We need L history to predict T_pred future. Last index:
        #   x[t : t+L] -> y[t+L : t+L+T_pred]
        # So the last valid start is T - L - T_pred (inclusive).
        self.n = max(0, data.shape[0] - L - T_pred + 1)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.L]
        y = self.data[idx + self.L : idx + self.L + self.T_pred]
        return x, y


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------
def make_model(name: str, n_channels: int, L: int, T_pred: int, device: torch.device,
               batch_size: int = 32):
    """Construct SDFormer / PatchTST / iTransformer with CPN-appropriate settings."""
    configs = argparse.Namespace(
        task_name="long_term_forecast",
        seq_len=L,
        label_len=0,
        pred_len=T_pred,
        enc_in=n_channels,
        dec_in=n_channels,
        c_out=n_channels,
        d_model=128,            # smaller than ETT (512) to fit the CPN scale
        n_heads=4,
        e_layers=1,             # 1 layer is enough for the synthetic CPN series
        d_layers=1,
        d_ff=256,
        moving_avg=25,
        factor=1,
        distil=True,
        dropout=0.1,
        embed="timeF",
        freq="h",
        activation="gelu",
        channel_independence=1,
        decomp_method="moving_avg",
        use_norm=1,
        patch_sizes="8,16,32",
        # SDFormer ablation flags (all disabled for the main run)
        use_shared_encoder=0,
        use_uniform_atsr=0,
        use_equal_fusion=0,
        use_standard_norm=0,
        # SDFormer norm-MLP width
        p_hidden_dims=[128, 128],
        p_hidden_layers=2,
        # Mamba / S-Mamba
        expand=2,
        d_conv=4,
        # WPMixer / iTransformer
        batch_size=batch_size,
        device=device,
        patch_len=16,
        use_amp=False,
        seq_len_orig=L,
    )

    name = name.lower()
    if name == "sdformer":
        from models.SDFormer import Model
    elif name == "patchtst":
        from models.PatchTST import Model
    elif name == "itransformer":
        from models.iTransformer import Model
    elif name == "smamba":
        # Pure-Python Mamba (no mamba_ssm CUDA dep) standing in for S-Mamba
        from models.MambaSimple import Model
    elif name == "wpmixer":
        from models.WPMixer import Model
    elif name == "autoformer":
        from models.Autoformer import Model
    elif name == "fedformer":
        from models.FEDformer import Model
    else:
        raise ValueError(f"Unknown model: {name}")

    model = Model(configs).to(device)
    return model


def standardise(train: np.ndarray, *others: np.ndarray):
    """Per-channel z-score using train statistics; invert with returned mu/sig."""
    mu = train.mean(axis=0, keepdims=True)
    sig = train.std(axis=0, keepdims=True) + 1e-6
    out = [(train - mu) / sig]
    out += [(o - mu) / sig for o in others]
    return out, mu, sig


def train_one(
    model_name: str,
    train_data: np.ndarray,
    val_data: np.ndarray,
    L: int,
    T_pred: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    seed: int = 2021,
    calendar_fn: Optional[callable] = None,
) -> nn.Module:
    """Train one model on the (train, val) data and return the fitted model."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n_channels = train_data.shape[1]
    model = make_model(model_name, n_channels, L, T_pred, device, batch_size=batch_size)
    optim = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    train_ds = CPNSlidingWindow(train_data, L, T_pred)
    val_ds = CPNSlidingWindow(val_data, L, T_pred)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    def _dec_inputs(B, start_ts_idx: int = 0):
        # Encoder-only models (SDFormer, PatchTST, iTransformer) ignore x_dec
        # and x_mark entirely. Encoder-decoder models (Autoformer, FEDformer)
        # need a decoder input of length `T_pred` and a calendar stamp; we
        # pass `x_mark = None` to skip the temporal embedding, which avoids
        # the channel-shape mismatch in the Time-Series-Library embedding
        # pipeline that is hard-coded to the ETT calendar layout.
        x_dec = torch.zeros(B, T_pred, n_channels, device=device)
        x_mark = None
        return x_dec, x_mark

    best_val = float("inf")
    best_state = None
    patience, bad = 3, 0
    for ep in range(epochs):
        model.train()
        for x, y in train_loader:
            x = x.to(device); y = y.to(device)
            x_dec, x_mark = _dec_inputs(x.shape[0], start_ts_idx=0)
            out = model(x, None, x_dec, x_mark)
            if out is None:
                continue
            if out.shape[1] > y.shape[1]:
                out = out[:, -y.shape[1]:, :]
            loss = loss_fn(out, y)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

        model.eval()
        val_losses = []
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device); y = y.to(device)
                x_dec, x_mark = _dec_inputs(x.shape[0], start_ts_idx=0)
                out = model(x, None, x_dec, x_mark)
                if out is None:
                    continue
                if out.shape[1] > y.shape[1]:
                    out = out[:, -y.shape[1]:, :]
                val_losses.append(loss_fn(out, y).item())
        val_loss = float(np.mean(val_losses)) if val_losses else float("inf")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict_rolling(
    model: nn.Module,
    full_data: np.ndarray,
    L: int,
    T_pred: int,
    device: torch.device,
    batch_size: int = 64,
    calendar_fn: Optional[callable] = None,
) -> np.ndarray:
    """
    Non-overlapping rolling prediction: for each start t in [0, T-T_pred, T_pred),
    feed x[t : t+L] -> predict y[t+L : t+L+T_pred] and accumulate per-step
    predictions so each test time gets a forecast (averaged across windows
    that cover it).

    Returns an array of shape (T, N) of one-step-ahead predictions; rows
    where the model could not predict are set to 0.
    """
    model.eval()
    T, N = full_data.shape
    preds = np.zeros((T, N), dtype=np.float64)
    counts = np.zeros((T, 1), dtype=np.float64)
    data_t = torch.from_numpy(full_data.astype(np.float32))
    starts = list(range(0, T - L - T_pred + 1, T_pred))
    if not starts:
        starts = [0]
    for s in starts:
        x = data_t[s : s + L].unsqueeze(0).to(device)  # (1, L, N)
        x_dec = torch.zeros(1, T_pred, N, device=device)
        x_mark = None
        y_hat = model(x, None, x_dec, x_mark)
        if y_hat is None:
            continue
        y_hat = y_hat[0].cpu().numpy()  # (T_pred, N)
        for h in range(T_pred):
            t_idx = s + L + h
            if 0 <= t_idx < T:
                preds[t_idx] += y_hat[h]
                counts[t_idx] += 1
    counts[counts == 0] = 1.0
    preds = preds / counts
    return preds


# ---------------------------------------------------------------------------
# Combined run: train all models, predict on test horizon, score cost
# ---------------------------------------------------------------------------
@dataclass
class HorizonResult:
    horizon: int
    models: dict   # model_name -> dict(MSE, MAE, cost, sla, idle)


def run_horizon(
    horizon: int,
    models: list[str],
    train_data: np.ndarray,
    val_data: np.ndarray,
    test_data: np.ndarray,
    L: int,
    epochs: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    idle_lambda: float,
) -> HorizonResult:
    print(f"\n=== Horizon T = {horizon} ===", flush=True)
    result = {"horizon": horizon, "models": {}}

    # Calendar features: for freq='h' Time-Series-Library uses 4 columns
    # (month, day, weekday, hour). Encoder-decoder models (Autoformer,
    # FEDformer) consume these via their temporal embedding; encoder-only
    # models (SDFormer, PatchTST, iTransformer) ignore them.
    total_T = train_data.shape[0] + val_data.shape[0] + test_data.shape[0]
    base_ts = pd.date_range("2020-01-01", periods=total_T, freq="h")

    def calendar_fn(start_idx: int, length: int) -> np.ndarray:
        ts = base_ts[start_idx : start_idx + length]
        feats = np.stack([
            ts.month.values - 1,
            ts.day.values - 1,
            ts.weekday.values,
            ts.hour.values,
        ], axis=-1).astype(np.float32)
        return feats

    # Sanity baselines (don't need training)
    sched = Scheduler(CPNDemandSimulator(SimConfig()))  # sim is only used for id; we use test_data directly
    truth = test_data  # (T, N)
    reactive = evaluate_reservation(truth, np.zeros_like(truth), idle_lambda)
    perfect = evaluate_reservation(truth, truth.copy(), idle_lambda)
    # Moving average of the previous L hours
    Tt = truth.shape[0]
    ma_reserve = np.zeros_like(truth)
    for t in range(Tt):
        lo = max(0, t - L)
        if t > 0:
            ma_reserve[t] = truth[lo:t].mean(axis=0)
    ma = evaluate_reservation(truth, ma_reserve, idle_lambda)
    print(f"  [Reactive  ] cost={reactive.total_cost:.1f}  (sla={reactive.total_sla:.1f}, idle={reactive.total_idle:.1f})")
    print(f"  [MovingAvg ] cost={ma.total_cost:.1f}  (sla={ma.total_sla:.1f}, idle={ma.total_idle:.1f})")
    print(f"  [Perfect   ] cost={perfect.total_cost:.1f}  (sla={perfect.total_sla:.1f}, idle={perfect.total_idle:.1f})")
    result["models"]["Reactive"] = {
        "MSE": None, "MAE": None,
        "cost": reactive.total_cost, "sla": reactive.total_sla, "idle": reactive.total_idle,
    }
    result["models"]["MovingAvg"] = {
        "MSE": None, "MAE": None,
        "cost": ma.total_cost, "sla": ma.total_sla, "idle": ma.total_idle,
    }
    result["models"]["Perfect"] = {
        "MSE": None, "MAE": None,
        "cost": perfect.total_cost, "sla": perfect.total_sla, "idle": perfect.total_idle,
    }

    for mname in models:
        t0 = time.time()
        mdl = train_one(
            mname, train_data, val_data, L, horizon,
            epochs=epochs, batch_size=batch_size, lr=lr, device=device,
            calendar_fn=calendar_fn,
        )
        # Build a full-series for rolling prediction: train + val + test
        full = np.concatenate([train_data, val_data, test_data], axis=0)
        preds = predict_rolling(mdl, full, L, horizon, device,
                                calendar_fn=calendar_fn)
        # Only score on the test portion
        train_n = train_data.shape[0]
        val_n = val_data.shape[0]
        test_n = test_data.shape[0]
        preds_test = preds[train_n + val_n - L: train_n + val_n + test_n]  # align
        # Trim/pad to test length
        preds_test = preds_test[:test_n]
        if preds_test.shape[0] < test_n:
            pad = np.zeros((test_n - preds_test.shape[0], preds_test.shape[1]))
            preds_test = np.concatenate([preds_test, pad], axis=0)

        # Forecasting accuracy on test (in original scale)
        mse = float(((preds_test - test_data) ** 2).mean())
        mae = float(np.abs(preds_test - test_data).mean())
        # Reservation: predicted demand (clipped at zero)
        reserve = np.maximum(preds_test, 0.0)
        c = evaluate_reservation(test_data, reserve, idle_lambda)
        elapsed = time.time() - t0
        print(f"  [{mname:11s}] MSE={mse:.4f}  MAE={mae:.4f}  "
              f"cost={c.total_cost:.1f}  (sla={c.total_sla:.1f}, idle={c.total_idle:.1f})  "
              f"[{elapsed:.1f}s]")
        result["models"][mname] = {
            "MSE": mse, "MAE": mae,
            "cost": c.total_cost, "sla": c.total_sla, "idle": c.total_idle,
        }

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sites", type=int, default=10)
    parser.add_argument("--T_total", type=int, default=672)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--L", type=int, default=96, help="look-back window")
    parser.add_argument("--horizons", type=int, nargs="+", default=[24, 48, 96, 168])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--models", type=str, nargs="+",
                        default=["SDFormer", "PatchTST", "iTransformer"])
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--out", type=str, default="cpn_simulation/results.json")
    parser.add_argument("--idle_lambda", type=float, default=0.5)
    parser.add_argument("--no_cuda", action="store_true", help="force CPU even if CUDA is available")
    args = parser.parse_args()

    if args.no_cuda or not torch.cuda.is_available():
        # Use plain CPU: WPMixer's DWT operator has dtype/device mismatches
        # on MPS that are unrelated to the simulation results.
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print(f"Device: {device}")

    # Generate demand
    cfg = SimConfig(n_sites=args.n_sites, T=args.T_total, seed=args.seed)
    sim = CPNDemandSimulator(cfg)
    demand = sim.generate_demand()  # (T, M, K)
    Tt, M, K = demand.shape
    flat = demand.transpose(0, 2, 1).reshape(Tt, M * K)  # (T, N)

    n_train = int(Tt * args.train_ratio)
    n_val = int(Tt * args.val_ratio)
    train_data = flat[:n_train]
    val_data = flat[n_train : n_train + n_val]
    test_data = flat[n_train + n_val :]
    print(f"Demand shape: {demand.shape}, channels: {M*K}, "
          f"train/val/test = {len(train_data)}/{len(val_data)}/{len(test_data)}")

    all_results = []
    for H in args.horizons:
        r = run_horizon(
            horizon=H, models=args.models,
            train_data=train_data, val_data=val_data, test_data=test_data,
            L=args.L, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            device=device, idle_lambda=args.idle_lambda,
        )
        all_results.append(r)

    # Persist results
    out_path = os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    serialisable = []
    for r in all_results:
        serialisable.append(r)  # r is already a dict {horizon, models}
    with open(out_path, "w") as f:
        json.dump(
            {"args": vars(args), "results": serialisable},
            f, indent=2, default=float,
        )
    print(f"\nSaved results to: {out_path}")


if __name__ == "__main__":
    main()
