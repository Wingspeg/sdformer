"""
Multi-site Computing Power Network (CPN) simulator.

Generates synthetic supply-demand time series for a network of M sites,
each with K resource dimensions (CPU / Mem / GPU). The demand at each
site mixes a stable periodic component with sparse, heavy-tailed burst
events, mirroring the behavioral assumption made in the SDFormer paper
(Section 3.1). The simulator is intentionally lightweight so that
forecasting models can be trained and evaluated entirely on CPU within
minutes, and the resulting scheduler cost C (Eq. 4) can be reported
alongside the forecasting MSE/MAE in the paper.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SimConfig:
    n_sites: int = 10                  # M: number of CPN sites
    n_resources: int = 3               # K: CPU / Mem / GPU
    T: int = 4 * 7 * 24                # 4 weeks of hourly data = 672 steps
    seed: int = 2024                   # RNG seed for reproducibility

    # Periodic demand component (per site, per resource)
    baseline_frac_lo: float = 0.30     # baseline demand / capacity (lower bound)
    baseline_frac_hi: float = 0.60     # baseline demand / capacity (upper bound)
    daily_amp: float = 0.18            # amplitude of 24h cycle, fraction of baseline
    weekly_amp: float = 0.20           # amplitude of 168h cycle, fraction of baseline
    phase_range: float = 2 * np.pi     # phase offset drawn uniformly

    # Burst component (per site, per resource, per time step)
    burst_prob: float = 0.05            # Poisson arrival probability
    burst_sigma: float = 0.9            # log-normal sigma (heavy tail)
    burst_scale: float = 0.35           # burst amplitude / capacity

    # Per-site capacity ranges (CPU cores, Mem GB, GPU count)
    cpu_cap_range: tuple = (32.0, 128.0)
    mem_cap_range: tuple = (64.0, 512.0)
    gpu_cap_range: tuple = (0.0, 8.0)

    # Scheduling penalty (Eq. 4)
    idle_lambda: float = 0.5            # weight on idle capacity in cost C


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------
class CPNDemandSimulator:
    """Generates ground-truth demand trajectories for an M-site, K-resource CPN."""

    RESOURCE_NAMES = ("cpu", "mem", "gpu")

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        M, K = cfg.n_sites, cfg.n_resources
        assert K == 3, "Simulator configured for exactly K=3 (CPU/Mem/GPU)"

        # ---- Site capacities (heterogeneous) ----
        self.capacities = np.zeros((M, K), dtype=np.float64)
        self.capacities[:, 0] = self.rng.uniform(*cfg.cpu_cap_range, size=M)
        self.capacities[:, 1] = self.rng.uniform(*cfg.mem_cap_range, size=M)
        self.capacities[:, 2] = self.rng.uniform(*cfg.gpu_cap_range, size=M)

        # ---- Per-site baseline utilisation and cycle amplitudes ----
        self.baseline_frac = self.rng.uniform(
            cfg.baseline_frac_lo, cfg.baseline_frac_hi, size=(M, K)
        )
        # baseline absolute = baseline_frac * capacity
        self.daily_amp_abs = (
            self.rng.uniform(0.0, cfg.daily_amp, size=(M, K)) * self.capacities
        )
        self.weekly_amp_abs = (
            self.rng.uniform(0.0, cfg.weekly_amp, size=(M, K)) * self.capacities
        )
        self.phase = self.rng.uniform(0.0, cfg.phase_range, size=(M, K))

    # ------------------------------------------------------------------
    def generate_demand(self) -> np.ndarray:
        """Return a (T, M, K) demand array (non-negative).

        Burst model: sparse Bernoulli arrivals (probability ``burst_prob``)
        with log-normal heavy-tail magnitudes. Matches the heavy-tailed
        burst assumption of Section 3.1 and the burstiness statistics of
        the ETTh1 / ETTh2 benchmarks that motivate SDFormer.
        """
        cfg = self.cfg
        T, M, K = cfg.T, cfg.n_sites, cfg.n_resources
        t = np.arange(T, dtype=np.float64)

        # Periodic component
        baseline = self.baseline_frac[None, :, :] * self.capacities[None, :, :]
        daily = self.daily_amp_abs[None, :, :] * np.sin(
            2 * np.pi * t[:, None, None] / 24.0 + self.phase[None, :, :]
        )
        weekly = self.weekly_amp_abs[None, :, :] * np.sin(
            2 * np.pi * t[:, None, None] / 168.0 + self.phase[None, :, :]
        )
        periodic = baseline + daily + weekly

        # Burst component (Poisson arrival, log-normal heavy-tail magnitude)
        burst_active = self.rng.binomial(1, cfg.burst_prob, size=(T, M, K))
        burst_mag = np.exp(
            self.rng.normal(0.0, cfg.burst_sigma, size=(T, M, K))
        )
        burst_abs = burst_active * burst_mag * cfg.burst_scale * self.capacities[None, :, :]

        demand = np.maximum(periodic + burst_abs, 0.0)
        return demand


# ---------------------------------------------------------------------------
# Scheduling strategies + cost evaluation (Eq. 4 of the paper)
# ---------------------------------------------------------------------------
@dataclass
class CostBreakdown:
    """Aggregate cost components for a scheduling strategy."""
    total_cost: float
    total_sla: float
    total_idle: float
    sla_per_step: np.ndarray
    idle_per_step: np.ndarray
    cost_per_step: np.ndarray


def evaluate_reservation(
    demand: np.ndarray,
    reserve: np.ndarray,
    idle_lambda: float = 0.5,
) -> CostBreakdown:
    """
    Apply the scheduling cost C of Eq. 4 to a sequence of reservations.

    SLA_{c,t} = max(0, d_{c,t} - reserve_{c,t})
    idle_{c,t} = max(0, reserve_{c,t} - d_{c,t})
    cost_{c,t} = SLA + lambda * idle
    where c indexes site-resource channels (works for any demand.ndim >= 2).
    """
    if reserve.shape != demand.shape:
        raise ValueError(
            f"reserve shape {reserve.shape} != demand shape {demand.shape}"
        )
    sla = np.maximum(demand - reserve, 0.0)
    idle = np.maximum(reserve - demand, 0.0)
    cost = sla + idle_lambda * idle
    # Per-step totals: sum over all non-time axes (channel dimensions)
    reduce_axes = tuple(range(1, demand.ndim))
    return CostBreakdown(
        total_cost=float(cost.sum()),
        total_sla=float(sla.sum()),
        total_idle=float(idle.sum()),
        sla_per_step=sla.sum(axis=reduce_axes) if demand.ndim > 1 else sla,
        idle_per_step=idle.sum(axis=reduce_axes) if demand.ndim > 1 else idle,
        cost_per_step=cost.sum(axis=reduce_axes) if demand.ndim > 1 else cost,
    )


# ---------------------------------------------------------------------------
# Forecasting-based scheduling
# ---------------------------------------------------------------------------
class Scheduler:
    """Wraps demand generation and reservation evaluation for several strategies."""

    def __init__(self, sim: CPNDemandSimulator):
        self.sim = sim
        self.cfg = sim.cfg

    def ground_truth(self) -> np.ndarray:
        return self.sim.generate_demand()

    def evaluate_reactive(self, demand: np.ndarray) -> CostBreakdown:
        """No reservation; everything above zero is SLA violation."""
        reserve = np.zeros_like(demand)
        return evaluate_reservation(demand, reserve, self.cfg.idle_lambda)

    def evaluate_moving_average(
        self, demand: np.ndarray, window: int = 24
    ) -> CostBreakdown:
        """Reservation = trailing moving average of past demand."""
        T, M, K = demand.shape
        reserve = np.zeros_like(demand)
        for t in range(T):
            lo = max(0, t - window)
            reserve[t] = demand[lo:t].mean(axis=0) if t > 0 else 0.0
        return evaluate_reservation(demand, reserve, self.cfg.idle_lambda)

    def evaluate_perfect(self, demand: np.ndarray) -> CostBreakdown:
        """Reservation = true demand (oracle upper bound)."""
        return evaluate_reservation(demand, demand.copy(), self.cfg.idle_lambda)

    def evaluate_prediction(
        self, demand: np.ndarray, pred_demand: np.ndarray
    ) -> CostBreakdown:
        """Reservation = predicted next-step demand (T-element series)."""
        if pred_demand.shape != demand.shape:
            raise ValueError(
                f"pred_demand shape {pred_demand.shape} != demand shape {demand.shape}"
            )
        return evaluate_reservation(
            demand, np.maximum(pred_demand, 0.0), self.cfg.idle_lambda
        )


# ---------------------------------------------------------------------------
# Dataset construction: convert the (T, M, K) demand cube to a (L, N) CSV
# that the SDFormer pipeline (Dataset_Custom) can consume directly.
# ---------------------------------------------------------------------------
def write_forecasting_dataset(
    sim: CPNDemandSimulator,
    out_dir: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.1,
) -> dict:
    """
    Write a long-format CSV with columns [date, cpu_s0, mem_s0, gpu_s0, cpu_s1, ...]
    plus a sidecar metadata JSON file describing column order.
    """
    os.makedirs(out_dir, exist_ok=True)
    demand = sim.generate_demand()  # (T, M, K)
    T, M, K = demand.shape

    # Build column names: site-major ordering matches paper Eq. 2: x_t = [x_{1,t},...,x_{M,t}]
    col_names = ["date"]
    column_index: list[tuple[int, int, str]] = []
    for m in range(M):
        for k, kname in enumerate(CPNDemandSimulator.RESOURCE_NAMES):
            col_names.append(f"{kname}_s{m}")
            column_index.append((m, k, kname))

    # Build timestamps (start at 2020-01-01 00:00 UTC, hourly)
    dates = pd.date_range("2020-01-01", periods=T, freq="h")
    flat = demand.transpose(0, 2, 1).reshape(T, M * K)  # (T, M*K) site-major already? actually time x (K x M)
    # We want site-major inside the channels; SDFormer uses channel-independent
    # over the N = M*K channels, ordering does not affect numerics, so any
    # consistent flat works. Choose resource-major to match the loop above.
    flat_resource_major = np.zeros((T, M * K), dtype=np.float64)
    for m in range(M):
        for k in range(K):
            flat_resource_major[:, m * K + k] = demand[:, m, k]

    df = pd.DataFrame(flat_resource_major, columns=col_names[1:])
    df.insert(0, "date", dates)

    # Train / val / test split, no shuffle (time-series)
    n_train = int(T * train_ratio)
    n_val = int(T * val_ratio)
    train_df = df.iloc[:n_train].reset_index(drop=True)
    val_df = df.iloc[n_train : n_train + n_val].reset_index(drop=True)
    test_df = df.iloc[n_train + n_val :].reset_index(drop=True)

    train_path = os.path.join(out_dir, "CPN_sim_train.csv")
    val_path = os.path.join(out_dir, "CPN_sim_val.csv")
    test_path = os.path.join(out_dir, "CPN_sim_test.csv")
    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    meta = {
        "task": "long_term_forecast",
        "n_sites": M,
        "n_resources": K,
        "resource_names": list(CPNDemandSimulator.RESOURCE_NAMES),
        "T_total": T,
        "T_train": n_train,
        "T_val": n_val,
        "T_test": T - n_train - n_val,
        "n_channels": M * K,
        "columns": list(df.columns),
        "freq": "h",
        "sim_config": {k: float(v) if isinstance(v, (np.floating,)) else v
                        for k, v in asdict(sim.cfg).items()},
        "site_capacities": sim.capacities.tolist(),
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2, default=float)

    # Also write the full demand array as .npy for the scheduling evaluation
    np.save(os.path.join(out_dir, "demand_full.npy"), demand)
    np.save(os.path.join(out_dir, "capacities.npy"), sim.capacities)
    return meta


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="CPN demand simulator")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="output directory for the generated dataset")
    parser.add_argument("--n_sites", type=int, default=10)
    parser.add_argument("--T", type=int, default=4 * 7 * 24)
    parser.add_argument("--seed", type=int, default=2024)
    args = parser.parse_args()

    cfg = SimConfig(n_sites=args.n_sites, T=args.T, seed=args.seed)
    sim = CPNDemandSimulator(cfg)
    sched = Scheduler(sim)

    # Quick sanity stats on the generated demand
    demand = sched.ground_truth()
    cap = sim.capacities[None, :, :]  # (1, M, K) broadcast
    util = (demand / cap)
    print(f"Demand shape: {demand.shape}")
    print(f"Mean utilisation (over T,M,K): {util.mean():.3f}")
    print(f"Max utilisation: {util.max():.3f}")
    print(f"Fraction of steps where any (m,k) exceeds 95% capacity: "
          f"{(util > 0.95).any(axis=(1,2)).mean():.3f}")
    print(f"Fraction of steps where any (m,k) exceeds 100% capacity: "
          f"{(util > 1.0).any(axis=(1,2)).mean():.3f}")

    # Sanity check on scheduling strategies
    reactive = sched.evaluate_reactive(demand)
    ma = sched.evaluate_moving_average(demand, window=24)
    perfect = sched.evaluate_perfect(demand)
    print()
    print(f"[Reactive   ] total cost = {reactive.total_cost:.1f}  "
          f"(SLA={reactive.total_sla:.1f}, idle={reactive.total_idle:.1f})")
    print(f"[MovingAvg  ] total cost = {ma.total_cost:.1f}  "
          f"(SLA={ma.total_sla:.1f}, idle={ma.total_idle:.1f})")
    print(f"[Perfect    ] total cost = {perfect.total_cost:.1f}  "
          f"(SLA={perfect.total_sla:.1f}, idle={perfect.total_idle:.1f})")

    # Write the dataset
    meta = write_forecasting_dataset(sim, args.out_dir)
    print()
    print(f"Wrote forecasting dataset + meta to: {args.out_dir}")
    print(f"  channels: {meta['n_channels']} (= M*K = {meta['n_sites']}*{meta['n_resources']})")
    print(f"  splits: train={meta['T_train']}, val={meta['T_val']}, test={meta['T_test']}")


if __name__ == "__main__":
    main()
