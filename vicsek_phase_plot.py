#!/usr/bin/env python3
"""
2D Vicsek の秩序変数 vs ノイズ η を、ブラウザ版 viscek.html と同じ力学（餌なし）で計算します。

対応させている条件（デフォルト）
--------------------------------
- 周期境界の正方形 [0, L)^2（HTML の SZ に相当）
- 相互作用半径 R = 0.1 * L（viz. `const R = SZ * 0.1`）
- 速度 SPEED（ステップごとに位置更新）
- ノイズ: `new_angle += (rand - 0.5) * eta * 2*pi` と同じ幅の一様乱数加算
- **逐次更新**: agent 0→N-1 の順で、その時点の他粒子の状態を使う（JavaScript と同順）

グラフは「複数シードでの初期条件平均」と「緩和後の時間平均」を取ります。

デフォルトは「素早く形だけ見る」向けで総ステップはおおよそ 10^4 規模です。
本番の曲線には `--relax` / `--sample` / `--seeds` / `--eta-points` を十分大きくしてください。
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class Params:
    L: float = 400.0
    n_particles: int = 100
    speed: float = 1.8
    ratio_r_to_l: float = 0.1
    relax_steps: int = 100
    sample_steps: int = 100
    sample_stride: int = 10
    n_seeds: int = 8


def interaction_radius(params: Params) -> float:
    return params.ratio_r_to_l * params.L


def wrap_delta(delta: float, length: float) -> float:
    half = length * 0.5
    if delta > half:
        delta -= length
    elif delta < -half:
        delta += length
    return delta


def min_image(dx: np.ndarray, length: float) -> np.ndarray:
    """Periodic minimum-image wrap for displacement arrays."""
    return dx - length * np.round(dx / length)


def polar_order(mx: np.ndarray | float, my: np.ndarray | float) -> np.ndarray | float:
    """|⟨v⟩| = |(1/N) ∑_i v_i| with v_i = (cos θ_i, sin θ_i).

    mx, my must already be **means**: ⟨cos θ⟩, ⟨sin θ⟩ (e.g. from `np.cos(theta).mean()`).
    Do not pass raw sums here (use sum / N first).
    """
    return np.hypot(mx, my)


def simulate_one_eta(
    eta: float,
    rng: np.random.Generator,
    params: Params,
    on_step=None,
) -> float:
    """Return time-averaged polar order after relaxation.

    on_step: optional callback invoked after each integration step (for progress bars).
    """
    L = params.L
    speed = params.speed
    R = interaction_radius(params)
    r2 = R * R
    n = params.n_particles

    x = rng.random(n) * L
    y = rng.random(n) * L
    theta = rng.random(n) * (2 * np.pi)

    def step() -> None:
        """Matches viscek.html sequential in-place updates (same neighbor rule)."""
        for i in range(n):
            xi, yi = x[i], y[i]
            dx = min_image(x - xi, L)
            dy = min_image(y - yi, L)
            mask = dx * dx + dy * dy < r2
            mask[i] = False
            sx = float(np.cos(theta[i]) + np.cos(theta)[mask].sum())
            sy = float(np.sin(theta[i]) + np.sin(theta)[mask].sum())
            new_angle = np.arctan2(sy, sx)
            new_angle += (rng.random() - 0.5) * eta * (2 * np.pi)
            new_angle = np.arctan2(np.sin(new_angle), np.cos(new_angle))
            theta[i] = new_angle
            x[i] = (xi + np.cos(theta[i]) * speed + L) % L
            y[i] = (yi + np.sin(theta[i]) * speed + L) % L

    for _ in range(params.relax_steps):
        step()
        if on_step:
            on_step()

    vals: list[float] = []
    for s in range(params.sample_steps):
        step()
        if on_step:
            on_step()
        if (s + 1) % params.sample_stride != 0:
            continue
        mx = np.cos(theta).mean()
        my = np.sin(theta).mean()
        vals.append(float(polar_order(mx, my)))

    return float(np.mean(vals)) if vals else 0.0


def main() -> None:
    p = argparse.ArgumentParser(description="Vicsek phase curve (noise vs polar order)")
    p.add_argument("--L", type=float, default=400.0, help="Box side (matches typical viscek canvas SZ)")
    p.add_argument("--N", type=int, default=100, dest="n_particles", help="Particle count")
    p.add_argument("--speed", type=float, default=1.8)
    p.add_argument("--ratio-r", type=float, default=0.1, dest="ratio_r_to_l", help="R/L (HTML: 0.1)")
    p.add_argument("--relax", type=int, default=100, dest="relax_steps")
    p.add_argument("--sample", type=int, default=100, dest="sample_steps")
    p.add_argument("--stride", type=int, default=10, dest="sample_stride")
    p.add_argument("--seeds", type=int, default=8, dest="n_seeds")
    p.add_argument("--eta-min", type=float, default=0.0)
    p.add_argument("--eta-max", type=float, default=1.0)
    p.add_argument("--eta-points", type=int, default=11)
    p.add_argument("--out-prefix", type=str, default="vicsek_phase")
    p.add_argument("--no-progress", action="store_true", help="Disable tqdm step progress bar")
    args = p.parse_args()

    params = Params(
        L=args.L,
        n_particles=args.n_particles,
        speed=args.speed,
        ratio_r_to_l=args.ratio_r_to_l,
        relax_steps=args.relax_steps,
        sample_steps=args.sample_steps,
        sample_stride=args.sample_stride,
        n_seeds=args.n_seeds,
    )

    base = Path(__file__).resolve().parent
    eta_grid = np.linspace(args.eta_min, args.eta_max, args.eta_points)

    csv_path = base / f"{args.out_prefix}_data.csv"
    plot_path = base / f"{args.out_prefix}_plot.png"

    rows = []
    means = []
    stderr = []

    total_steps_one_run = params.relax_steps + params.sample_steps
    total_steps_all = int(len(eta_grid) * params.n_seeds * total_steps_one_run)

    tqdm_mod = None
    if not args.no_progress:
        try:
            from tqdm import tqdm as tqdm_mod
        except ImportError:
            tqdm_mod = None
            print("# Install `tqdm` (`pip install tqdm`) for step progress bar, or use --no-progress")

    pbar = None
    if tqdm_mod is not None:
        pbar = tqdm_mod(
            total=total_steps_all,
            unit="step",
            desc="Simulation steps",
            mininterval=0.2,
            smoothing=0.05,
        )

    print("# Params:", params)
    print("# eta_min..max:", args.eta_min, args.eta_max, "points:", args.eta_points)
    print("# Total integration steps:", total_steps_all)

    def on_step() -> None:
        if pbar:
            pbar.update(1)

    for eta in eta_grid:
        orders = []
        if pbar:
            pbar.set_postfix_str(f"η={eta:.3f}")
        for seed in range(params.n_seeds):
            rng = np.random.default_rng(seed * 100_003 + int(round(eta * 1e6)))
            orders.append(simulate_one_eta(float(eta), rng, params, on_step=on_step if pbar else None))
        m = float(np.mean(orders))
        se = float(np.std(orders, ddof=1) / np.sqrt(len(orders))) if len(orders) > 1 else 0.0
        means.append(m)
        stderr.append(se)
        rows.append(
            {
                "eta": float(eta),
                "polar_order_mean": m,
                "polar_order_stderr": se,
                "n_seeds": params.n_seeds,
                "relax_steps": params.relax_steps,
                "sample_steps": params.sample_steps,
                "sample_stride": params.sample_stride,
                "L": params.L,
                "N": params.n_particles,
                "speed": params.speed,
                "R_over_L": params.ratio_r_to_l,
            }
        )
        print(f"eta={eta:.4f}  <|v|>={m:.4f} +/- {se:.4f}")

    if pbar:
        pbar.close()

    with csv_path.open("w", newline="", encoding="utf-8") as fp:
        w = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4), dpi=120)
        ax.errorbar(
            eta_grid,
            means,
            yerr=stderr,
            fmt="-o",
            markersize=3,
            linewidth=1.2,
            capsize=2,
            color="#0369a1",
            ecolor="#94a3b8",
            elinewidth=1,
        )
        ax.set_xlabel(r"$\eta$")
        ax.set_ylabel(r"$|\langle \mathbf{v} \rangle|$")
        ax.set_title("Vicsek 2D")
        ax.grid(True, alpha=0.35)
        ax.set_ylim(0, 1.05)
        fig.tight_layout()
        fig.savefig(plot_path)
        plt.close(fig)
        print("# Wrote:", csv_path)
        print("# Wrote:", plot_path)
    except ImportError:
        print("# matplotlib not installed; CSV only:", csv_path)


if __name__ == "__main__":
    main()
