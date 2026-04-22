"""
Plot success rates across action horizons from evaluate_checkpoints.sh output.

Scans an output directory with the structure:
    <experiment_path>/T_a_<horizon>/<checkpoint_stem>/results.pkl

For each horizon, selects the checkpoint with the highest success rate and
produces a figure with overlaid traces for multiple experiments.

Example usage
-------------

Single experiment:
   python src/eval/plot_eval_results.py \
       --experiment-path outputs/2_obs_one_leg_scripted_r3m \
       --plot-name "One-Leg - Markovian Expert (500 Trials)" \
       --output outputs/plots/baseline_markovian_expert_one_leg.png

Multiple experiments with custom legend labels:
   python src/eval/plot_eval_results.py \
       --experiment-path outputs/2_obs_one_leg_scripted_r3m outputs/2_obs_one_leg_teleop_r3m \
       --experiment-name "Markovian Expert" "Non-Markovian Expert" \
       --plot-name "One-Leg - Markovian Expert vs Non-Markovian Expert (500 Trials)" \
       --output outputs/plots/comparison_markovian_expert_non_markovian_expert_one_leg.png

Don't set --output to skip saving. Set --show to open an interactive window.
"""

from __future__ import annotations

import argparse
import math
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.proportion as smp
from matplotlib.ticker import ScalarFormatter

# -----------------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------------

NAVY = "#1f3b6f"
GRID_COLOR = "#bdbdbd"


def _generate_color_palette(num_colors: int) -> List[str]:
    if num_colors <= 0:
        return []
    if num_colors == 1:
        return [NAVY]
    # Interpolate green → red
    colors = []
    for i in range(num_colors):
        t = i / (num_colors - 1)
        r = int(t * 255)
        g = int((1 - t) * 200)
        b = 80
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class CheckpointResult:
    horizon: int
    success_rate: float
    num_trials: int
    checkpoint_dir: Path
    num_checkpoints_available: int = 1
    all_checkpoint_trials: List[int] = field(default_factory=list)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def _load_results_pkl(results_path: Path) -> Tuple[float, int]:
    """Return (success_rate, num_trials) from a results.pkl file."""
    with results_path.open("rb") as f:
        data = pickle.load(f)
    n_total = data.get("n_total", 0)
    n_success = data.get("n_success", 0)
    if n_total == 0:
        return float("nan"), 0
    return n_success / n_total, n_total


def _parse_horizon(dirname: str) -> Optional[int]:
    """Parse integer horizon from a directory name like 'T_a_8'."""
    if not dirname.startswith("T_a_"):
        return None
    try:
        return int(dirname[4:])
    except ValueError:
        return None


def collect_best_results(experiment_path: Path) -> List[CheckpointResult]:
    """For each T_a_<N> sub-directory, find the best checkpoint by success rate."""
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path '{experiment_path}' does not exist.")

    results: List[CheckpointResult] = []
    for horizon_dir in sorted(experiment_path.iterdir()):
        if not horizon_dir.is_dir():
            continue
        horizon = _parse_horizon(horizon_dir.name)
        if horizon is None:
            continue

        candidates: List[CheckpointResult] = []
        checkpoint_dirs = [d for d in sorted(horizon_dir.iterdir()) if d.is_dir()]
        for ckpt_dir in checkpoint_dirs:
            pkl = ckpt_dir / "results.pkl"
            if not pkl.exists():
                continue
            rate, total = _load_results_pkl(pkl)
            if math.isnan(rate):
                continue
            candidates.append(CheckpointResult(horizon, rate, total, ckpt_dir))

        if not candidates:
            print(f"  Skipping {horizon_dir.name}: no valid results.pkl found.")
            continue

        # Pick the checkpoint with the most trials, then highest success rate
        max_trials = max(c.num_trials for c in candidates)
        top = [c for c in candidates if c.num_trials == max_trials]
        best = max(top, key=lambda c: c.success_rate)
        best.num_checkpoints_available = len(candidates)
        best.all_checkpoint_trials = [c.num_trials for c in candidates]
        results.append(best)

    results.sort(key=lambda r: r.horizon)
    return results


def collect_all_checkpoint_results(experiment_path: Path) -> Dict[str, List[CheckpointResult]]:
    """Collect all checkpoints grouped by checkpoint name across all horizons."""
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path '{experiment_path}' does not exist.")

    by_ckpt: Dict[str, List[CheckpointResult]] = {}
    for horizon_dir in sorted(experiment_path.iterdir()):
        if not horizon_dir.is_dir():
            continue
        horizon = _parse_horizon(horizon_dir.name)
        if horizon is None:
            continue
        for ckpt_dir in sorted(horizon_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            pkl = ckpt_dir / "results.pkl"
            if not pkl.exists():
                continue
            rate, total = _load_results_pkl(pkl)
            if math.isnan(rate):
                continue
            name = ckpt_dir.name
            by_ckpt.setdefault(name, []).append(CheckpointResult(horizon, rate, total, ckpt_dir))

    for lst in by_ckpt.values():
        lst.sort(key=lambda r: r.horizon)
    return by_ckpt


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def make_plot(
    experiments: List[Tuple[str, Sequence[CheckpointResult], str]],
    dpi: int,
    plot_name: Optional[str] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.set_facecolor("white")

    all_horizons = sorted({r.horizon for _, results, _ in experiments for r in results})
    show_ckpt_labels = len(experiments) == 1

    for exp_name, results, color in experiments:
        if not results:
            continue
        horizons = np.array([r.horizon for r in results], dtype=float)
        rates = np.array([r.success_rate for r in results], dtype=float)
        trials = np.array([r.num_trials for r in results], dtype=int)

        ci = np.array(
            [smp.proportion_confint(int(p * n), n, alpha=0.05, method="wilson") for p, n in zip(rates, trials)]
        )
        yerr = np.vstack([np.clip(rates - ci[:, 0], 0, 1), np.clip(ci[:, 1] - rates, 0, 1)])

        ax.plot(horizons, rates, color=color, linewidth=1.5, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=exp_name, zorder=3)
        ax.errorbar(horizons, rates, yerr=yerr, fmt="none", ecolor=color, elinewidth=1.0,
                    capsize=4.0, capthick=1.0, alpha=0.9, zorder=2)

        if show_ckpt_labels:
            for res in results:
                if res.num_checkpoints_available > 1:
                    name = res.checkpoint_dir.name
                    label = name[:10] + "..." if len(name) > 10 else name
                    ax.annotate(label, xy=(res.horizon, res.success_rate), xytext=(0, 5),
                                textcoords="offset points", fontsize=6, color=color,
                                ha="center", va="bottom", alpha=0.8)

    # ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    if all_horizons:
        ax.set_xticks(all_horizons)
        ax.set_xticklabels([str(h) for h in all_horizons])

    title = plot_name or ("Action Horizon Comparison" if len(experiments) > 1 else experiments[0][0])
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Action Horizon (steps)", fontsize=12)
    ax.set_ylabel("Success Rate", fontsize=12)

    ax.grid(True, which="major", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", color=GRID_COLOR, linestyle="-", linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#4f4f4f")

    ax.tick_params(axis="both", which="major", labelsize=8, length=6, width=1)
    ax.tick_params(axis="x", which="minor", length=4, width=0.8)
    ax.tick_params(axis="y", which="minor", left=False)

    if len(experiments) > 1:
        ax.legend(loc="best", fontsize=9, framealpha=0.9, edgecolor="#4f4f4f")

    fig.tight_layout()
    fig.set_dpi(dpi)
    return fig


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot success rates across action horizons from evaluate_checkpoints.sh output."
    )
    parser.add_argument("--experiment-path", type=Path, nargs="+", required=True,
                        help="Path(s) to experiment output directories containing T_a_<N> sub-folders.")
    parser.add_argument("--experiment-name", type=str, nargs="+", default=None,
                        help="Legend label(s) for each experiment (defaults to directory names).")
    parser.add_argument("--plot-name", type=str, default=None, help="Plot title.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save the figure (PNG, PDF, etc.). Omit to skip saving.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving to disk.")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Plot every checkpoint instead of just the best per horizon.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    exp_paths = args.experiment_path
    if args.experiment_name is None:
        exp_labels = [p.name for p in exp_paths]
    else:
        if len(args.experiment_name) != len(exp_paths):
            raise ValueError(
                f"--experiment-name count ({len(args.experiment_name)}) must match "
                f"--experiment-path count ({len(exp_paths)})"
            )
        exp_labels = args.experiment_name

    experiments: List[Tuple[str, Sequence[CheckpointResult], str]] = []

    if args.all_checkpoints:
        if len(exp_paths) > 1:
            raise ValueError("--all-checkpoints is only supported for a single --experiment-path")
        by_ckpt = collect_all_checkpoint_results(exp_paths[0])
        if not by_ckpt:
            raise RuntimeError(f"No valid checkpoints found under {exp_paths[0]}.")
        palette = _generate_color_palette(len(by_ckpt))
        for idx, (ckpt_name, results) in enumerate(sorted(by_ckpt.items())):
            experiments.append((ckpt_name, results, palette[idx]))
            print(f"\n{ckpt_name}:")
            for r in results:
                print(f"  T_a_{r.horizon}: success_rate={r.success_rate:.3f} ({r.num_trials} trials)")
    else:
        palette = [NAVY] if len(exp_paths) == 1 else _generate_color_palette(len(exp_paths))
        for idx, (exp_path, label) in enumerate(zip(exp_paths, exp_labels)):
            results = collect_best_results(exp_path)
            if not results:
                print(f"Warning: no valid results.pkl files found under {exp_path}. Skipping.")
                continue
            experiments.append((label, results, palette[idx]))
            print(f"\n{label} — best checkpoint per horizon:")
            for r in results:
                if r.num_checkpoints_available > 1:
                    trials_str = ", ".join(str(t) for t in r.all_checkpoint_trials)
                    n_tag = f" [{r.num_checkpoints_available} ckpts: {trials_str} trials]"
                else:
                    n_tag = ""
                print(f"  T_a_{r.horizon}: {r.success_rate:.3f} ({r.num_trials} trials) -> {r.checkpoint_dir}{n_tag}")

    if not experiments:
        raise RuntimeError("No valid experiments to plot.")

    fig = make_plot(experiments, dpi=args.dpi, plot_name=args.plot_name)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
        print(f"\nSaved figure to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
