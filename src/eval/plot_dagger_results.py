"""
Plot success rates across Execution Horizons from DAgger evaluation output.

Unlike ``plot_eval_results.py`` (where every horizon lives in a single output
directory as ``T_a_<horizon>`` sub-folders), DAgger results store each action
horizon in its own output directory. The horizon is encoded as the number after
``ah`` in the directory-name postfix, e.g.::

    2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah1
    2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah3
    2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah6
    2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah10
    2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah15

Each such directory contains the usual ``<checkpoint_stem>/results.pkl`` layout
(optionally nested under a ``T_a_<horizon>`` sub-folder). For each horizon, the
checkpoint with the highest success rate is selected and overlaid traces for
multiple experiments are produced.

An "experiment" is therefore a *list* of output directories (one per horizon).

Example usage
-------------

Single experiment (one list of per-horizon directories):
   python src/eval/plot_dagger_results.py \
       --experiment-path \
           outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah1 \
           outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah3 \
           outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah6 \
           outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah10 \
           outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah15 \
       --plot-name "One-Leg - DAgger iter0 (Non-Markovian Expert)" \
       --output outputs/plots/dagger_iter0_one_leg.png

Multiple experiments (repeat --experiment-path once per experiment):
   python src/eval/plot_dagger_results.py \
       --experiment-path outputs/..._dagger_iter0_ah1 outputs/..._dagger_iter0_ah3 ... \
       --experiment-path outputs/..._dagger_iter1_ah1 outputs/..._dagger_iter1_ah3 ... \
       --experiment-name "DAgger iter0" "DAgger iter1" \
       --plot-name "One-Leg - DAgger iter0 vs iter1" \
       --output outputs/plots/dagger_iter0_vs_iter1_one_leg.png

Don't set --output to skip saving. Set --show to open an interactive window.
"""

from __future__ import annotations

import argparse
import math
import pickle
import re
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
    """Parse integer horizon from the ``ah<N>`` postfix of a directory name.

    e.g. ``..._dagger_iter0_ah10`` -> 10.
    """
    match = re.search(r"ah(\d+)$", dirname)
    if match is None:
        return None
    return int(match.group(1))


def _find_checkpoint_dirs(experiment_path: Path) -> List[Path]:
    """Return checkpoint directories (those containing a results.pkl) under a path.

    The DAgger layout nests checkpoints under a ``T_a_<N>`` sub-folder, but we
    search recursively to be robust to either flat or nested layouts.
    """
    ckpt_dirs = {pkl.parent for pkl in experiment_path.glob("**/results.pkl")}
    return sorted(ckpt_dirs)


def collect_best_results(experiment_paths: Sequence[Path]) -> List[CheckpointResult]:
    """For each per-horizon directory, find the best checkpoint by success rate."""
    results: List[CheckpointResult] = []
    for experiment_path in experiment_paths:
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment path '{experiment_path}' does not exist.")

        horizon = _parse_horizon(experiment_path.name)
        if horizon is None:
            print(f"  Skipping {experiment_path.name}: no 'ah<N>' postfix found.")
            continue

        candidates: List[CheckpointResult] = []
        for ckpt_dir in _find_checkpoint_dirs(experiment_path):
            pkl = ckpt_dir / "results.pkl"
            rate, total = _load_results_pkl(pkl)
            if math.isnan(rate):
                continue
            candidates.append(CheckpointResult(horizon, rate, total, ckpt_dir))

        if not candidates:
            print(f"  Skipping {experiment_path.name}: no valid results.pkl found.")
            continue

        # Consider every checkpoint with valid results (any number of trials).
        # Pick the highest success rate, breaking ties by number of trials.
        # Useful to plot intermediate results before an evaluation run is finished.
        best = max(candidates, key=lambda c: (c.success_rate, c.num_trials))
        best.num_checkpoints_available = len(candidates)
        best.all_checkpoint_trials = [c.num_trials for c in candidates]
        results.append(best)

    results.sort(key=lambda r: r.horizon)
    return results


def collect_all_checkpoint_results(
    experiment_paths: Sequence[Path],
) -> Dict[str, List[CheckpointResult]]:
    """Collect all checkpoints grouped by checkpoint name across all horizons."""
    by_ckpt: Dict[str, List[CheckpointResult]] = {}
    for experiment_path in experiment_paths:
        if not experiment_path.exists():
            raise FileNotFoundError(f"Experiment path '{experiment_path}' does not exist.")

        horizon = _parse_horizon(experiment_path.name)
        if horizon is None:
            continue

        for ckpt_dir in _find_checkpoint_dirs(experiment_path):
            pkl = ckpt_dir / "results.pkl"
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

    title = plot_name or ("Execution Horizon Comparison" if len(experiments) > 1 else experiments[0][0])
    ax.set_title(title, fontsize=12, fontweight="bold", pad=10)
    ax.set_xlabel("Execution Horizon (steps)", fontsize=12)
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
        description="Plot success rates across Execution Horizons from DAgger evaluation output."
    )
    parser.add_argument("--experiment-path", type=Path, nargs="+", action="append", required=True,
                        metavar="PATH",
                        help="A list of per-horizon output directories (each with an 'ah<N>' "
                             "postfix). Repeat the flag once per experiment for multi-experiment mode.")
    parser.add_argument("--experiment-name", type=str, nargs="+", default=None,
                        help="Legend label(s), one per experiment (defaults to a shared prefix).")
    parser.add_argument("--plot-name", type=str, default=None, help="Plot title.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save the figure (PNG, PDF, etc.). Omit to skip saving.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving to disk.")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Plot every checkpoint instead of just the best per horizon.")
    return parser.parse_args()


def _default_label(paths: Sequence[Path]) -> str:
    """Derive a legend label from a list of per-horizon directories.

    Uses the directory name with the ``_ah<N>`` postfix stripped, since that is
    what distinguishes horizons within a single experiment.
    """
    if not paths:
        return "experiment"
    name = paths[0].name
    return re.sub(r"_?ah\d+$", "", name) or name


def main() -> None:
    args = parse_args()

    # args.experiment_path is a list of lists (one list per experiment).
    exp_path_lists: List[List[Path]] = args.experiment_path
    if args.experiment_name is None:
        exp_labels = [_default_label(paths) for paths in exp_path_lists]
    else:
        if len(args.experiment_name) != len(exp_path_lists):
            raise ValueError(
                f"--experiment-name count ({len(args.experiment_name)}) must match "
                f"the number of --experiment-path lists ({len(exp_path_lists)})"
            )
        exp_labels = args.experiment_name

    experiments: List[Tuple[str, Sequence[CheckpointResult], str]] = []

    if args.all_checkpoints:
        if len(exp_path_lists) > 1:
            raise ValueError("--all-checkpoints is only supported for a single --experiment-path list")
        by_ckpt = collect_all_checkpoint_results(exp_path_lists[0])
        if not by_ckpt:
            raise RuntimeError(f"No valid checkpoints found under {exp_path_lists[0]}.")
        palette = _generate_color_palette(len(by_ckpt))
        for idx, (ckpt_name, results) in enumerate(sorted(by_ckpt.items())):
            experiments.append((ckpt_name, results, palette[idx]))
            print(f"\n{ckpt_name}:")
            for r in results:
                print(f"  ah{r.horizon}: success_rate={r.success_rate:.3f} ({r.num_trials} trials)")
    else:
        palette = [NAVY] if len(exp_path_lists) == 1 else _generate_color_palette(len(exp_path_lists))
        for idx, (paths, label) in enumerate(zip(exp_path_lists, exp_labels)):
            existing = [p for p in paths if p.exists()]
            for p in paths:
                if not p.exists():
                    print(f"\n{'!'*60}")
                    print(f"WARNING: experiment path does not exist, skipping:")
                    print(f"  {p}")
                    print(f"{'!'*60}")
            if not existing:
                print(f"Warning: no valid experiment paths found for '{label}'. Skipping.")
                continue
            results = collect_best_results(existing)
            if not results:
                print(f"Warning: no valid results.pkl files found for '{label}'. Skipping.")
                continue
            experiments.append((label, results, palette[idx]))
            print(f"\n{label} — best checkpoint per horizon:")
            for r in results:
                if r.num_checkpoints_available > 1:
                    trials_str = ", ".join(str(t) for t in r.all_checkpoint_trials)
                    n_tag = f" [{r.num_checkpoints_available} ckpts: {trials_str} trials]"
                else:
                    n_tag = ""
                print(f"  ah{r.horizon}: {r.success_rate:.3f} ({r.num_trials} trials) -> {r.checkpoint_dir}{n_tag}")

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
