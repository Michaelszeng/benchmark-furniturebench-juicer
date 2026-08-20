"""
Plot success rates across Execution Horizons from evaluate_checkpoints.sh output.

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

    Human Expert - Context Length Ablation:
        python src/eval/plot_eval_results.py \
            --experiment-path outputs/2_obs_one_leg_teleop_r3m_cross_attn_double_enc outputs/4_obs_one_leg_teleop_r3m_double_enc outputs/8_obs_one_leg_teleop_r3m_double_enc outputs/12_obs_one_leg_teleop_r3m_double_enc outputs/16_obs_one_leg_teleop_r3m_double_enc outputs/20_obs_one_leg_teleop_r3m_double_enc \
            --experiment-name '$T_o=2$' '$T_o=4$' '$T_o=8$' '$T_o=12$' '$T_o=16$' '$T_o=20$' \
            --plot-name "Human Expert ($\mathbf{N}_{\mathbf{\mathrm{demo}}}=200$) - Context Length Ablation" \
            --fig-width 4 \
            --output outputs/plots/comparison_context_length_double_enc_human_expert_one_leg_v2.png

    Non-Markovian Expert - Context Length Ablation:
        python src/eval/plot_eval_results.py \
            --experiment-path outputs/2_obs_one_leg_teleop_r3m_cross_attn_double_enc_scripted_non_markovian_1000_eps outputs/4_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps outputs/8_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps outputs/12_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps outputs/16_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps outputs/20_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps \
            --experiment-name '$T_o=2$' '$T_o=4$' '$T_o=8$' '$T_o=12$' '$T_o=16$' '$T_o=20$' \
            --plot-name "Non-Markovian Expert ($\mathbf{N}_{\mathbf{\mathrm{demo}}}=1000$) - Context Length Ablation" \
            --legend-loc upper-right \
            --fig-width 4 \
            --y-min 0.75 \
            --output outputs/plots/comparison_context_length_double_enc_non_markovian_1000_expert_one_leg_v2.png

    python src/eval/plot_eval_results.py \
    --experiment-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_5_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_25_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_0625_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_03125_200_eps \
    --plot-name 'α=0.1' \
    --experiment-name '$\sigma=1$' '$\sigma=\frac{1}{2}$' '$\sigma=\frac{1}{4}$' '$\sigma=\frac{1}{8}$' '$\sigma=\frac{1}{16}$' \
    --output outputs/plots/noise_inkection_ablation_alpha_0_1.png

    python src/eval/plot_eval_results.py \
    --experiment-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_25_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_25_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_3_noise_0_25_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_4_noise_0_25_200_eps \
    --plot-name 'σ=0.25' \
    --experiment-name 'α=0.1' 'α=0.2' 'α=0.3' 'α=0.4' \
    --output outputs/plots/noise_injection_ablation_sigma_0_25.png

    python src/eval/plot_eval_results.py \
    --experiment-path outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_noise_0_200_eps \
    --plot-name '1-obs' \
    --output outputs/plots/1_obs_test.png

    python src/eval/plot_eval_results.py \
    --experiment-path /data/locomotion/michzeng/ManiSkill/outputs/2_obs_human_expert_attention_double_enc /data/locomotion/michzeng/ManiSkill/outputs/4_obs_human_expert_attention_double_enc /data/locomotion/michzeng/ManiSkill/outputs/8_obs_human_expert_attention_double_enc /data/locomotion/michzeng/ManiSkill/outputs/12_obs_human_expert_attention_double_enc /data/locomotion/michzeng/ManiSkill/outputs/16_obs_human_expert_attention_double_enc /data/locomotion/michzeng/ManiSkill/outputs/20_obs_human_expert_attention_double_enc \
    --experiment-name '$T_o=2$' '$T_o=4$' '$T_o=8$' '$T_o=12$' '$T_o=16$' '$T_o=20$' \
    --legend-loc upper-right \
    --fig-width 3.2 \
    --fig-height 2.0 \
    --output outputs/plots/comparison_context_length_double_enc_pusht_m.png

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
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, ScalarFormatter

# Use DejaVu Sans Mono for all text in the figure.
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["DejaVu Sans Mono"]
# Use Computer Modern for math (same as LaTeX default), applied to $...$ strings.
plt.rcParams["mathtext.fontset"] = "cm"

# -----------------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------------

NAVY = "#2f5fb3"
DARK_RED = "#c0392b"
GRID_COLOR = "#bdbdbd"

# -----------------------------------------------------------------------------
# CoRL pre-print layout
# -----------------------------------------------------------------------------
# Sized to sit comfortably in a single CoRL column with ~8-10 pt in-figure text.

FIG_WIDTH_IN = 3.4
FIG_HEIGHT_IN = 2.6

AXIS_TITLE_FS = 9    # x/y axis titles
TITLE_FS = 10        # figure title
TICK_FS = 7          # tick labels (kept at the ~7 pt readability floor)
LEGEND_FS = 7        # legend labels


def _generate_color_palette(num_colors: int) -> List[str]:
    """Interpolate a gradient from navy (baseline) to dark red.

    The first color is always navy so the earliest-submitted checkpoint keeps
    the canonical "baseline" color.
    """
    if num_colors <= 0:
        return []
    if num_colors == 1:
        return [NAVY]

    r0, g0, b0 = (int(NAVY[i:i + 2], 16) for i in (1, 3, 5))
    r1, g1, b1 = (int(DARK_RED[i:i + 2], 16) for i in (1, 3, 5))
    colors = []
    for i in range(num_colors):
        t = i / (num_colors - 1)
        r = round(r0 + (r1 - r0) * t)
        g = round(g0 + (g1 - g0) * t)
        b = round(b0 + (b1 - b0) * t)
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
    """Return (success_rate, num_trials) from a results.pkl or summary.pkl file.

    Supports two formats:
    - Standard format: {"n_total": int, "n_success": int, ...}
    - Push-T summary format: {"trial_result": ["success"|"failure", ...], ...}
    """
    with results_path.open("rb") as f:
        data = pickle.load(f)
    if "trial_result" in data:
        trial_results = data["trial_result"]
        n_total = len(trial_results)
        n_success = sum(1 for r in trial_results if r == "success")
    else:
        n_total = data.get("n_total", 0)
        n_success = data.get("n_success", 0)
    if n_total == 0:
        return float("nan"), 0
    return n_success / n_total, n_total


def _parse_horizon(dirname: str) -> Optional[int]:
    """Parse integer horizon from a directory name.

    Supports two naming conventions:
    - Standard: 'T_a_8'
    - IsaacLab: 'ah8'
    """
    if dirname.startswith("T_a_"):
        try:
            return int(dirname[4:])
        except ValueError:
            return None
    if dirname.startswith("ah"):
        try:
            return int(dirname[2:])
        except ValueError:
            return None
    return None


def _is_checkpoint_dir(dirname: str) -> bool:
    """Return True if *dirname* is a checkpoint directory named 'epoch=*'."""
    return dirname.startswith("epoch=")


def _find_results_pkl(directory: Path) -> Optional[Path]:
    """Return the first results pkl found in *directory*, or None."""
    for name in ("results.pkl", "summary.pkl"):
        p = directory / name
        if p.exists():
            return p
    return None


def collect_best_results(experiment_path: Path) -> List[CheckpointResult]:
    """For each horizon sub-directory, find the best checkpoint by success rate.

    Supports two directory layouts:

    Standard layout (horizon outer, checkpoint inner):
        <experiment_path>/T_a_<N>/<checkpoint>/results.pkl

    Inverted layout (checkpoint outer, horizon inner — used by IsaacLab):
        <experiment_path>/<checkpoint>/ah<N>/results.pkl
    """
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path '{experiment_path}' does not exist.")

    top_dirs = [d for d in sorted(experiment_path.iterdir()) if d.is_dir()]

    # Detect layout: if any top-level dir parses as a horizon → standard layout.
    if any(_parse_horizon(d.name) is not None for d in top_dirs):
        return _collect_standard_layout(experiment_path, top_dirs)
    else:
        return _collect_inverted_layout(experiment_path, top_dirs)


def _collect_standard_layout(
    experiment_path: Path, top_dirs: List[Path]
) -> List[CheckpointResult]:
    """Standard layout: <experiment_path>/T_a_<N>/<checkpoint>/results.pkl"""
    results: List[CheckpointResult] = []
    for horizon_dir in top_dirs:
        horizon = _parse_horizon(horizon_dir.name)
        if horizon is None:
            continue

        candidates: List[CheckpointResult] = []
        for ckpt_dir in sorted(horizon_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            if not _is_checkpoint_dir(ckpt_dir.name):
                continue
            pkl = _find_results_pkl(ckpt_dir)
            if pkl is None:
                continue
            rate, total = _load_results_pkl(pkl)
            if math.isnan(rate):
                continue
            candidates.append(CheckpointResult(horizon, rate, total, ckpt_dir))

        if not candidates:
            print(f"  Skipping {horizon_dir.name}: no valid results.pkl found.")
            continue

        max_trials = max(c.num_trials for c in candidates)
        top = [c for c in candidates if c.num_trials == max_trials]
        best = max(top, key=lambda c: c.success_rate)
        best.num_checkpoints_available = len(candidates)
        best.all_checkpoint_trials = [c.num_trials for c in candidates]
        results.append(best)

    results.sort(key=lambda r: r.horizon)
    return results


def _collect_inverted_layout(
    experiment_path: Path, top_dirs: List[Path]
) -> List[CheckpointResult]:
    """Inverted layout: <experiment_path>/<checkpoint>/ah<N>/results.pkl"""
    # Collect all (horizon, rate, total, ckpt_dir) across all checkpoint dirs.
    by_horizon: Dict[int, List[CheckpointResult]] = {}
    for ckpt_dir in top_dirs:
        if not _is_checkpoint_dir(ckpt_dir.name):
            continue
        for horizon_dir in sorted(ckpt_dir.iterdir()):
            if not horizon_dir.is_dir():
                continue
            horizon = _parse_horizon(horizon_dir.name)
            if horizon is None:
                continue
            pkl = _find_results_pkl(horizon_dir)
            if pkl is None:
                continue
            rate, total = _load_results_pkl(pkl)
            if math.isnan(rate):
                continue
            by_horizon.setdefault(horizon, []).append(
                CheckpointResult(horizon, rate, total, horizon_dir)
            )

    results: List[CheckpointResult] = []
    for horizon, candidates in sorted(by_horizon.items()):
        if not candidates:
            continue
        max_trials = max(c.num_trials for c in candidates)
        top = [c for c in candidates if c.num_trials == max_trials]
        best = max(top, key=lambda c: c.success_rate)
        best.num_checkpoints_available = len(candidates)
        best.all_checkpoint_trials = [c.num_trials for c in candidates]
        results.append(best)

    return results


def collect_all_checkpoint_results(experiment_path: Path) -> Dict[str, List[CheckpointResult]]:
    """Collect all checkpoints grouped by checkpoint name across all horizons.

    Supports both the standard layout (horizon outer, checkpoint inner) and the
    inverted IsaacLab layout (checkpoint outer, horizon inner).
    """
    if not experiment_path.exists():
        raise FileNotFoundError(f"Experiment path '{experiment_path}' does not exist.")

    top_dirs = [d for d in sorted(experiment_path.iterdir()) if d.is_dir()]
    standard_layout = any(_parse_horizon(d.name) is not None for d in top_dirs)

    by_ckpt: Dict[str, List[CheckpointResult]] = {}
    if standard_layout:
        for horizon_dir in top_dirs:
            horizon = _parse_horizon(horizon_dir.name)
            if horizon is None:
                continue
            for ckpt_dir in sorted(horizon_dir.iterdir()):
                if not ckpt_dir.is_dir() or not _is_checkpoint_dir(ckpt_dir.name):
                    continue
                pkl = _find_results_pkl(ckpt_dir)
                if pkl is None:
                    continue
                rate, total = _load_results_pkl(pkl)
                if math.isnan(rate):
                    continue
                by_ckpt.setdefault(ckpt_dir.name, []).append(
                    CheckpointResult(horizon, rate, total, ckpt_dir)
                )
    else:
        for ckpt_dir in top_dirs:
            if not _is_checkpoint_dir(ckpt_dir.name):
                continue
            for horizon_dir in sorted(ckpt_dir.iterdir()):
                if not horizon_dir.is_dir():
                    continue
                horizon = _parse_horizon(horizon_dir.name)
                if horizon is None:
                    continue
                pkl = _find_results_pkl(horizon_dir)
                if pkl is None:
                    continue
                rate, total = _load_results_pkl(pkl)
                if math.isnan(rate):
                    continue
                by_ckpt.setdefault(ckpt_dir.name, []).append(
                    CheckpointResult(horizon, rate, total, horizon_dir)
                )

    for lst in by_ckpt.values():
        lst.sort(key=lambda r: r.horizon)
    return by_ckpt


# -----------------------------------------------------------------------------
# Verbose reporting
# -----------------------------------------------------------------------------

_BOLD = "\033[1m"
_RESET = "\033[0m"


def print_checkpoint_table(label: str, by_ckpt: Dict[str, List[CheckpointResult]]) -> None:
    """Print a success-rate table with T_a rows and checkpoint columns.

    The best-performing checkpoint column (highest mean success rate across
    horizons) is rendered in bold.
    """
    if not by_ckpt:
        print(f"\n{label}: no valid checkpoints found.")
        return

    ckpt_names = sorted(by_ckpt.keys())
    horizons = sorted({r.horizon for lst in by_ckpt.values() for r in lst})

    # (checkpoint, horizon) -> success_rate lookup for cell fills.
    rate_map: Dict[Tuple[str, int], float] = {}
    for name, lst in by_ckpt.items():
        for r in lst:
            rate_map[(name, r.horizon)] = r.success_rate

    # Best column = checkpoint with the highest mean success rate over its
    # available horizons.
    def _mean_rate(name: str) -> float:
        vals = [rate_map[(name, h)] for h in horizons if (name, h) in rate_map]
        return sum(vals) / len(vals) if vals else float("-inf")

    best_ckpt = max(ckpt_names, key=_mean_rate)

    # Abbreviate checkpoint names to their first 9 characters (e.g. "epoch=065")
    # to keep the columns narrow.
    disp_names = {name: name[:9] for name in ckpt_names}

    # Column widths (raw, ANSI codes are applied after padding).
    left_header = "T_a"
    left_width = max(len(left_header), max(len(str(h)) for h in horizons))
    col_widths = {name: max(len(disp_names[name]), len("0.000")) for name in ckpt_names}

    def _cell(text: str, width: int, bold: bool) -> str:
        s = text.rjust(width)
        return f"{_BOLD}{s}{_RESET}" if bold else s

    print(f"\n{label} — success rate by checkpoint (rows: T_a, cols: checkpoint):")

    header = "  " + left_header.rjust(left_width) + " │ " + "  ".join(
        _cell(disp_names[name], col_widths[name], name == best_ckpt) for name in ckpt_names
    )
    print(header)

    sep_cols = "──".join("─" * col_widths[name] for name in ckpt_names)
    print("  " + "─" * left_width + "─┼─" + sep_cols)

    for h in horizons:
        row_cells = []
        for name in ckpt_names:
            if (name, h) in rate_map:
                text = f"{rate_map[(name, h)]:.3f}"
            else:
                text = "-"
            row_cells.append(_cell(text, col_widths[name], name == best_ckpt))
        print("  " + str(h).rjust(left_width) + " │ " + "  ".join(row_cells))

    print(f"  (best column: {best_ckpt}, mean={_mean_rate(best_ckpt):.3f})")


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def make_plot(
    experiments: List[Tuple[str, Sequence[CheckpointResult], str]],
    dpi: int,
    plot_name: Optional[str] = None,
    legend_loc: Optional[str] = None,
    fig_width: Optional[float] = None,
    fig_height: Optional[float] = None,
    y_min: Optional[float] = None,
) -> plt.Figure:
    # constrained_layout snugly packs the axis titles against the axes.
    fig, ax = plt.subplots(figsize=(fig_width or FIG_WIDTH_IN, fig_height or FIG_HEIGHT_IN), layout="constrained")
    # Minimize the padding between the axes/labels and the figure edge.
    fig.get_layout_engine().set(w_pad=0.0, h_pad=0.0, wspace=0.0, hspace=0.0)
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
        ax.errorbar(horizons, rates, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=2.0, capthick=0.85, alpha=0.9, zorder=2)

        if show_ckpt_labels:
            for res in results:
                if res.num_checkpoints_available > 1:
                    name = res.checkpoint_dir.name
                    label = name[:10] + "..." if len(name) > 10 else name
                    ax.annotate(label, xy=(res.horizon, res.success_rate), xytext=(0, 5),
                                textcoords="offset points", fontsize=6, color=color,
                                ha="center", va="bottom", alpha=0.8)

    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())
    if all_horizons:
        ax.set_xticks(all_horizons)
        ax.set_xticklabels([str(h) for h in all_horizons])

    if plot_name:
        ax.set_title(plot_name, fontsize=TITLE_FS, fontweight="bold", pad=6)
    ax.set_xlabel("Execution Horizon (steps)", fontsize=AXIS_TITLE_FS)
    ax.set_ylabel("Success Rate", fontsize=AXIS_TITLE_FS)

    ax.grid(True, which="major", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", color=GRID_COLOR, linestyle="-", linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#4f4f4f")

    ax.tick_params(axis="both", which="major", direction="in", labelsize=TICK_FS, length=2.5, width=0.8)
    ax.tick_params(axis="x", which="minor", direction="in", length=1.5, width=0.6)
    ax.tick_params(axis="y", which="minor", left=False)
    # Compact, fixed-width y labels locked to 0.1 increments so single-decimal
    # labels never collide/duplicate.
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", which="major", pad=3)

    if y_min is not None:
        ax.set_ylim(bottom=y_min)

    if len(experiments) > 1:
        ax.legend(loc=legend_loc or "best", fontsize=LEGEND_FS, framealpha=0.9, edgecolor="#4f4f4f",
                  borderpad=0.3, labelspacing=0.25, handlelength=1.4, handletextpad=0.4)

    fig.set_dpi(dpi)
    return fig


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot success rates across Execution Horizons from evaluate_checkpoints.sh output."
    )
    parser.add_argument("--experiment-path", type=Path, nargs="+", required=True,
                        help="Path(s) to experiment output directories containing T_a_<N> sub-folders.")
    parser.add_argument("--experiment-name", type=str, nargs="+", default=None,
                        help="Legend label(s) for each experiment (defaults to directory names).")
    parser.add_argument("--plot-name", type=str, default=None, help="Plot title.")
    parser.add_argument(
        "--legend-loc", type=str, default=None,
        choices=["upper-left", "upper-right", "lower-left", "lower-right"],
        help="Snap the legend to a specific corner (default: matplotlib auto-placement).",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save the figure (PNG, PDF, etc.). Omit to skip saving.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving to disk.")
    parser.add_argument("--fig-width", type=float, default=None,
                        help=f"Figure width in inches (default: {FIG_WIDTH_IN}).")
    parser.add_argument("--fig-height", type=float, default=None,
                        help=f"Figure height in inches (default: {FIG_HEIGHT_IN}).")
    parser.add_argument("--y-min", type=float, default=None,
                        help="Lower limit for the y-axis (default: matplotlib auto).")
    parser.add_argument("--all-checkpoints", action="store_true",
                        help="Plot every checkpoint instead of just the best per horizon.")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print a success-rate table (T_a rows, checkpoint columns) for "
                             "every evaluated checkpoint of each experiment.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    legend_loc = {
        "upper-left": "upper left",
        "upper-right": "upper right",
        "lower-left": "lower left",
        "lower-right": "lower right",
    }.get(args.legend_loc) if args.legend_loc else None

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
        if args.verbose:
            print_checkpoint_table(exp_paths[0].name, by_ckpt)
        else:
            for ckpt_name, results in sorted(by_ckpt.items()):
                print(f"\n{ckpt_name}:")
                for r in results:
                    print(f"  T_a_{r.horizon}: success_rate={r.success_rate:.3f} ({r.num_trials} trials)")
    else:
        palette = [NAVY] if len(exp_paths) == 1 else _generate_color_palette(len(exp_paths))
        for idx, (exp_path, label) in enumerate(zip(exp_paths, exp_labels)):
            if not exp_path.exists():
                print(f"\n{'!'*60}")
                print(f"WARNING: experiment path does not exist, skipping:")
                print(f"  {exp_path}")
                print(f"{'!'*60}")
                continue
            results = collect_best_results(exp_path)
            if not results:
                print(f"Warning: no valid results.pkl files found under {exp_path}. Skipping.")
                continue
            experiments.append((label, results, palette[idx]))
            if args.verbose:
                print_checkpoint_table(label, collect_all_checkpoint_results(exp_path))
            else:
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

    fig = make_plot(experiments, dpi=args.dpi, plot_name=args.plot_name, legend_loc=legend_loc,
                    fig_width=args.fig_width, fig_height=args.fig_height, y_min=args.y_min)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.01)
        print(f"\nSaved figure to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
