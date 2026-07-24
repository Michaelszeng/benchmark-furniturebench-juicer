"""
Combine two success-rate figures side-by-side to compare two types of experts.

Each expert is plotted in its own panel with independent x- and y-axes (the
ranges may differ between panels), but the two panels share common x-axis and
y-axis titles. The left panel is drawn in navy blue and the right panel in dark
red. Wilson confidence-interval error bars are included; per-checkpoint (epoch)
labels are intentionally omitted.

Reuses the data-loading logic from src/eval/plot_eval_results.py, which scans an
output directory with the structure:
    <experiment_path>/T_a_<horizon>/<checkpoint_stem>/results.pkl

Example usage
-------------

Human vs Markovian Expert:

   python src/eval/plot_expert_comparison.py \
       --left-path outputs/2_obs_one_leg_teleop_r3m \
       --right-path outputs/2_obs_one_leg_scripted_r3m_200_eps \
       --left-name "Human Expert" \
       --right-name "Markovian Expert" \
       --left-color navy \
       --right-color dark_red \
       --output outputs/plots/comparison_human_expert_markovian_expert_one_leg.png

Non-Markovian vs Markovian Expert:

    python src/eval/plot_expert_comparison.py \
       --left-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_200_eps \
       --right-path outputs/2_obs_one_leg_scripted_r3m_200_eps \
       --left-name "Non-Markovian Expert" \
       --right-name "Markovian Expert" \
       --left-color turquoise \
       --right-color dark_red \
       --output outputs/plots/comparison_non_markovian_expert_markovian_expert_one_leg.png


Don't set --output to skip saving. Set --show to open an interactive window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.proportion as smp
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, ScalarFormatter

from src.eval.plot_eval_results import CheckpointResult, collect_best_results

# Use DejaVu Sans Mono for all text in the figure.
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["DejaVu Sans Mono"]

# -----------------------------------------------------------------------------
# Visual constants
# -----------------------------------------------------------------------------

NAVY = "#1f3b6f"
DARK_RED = "#8b1a1a"
TURQUOISE = "#118c86"
GRID_COLOR = "#bdbdbd"

# Preset color names selectable via --left-color / --right-color. Any other
# string is passed straight through as a matplotlib color (hex, named, etc.).
PRESET_COLORS = {
    "navy": NAVY,
    "dark_red": DARK_RED,
    "turquoise": TURQUOISE,
}


def _resolve_color(color: str) -> str:
    """Map a preset name to its hex value, or pass the string through unchanged."""
    return PRESET_COLORS.get(color.lower(), color)

# -----------------------------------------------------------------------------
# CoRL pre-print layout
# -----------------------------------------------------------------------------
# CoRL uses a single-column US-letter layout with a text block ~6.0 in wide and
# 10 pt body text (captions ~9 pt). Sizing the figure to the full text width and
# keeping in-figure text at ~8-10 pt keeps everything legible without scaling.

FIG_WIDTH_IN = 6.0   # full CoRL text width
FIG_HEIGHT_IN = 2.2  # two side-by-side panels at a comfortable aspect ratio

AXIS_TITLE_FS = 9    # shared x/y axis titles
PANEL_TITLE_FS = 9   # per-panel titles
SUPTITLE_FS = 10     # overall figure title
TICK_FS = 7          # tick labels (kept at the ~7 pt readability floor)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _draw_panel(
    ax: plt.Axes,
    results: Sequence[CheckpointResult],
    color: str,
    title: str,
) -> None:
    """Draw a single expert's success-rate trace with error bars onto `ax`."""
    ax.set_facecolor("white")

    if results:
        horizons = np.array([r.horizon for r in results], dtype=float)
        rates = np.array([r.success_rate for r in results], dtype=float)
        trials = np.array([r.num_trials for r in results], dtype=int)

        ci = np.array(
            [smp.proportion_confint(int(p * n), n, alpha=0.05, method="wilson") for p, n in zip(rates, trials)]
        )
        yerr = np.vstack([np.clip(rates - ci[:, 0], 0, 1), np.clip(ci[:, 1] - rates, 0, 1)])

        ax.plot(horizons, rates, color=color, linewidth=1.5, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, zorder=3)
        ax.errorbar(horizons, rates, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=2.0, capthick=0.85, alpha=0.9, zorder=2)

        panel_horizons = sorted({r.horizon for r in results})
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.set_xticks(panel_horizons)
        ax.set_xticklabels([str(h) for h in panel_horizons])

    ax.set_title(title, fontsize=PANEL_TITLE_FS, fontweight="bold", pad=3)

    ax.grid(True, which="major", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", color=GRID_COLOR, linestyle="-", linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#4f4f4f")

    ax.tick_params(axis="both", which="major", direction="in", labelsize=TICK_FS, length=2.5, width=0.8)
    ax.tick_params(axis="x", which="minor", direction="in", length=1.5, width=0.6)
    ax.tick_params(axis="y", which="minor", left=False)
    # Compact, fixed-width y labels pulled close to the axis to save horizontal space.
    # Lock ticks to 0.1 increments so single-decimal labels never collide/duplicate.
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", which="major", pad=3)


def make_comparison_plot(
    left: Sequence[CheckpointResult],
    right: Sequence[CheckpointResult],
    left_name: str,
    right_name: str,
    dpi: int,
    plot_name: Optional[str] = None,
    left_color: str = NAVY,
    right_color: str = DARK_RED,
) -> plt.Figure:
    # sharex=False / sharey=False keeps the two panels' axes fully independent.
    # constrained_layout snugly packs the shared super-labels against the axes,
    # avoiding the large gap tight_layout leaves under a supxlabel.
    fig, (ax_left, ax_right) = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=False, sharey=False,
        layout="constrained",
    )
    # Trim the padding between the axes/labels and the figure edge.
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)

    _draw_panel(ax_left, left, _resolve_color(left_color), left_name)
    _draw_panel(ax_right, right, _resolve_color(right_color), right_name)

    # Shared axis titles: one x-label centered under both panels, one y-label at left.
    fig.supxlabel("Execution Horizon (steps)", fontsize=AXIS_TITLE_FS)
    fig.supylabel("Success Rate", fontsize=AXIS_TITLE_FS)

    if plot_name:
        fig.suptitle(plot_name, fontsize=SUPTITLE_FS, fontweight="bold")

    fig.set_dpi(dpi)
    return fig


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare two experts side-by-side with independent axes but shared axis titles."
    )
    parser.add_argument("--left-path", type=Path, required=True,
                        help="Path to the left experiment's output directory (T_a_<N> sub-folders).")
    parser.add_argument("--right-path", type=Path, required=True,
                        help="Path to the right experiment's output directory (T_a_<N> sub-folders).")
    parser.add_argument("--left-name", type=str, default=None,
                        help="Title for the left panel (defaults to directory name).")
    parser.add_argument("--right-name", type=str, default=None,
                        help="Title for the right panel (defaults to directory name).")
    preset_help = f"presets: {', '.join(PRESET_COLORS)}; or any matplotlib color"
    parser.add_argument("--left-color", type=str, default=NAVY,
                        help=f"Color for the left panel ({preset_help}; default {NAVY}).")
    parser.add_argument("--right-color", type=str, default=DARK_RED,
                        help=f"Color for the right panel ({preset_help}; default {DARK_RED}).")
    parser.add_argument("--plot-name", type=str, default=None, help="Overall figure title.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save the figure (PNG, PDF, etc.). Omit to skip saving.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving to disk.")
    return parser.parse_args()


def _load(exp_path: Path, label: str) -> Sequence[CheckpointResult]:
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment path '{exp_path}' does not exist.")
    results = collect_best_results(exp_path)
    if not results:
        raise RuntimeError(f"No valid results.pkl files found under {exp_path}.")
    print(f"\n{label} — best checkpoint per horizon:")
    for r in results:
        print(f"  T_a_{r.horizon}: {r.success_rate:.3f} ({r.num_trials} trials)")
    return results


def main() -> None:
    args = parse_args()

    left_name = args.left_name or args.left_path.name
    right_name = args.right_name or args.right_path.name

    left_results = _load(args.left_path, left_name)
    right_results = _load(args.right_path, right_name)

    fig = make_comparison_plot(
        left_results, right_results, left_name, right_name, dpi=args.dpi, plot_name=args.plot_name,
        left_color=args.left_color, right_color=args.right_color,
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight", pad_inches=0.02)
        print(f"\nSaved figure to {args.output}")

    if args.show:
        plt.show()
    else:
        plt.close(fig)


if __name__ == "__main__":
    main()
