"""
Combine two DAgger success-rate figures side-by-side to compare two experiments.

Formatting matches src/eval/plot_expert_comparison.py; data loading is reused
from src/eval/plot_dagger_results.py.

Example usage
-------------

Non-Markovian (left, navy) vs Markovian (right, red), each overlaying DAgger iterations iter-1..iter2:

   python src/eval/plot_dagger_comparison.py \
       --left-path \
           outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter-1_ah1 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter-1_ah3 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter-1_ah6 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter-1_ah10 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter-1_ah15 \
       --left-path \
           outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah1 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah3 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah6 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah10 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter0_ah15 \
       --left-path \
           outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah1 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah3 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah6 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah10 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter1_ah15 \
       --left-path \
           outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter2_ah1 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter2_ah3 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter2_ah6 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter2_ah10 outputs/2_obs_one_leg_scripted_r3m_non_markovian_dagger_iter2_ah15 \
       --right-path \
           outputs/2_obs_one_leg_scripted_r3m_dagger_iter-1_ah1 outputs/2_obs_one_leg_scripted_r3m_dagger_iter-1_ah3 outputs/2_obs_one_leg_scripted_r3m_dagger_iter-1_ah6 outputs/2_obs_one_leg_scripted_r3m_dagger_iter-1_ah10 outputs/2_obs_one_leg_scripted_r3m_dagger_iter-1_ah15 \
       --right-path \
           outputs/2_obs_one_leg_scripted_r3m_dagger_iter0_ah1 outputs/2_obs_one_leg_scripted_r3m_dagger_iter0_ah3 outputs/2_obs_one_leg_scripted_r3m_dagger_iter0_ah6 outputs/2_obs_one_leg_scripted_r3m_dagger_iter0_ah10 outputs/2_obs_one_leg_scripted_r3m_dagger_iter0_ah15 \
       --right-path \
           outputs/2_obs_one_leg_scripted_r3m_dagger_iter1_ah1 outputs/2_obs_one_leg_scripted_r3m_dagger_iter1_ah3 outputs/2_obs_one_leg_scripted_r3m_dagger_iter1_ah6 outputs/2_obs_one_leg_scripted_r3m_dagger_iter1_ah10 outputs/2_obs_one_leg_scripted_r3m_dagger_iter1_ah15 \
       --right-path \
           outputs/2_obs_one_leg_scripted_r3m_dagger_iter2_ah1 outputs/2_obs_one_leg_scripted_r3m_dagger_iter2_ah3 outputs/2_obs_one_leg_scripted_r3m_dagger_iter2_ah6 outputs/2_obs_one_leg_scripted_r3m_dagger_iter2_ah10 outputs/2_obs_one_leg_scripted_r3m_dagger_iter2_ah15 \
       --left-name "Scripted Non-Markovian Expert" \
       --right-name "Scripted Markovian Expert" \
       --labels "Baseline" "iter 1" "iter 2" "iter 3" \
       --reverse-legend \
       --output outputs/plots/comparison_dagger_non_markovian_vs_markovian_one_leg_v2.png

Don't set --output to skip saving. Set --show to open an interactive window.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.proportion as smp
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, ScalarFormatter

from src.eval.plot_dagger_results import (
    CheckpointResult,
    _default_label,
    collect_best_results,
)

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


def _hex_to_rgb(hex_color: str) -> Tuple[float, float, float]:
    h = hex_color.lstrip("#")
    return tuple(int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))  # type: ignore[return-value]


def _rgb_to_hex(rgb: Sequence[float]) -> str:
    return "#" + "".join(f"{int(round(max(0.0, min(1.0, c)) * 255)):02x}" for c in rgb)


def _shade_palette(base_color: str, n: int) -> List[str]:
    """Return `n` shades of `base_color`, from a light tint to the full color.

    The last (darkest) entry is the base color itself, so a single trace uses
    exactly the requested navy / dark-red.
    """
    base_rgb = _hex_to_rgb(_resolve_color(base_color))
    if n <= 1:
        return [_rgb_to_hex(base_rgb)]
    shades = []
    for t in np.linspace(0.4, 1.0, n):
        shades.append(_rgb_to_hex([1.0 * (1 - t) + base_rgb[j] * t for j in range(3)]))
    return shades

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
LEGEND_FS = 6        # per-panel legend


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _draw_panel(
    ax: plt.Axes,
    traces: Sequence[Tuple[str, Sequence[CheckpointResult]]],
    base_color: str,
    title: str,
    reverse_legend: bool = False,
) -> None:
    """Draw one panel that may overlay several success-rate traces onto `ax`."""
    ax.set_facecolor("white")

    colors = _shade_palette(base_color, len(traces))
    panel_horizons = sorted({r.horizon for _, results in traces for r in results})

    for (label, results), color in zip(traces, colors):
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
                markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
        ax.errorbar(horizons, rates, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=2.0, capthick=0.85, alpha=0.9, zorder=2)

    if panel_horizons:
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

    if len(traces) > 1:
        handles, labels = ax.get_legend_handles_labels()
        if reverse_legend:
            handles, labels = handles[::-1], labels[::-1]
        ax.legend(handles, labels, loc="upper right", fontsize=LEGEND_FS, framealpha=0.9,
                  edgecolor="#4f4f4f", handlelength=1.4, handletextpad=0.4, labelspacing=0.3,
                  borderpad=0.4, borderaxespad=0.3)


def make_comparison_plot(
    left: Sequence[Tuple[str, Sequence[CheckpointResult]]],
    right: Sequence[Tuple[str, Sequence[CheckpointResult]]],
    left_name: str,
    right_name: str,
    dpi: int,
    plot_name: Optional[str] = None,
    left_color: str = NAVY,
    right_color: str = DARK_RED,
    reverse_legend: bool = False,
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

    _draw_panel(ax_left, left, left_color, left_name, reverse_legend=reverse_legend)
    _draw_panel(ax_right, right, right_color, right_name, reverse_legend=reverse_legend)

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
        description="Compare two DAgger experiments side-by-side with independent axes but shared axis titles."
    )
    parser.add_argument("--left-path", type=Path, nargs="+", action="append", required=True, metavar="PATH",
                        help="One trace for the left panel: a list of per-horizon dirs (each with an 'ah<N>' "
                             "postfix). Repeat the flag once per trace (e.g. per DAgger iteration).")
    parser.add_argument("--right-path", type=Path, nargs="+", action="append", required=True, metavar="PATH",
                        help="One trace for the right panel: a list of per-horizon dirs. Repeat per trace.")
    parser.add_argument("--left-name", type=str, default=None, help="Title for the left panel.")
    parser.add_argument("--right-name", type=str, default=None, help="Title for the right panel.")
    parser.add_argument("--labels", type=str, nargs="+", default=None,
                        help="Legend labels applied to both panels' traces (one per trace). "
                             "Overridden by --left-label / --right-label when those are given.")
    parser.add_argument("--left-label", type=str, nargs="+", default=None,
                        help="Legend labels for the left panel's traces (one per --left-path).")
    parser.add_argument("--right-label", type=str, nargs="+", default=None,
                        help="Legend labels for the right panel's traces (one per --right-path).")
    preset_help = f"presets: {', '.join(PRESET_COLORS)}; or any matplotlib color"
    parser.add_argument("--left-color", type=str, default=NAVY,
                        help=f"Base color for the left panel's shade family ({preset_help}; default {NAVY}).")
    parser.add_argument("--right-color", type=str, default=DARK_RED,
                        help=f"Base color for the right panel's shade family ({preset_help}; default {DARK_RED}).")
    parser.add_argument("--reverse-legend", action="store_true",
                        help="Flip the order of legend entries in both panels.")
    parser.add_argument("--plot-name", type=str, default=None, help="Overall figure title.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save the figure (PNG, PDF, etc.). Omit to skip saving.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving to disk.")
    return parser.parse_args()


def _iter_label(paths: Sequence[Path]) -> str:
    """Derive a compact legend label from a trace's directories (e.g. 'iter1')."""
    for p in paths:
        match = re.search(r"iter(-?\d+)", p.name)
        if match:
            return f"iter{match.group(1)}"
    return _default_label(paths)


def _load_panel(
    trace_paths: Sequence[Sequence[Path]],
    labels: Optional[Sequence[str]],
    panel_name: str,
) -> List[Tuple[str, Sequence[CheckpointResult]]]:
    if labels is not None and len(labels) != len(trace_paths):
        raise ValueError(
            f"{panel_name}: number of labels ({len(labels)}) must match number of "
            f"traces ({len(trace_paths)})"
        )

    traces: List[Tuple[str, Sequence[CheckpointResult]]] = []
    for idx, paths in enumerate(trace_paths):
        label = labels[idx] if labels is not None else _iter_label(paths)

        existing = [p for p in paths if p.exists()]
        for p in paths:
            if not p.exists():
                print(f"\n{'!'*60}")
                print(f"WARNING: experiment path does not exist, skipping:")
                print(f"  {p}")
                print(f"{'!'*60}")
        if not existing:
            print(f"Warning: no valid experiment paths found for '{panel_name}/{label}'. Skipping.")
            continue

        results = collect_best_results(existing)
        if not results:
            print(f"Warning: no valid results.pkl files found for '{panel_name}/{label}'. Skipping.")
            continue

        traces.append((label, results))
        print(f"\n{panel_name} / {label} — best checkpoint per horizon:")
        for r in results:
            if r.num_checkpoints_available > 1:
                trials_str = ", ".join(str(t) for t in r.all_checkpoint_trials)
                n_tag = f" [{r.num_checkpoints_available} ckpts: {trials_str} trials]"
            else:
                n_tag = ""
            print(f"  ah{r.horizon}: {r.success_rate:.3f} ({r.num_trials} trials) -> {r.checkpoint_dir}{n_tag}")

    if not traces:
        raise RuntimeError(f"No valid traces to plot for panel '{panel_name}'.")
    return traces


def main() -> None:
    args = parse_args()

    # args.left_path / args.right_path are lists of lists (one inner list per trace).
    left_name = args.left_name or "Left"
    right_name = args.right_name or "Right"

    left_labels = args.left_label or args.labels
    right_labels = args.right_label or args.labels

    left = _load_panel(args.left_path, left_labels, left_name)
    right = _load_panel(args.right_path, right_labels, right_name)

    fig = make_comparison_plot(
        left, right, left_name, right_name, dpi=args.dpi, plot_name=args.plot_name,
        left_color=args.left_color, right_color=args.right_color,
        reverse_legend=args.reverse_legend,
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
