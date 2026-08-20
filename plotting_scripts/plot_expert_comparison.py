"""
Combine two success-rate figures side-by-side to compare two types of experts.

Each expert panel is drawn with independent x- and y-axes. When a single path
is supplied per side the panel title is the experiment name; when multiple paths
are supplied, each trace is labelled via a per-panel legend and ``--left-name``
/ ``--right-name`` are used as the panel titles. Wilson confidence-interval
error bars are included; per-checkpoint (epoch) labels are intentionally omitted.

Reuses the data-loading logic from src/eval/plot_eval_results.py, which scans an
output directory with the structure:
    <experiment_path>/T_a_<horizon>/<checkpoint_stem>/results.pkl

``--experiment-name`` lists names for all paths in order: left paths first,
then right paths. Omit it to fall back to directory names.

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

Data Ablations (Non-Markovian & Markovian Expert):

   python src/eval/plot_expert_comparison.py \
    --left-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_100_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_400_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_1000_eps \
    --right-path outputs/2_obs_one_leg_scripted_r3m_100_eps outputs/2_obs_one_leg_scripted_r3m_200_eps outputs/2_obs_one_leg_scripted_r3m_400_eps outputs/2_obs_one_leg_scripted_r3m_1000_eps \
    --left-name "Scripted Non-Markovian Expert" \
    --right-name "Scripted Markovian Expert" \
    --experiment-name "100" "200" "400" "1000" "100" "200" "400" "1000" \
    --shared-legend \
    --legend-title '$\mathbf{N}_{\mathbf{\mathrm{demo}}}$' \
    --legend-reverse \
    --output outputs/plots/figure_data_ablation.png
    
Noise Injection

    python src/eval/plot_expert_comparison.py \
    --left-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_03125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_0625_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_25_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_5_200_eps \
    --right-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_3_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_4_noise_0_125_200_eps \
    --left-name 'α=0.8' \
    --right-name 'σ=1/4' \
    --experiment-name '$\sigma=0$' '$\sigma=^{1}\!/\!_{16}$' '$\sigma=^{1}\!/\!_{8}$' '$\sigma=^{1}\!/\!_{4}$' '$\sigma=^{1}\!/\!_{2}$' '$\sigma=1$' '$\alpha=1.0$' '$\alpha=0.9$' '$\alpha=0.8$' '$\alpha=0.7$' '$\alpha=0.6$' \
    --legend-loc 'lower right' \
    --legend-reverse-left \
    --output outputs/plots/figure_noise_injection_ablation_v2.png

    python src/eval/plot_expert_comparison.py \
    --left-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_0625_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_25_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_5_200_eps \
    --right-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_3_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_4_noise_0_125_200_eps \
    --left-name 'α=0.8' \
    --right-name 'σ=1/4' \
    --experiment-name '$\sigma=0$' '$\sigma=^{1}\!/\!_{8}$' '$\sigma=^{1}\!/\!_{4}$' '$\sigma=^{1}\!/\!_{2}$' '$\sigma=1$' '$\alpha=1.0$' '$\alpha=0.9$' '$\alpha=0.8$' '$\alpha=0.7$' '$\alpha=0.6$' \
    --legend-loc 'lower right' \
    --legend-reverse-left \
    --output outputs/plots/figure_noise_injection_ablation_v4.png

    python src/eval/plot_expert_comparison.py \
    --left-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_0625_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_25_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_5_200_eps \
    --right-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_3_noise_0_125_200_eps outputs/2_obs_one_leg_scripted_r3m_non_markovian_alpha_0_4_noise_0_125_200_eps \
    --left-name 'α=0.8' \
    --right-name 'σ=1/4' \
    --experiment-name '$\sigma=^{1}\!/\!_{8}$' '$\sigma=^{1}\!/\!_{4}$' '$\sigma=^{1}\!/\!_{2}$' '$\sigma=1$' '$\alpha=0.9$' '$\alpha=0.8$' '$\alpha=0.7$' '$\alpha=0.6$' \
    --legend-loc 'lower right' \
    --legend-reverse-left \
    --output outputs/plots/figure_noise_injection_ablation_v5.png

    python src/eval/plot_expert_comparison.py \
    --left-path outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_noise_0_200_eps outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_0625_200_eps outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_125_200_eps outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_25_200_eps outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_5_200_eps \
    --right-path outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_noise_0_200_eps outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_1_noise_0_125_200_eps outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_2_noise_0_125_200_eps outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_3_noise_0_125_200_eps outputs/1_obs_one_leg_scripted_r3m_non_markovian_alpha_0_4_noise_0_125_200_eps \
    --left-name 'α=0.8' \
    --right-name 'σ=1/4' \
    --experiment-name '$\sigma=0$' '$\sigma=^{1}\!/\!_{8}$' '$\sigma=^{1}\!/\!_{4}$' '$\sigma=^{1}\!/\!_{2}$' '$\sigma=1$' '$\alpha=1.0$' '$\alpha=0.9$' '$\alpha=0.8$' '$\alpha=0.7$' '$\alpha=0.6$' \
    --legend-loc 'lower right' \
    --legend-reverse-left \
    --output outputs/plots/figure_noise_injection_ablation_v6.png



Double Encoder vs No Double Encoder:

    python src/eval/plot_expert_comparison.py \
    --left-path outputs/2_obs_one_leg_teleop_r3m_cross_attn_double_enc outputs/4_obs_one_leg_teleop_r3m_double_enc outputs/8_obs_one_leg_teleop_r3m_double_enc outputs/12_obs_one_leg_teleop_r3m_double_enc outputs/16_obs_one_leg_teleop_r3m_double_enc outputs/20_obs_one_leg_teleop_r3m_double_enc \
    --right-path outputs/2_obs_one_leg_teleop_r3m_cross_attn outputs/4_obs_one_leg_teleop_r3m outputs/8_obs_one_leg_teleop_r3m outputs/12_obs_one_leg_teleop_r3m outputs/16_obs_one_leg_teleop_r3m outputs/20_obs_one_leg_teleop_r3m \
    --left-name 'Double Encoder with Cross Attention' \
    --right-name 'Cross Attention Only' \
    --experiment-name '$T_o=2$' '$T_o=4$' '$T_o=8$' '$T_o=12$' '$T_o=16$' '$T_o=20$' '$T_o=2$' '$T_o=4$' '$T_o=8$' '$T_o=12$' '$T_o=16$' '$T_o=20$' \
    --shared-legend \
    --legend-title '$\mathbf{T}_{\mathbf{o}}$' \
    --legend-reverse \
    --output outputs/plots/comparison_double_encoder_vs_no_double_encoder.png

Non-Markovian Expert Context Ablation 200 vs 1000 demonstrations:

    python src/eval/plot_expert_comparison.py \
    --left-path outputs/2_obs_one_leg_teleop_r3m_cross_attn_double_enc_scripted_non_markovian_200_eps outputs/4_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_200_eps outputs/8_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_200_eps outputs/12_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_200_eps outputs/16_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_200_eps outputs/20_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_200_eps \
    --left-name "$\mathbf{N}_{\mathbf{\mathrm{demo}}}=200$" \
    --right-path outputs/2_obs_one_leg_teleop_r3m_cross_attn_double_enc_scripted_non_markovian_1000_eps outputs/4_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps outputs/8_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps outputs/12_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps outputs/16_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps outputs/20_obs_one_leg_teleop_r3m_double_enc_scripted_non_markovian_1000_eps \
    --right-name "$\mathbf{N}_{\mathbf{\mathrm{demo}}}=1000$" \
    --experiment-name '$T_o=2$' '$T_o=4$' '$T_o=8$' '$T_o=12$' '$T_o=16$' '$T_o=20$' '$T_o=2$' '$T_o=4$' '$T_o=8$' '$T_o=12$' '$T_o=16$' '$T_o=20$' \
    --shared-legend \
    --legend-title '$\mathbf{T}_{\mathbf{o}}$' \
    --legend-reverse \
    --output outputs/plots/comparison_context_length_double_enc_non_markovian_200_vs_1000_expert_one_leg.png


Don't set --output to skip saving. Set --show to open an interactive window.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.proportion as smp
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, ScalarFormatter

from src.eval.plot_eval_results import (
    CheckpointResult,
    collect_best_results,
    _generate_color_palette,
)

# Use DejaVu Sans Mono for all text in the figure.
plt.rcParams["font.family"] = "monospace"
plt.rcParams["font.monospace"] = ["DejaVu Sans Mono"]
# Use Computer Modern for math (same as LaTeX default), applied to $...$ strings.
plt.rcParams["mathtext.fontset"] = "cm"

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
LEGEND_FS = 7        # legend labels (only shown when multiple traces per panel)
LEGEND_TITLE_FS = 9  # shared legend title


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

def _draw_panel(
    ax: plt.Axes,
    experiments: Sequence[Tuple[str, Sequence[CheckpointResult], str]],
    title: str,
    show_legend: Optional[bool] = None,
    legend_loc: str = "best",
    legend_reverse: bool = False,
) -> None:
    """Draw one or more experiment traces onto *ax*.

    Parameters
    ----------
    experiments:
        Sequence of ``(label, results, color)`` tuples.
    title:
        Panel title drawn above the axes.
    show_legend:
        Whether to draw a per-panel legend. Defaults to ``True`` when more than
        one experiment is present, ``False`` otherwise. Pass ``False`` explicitly
        to suppress the legend even with multiple traces (e.g. when a shared
        figure-level legend is used instead).
    legend_loc:
        Matplotlib legend location for the per-panel legend (e.g. ``"best"``,
        ``"lower left"``). Defaults to ``"best"``.
    legend_reverse:
        Reverse the order of entries in the per-panel legend. Defaults to
        ``False``.
    """
    ax.set_facecolor("white")

    if show_legend is None:
        show_legend = len(experiments) > 1

    for label, results, color in experiments:
        if not results:
            continue

        horizons = np.array([r.horizon for r in results], dtype=float)
        rates = np.array([r.success_rate for r in results], dtype=float)
        trials = np.array([r.num_trials for r in results], dtype=int)

        ci = np.array(
            [smp.proportion_confint(int(p * n), n, alpha=0.05, method="wilson") for p, n in zip(rates, trials)]
        )
        yerr = np.vstack([np.clip(rates - ci[:, 0], 0, 1), np.clip(ci[:, 1] - rates, 0, 1)])

        # Always assign a label so handles are available for a figure-level legend.
        ax.plot(horizons, rates, color=color, linewidth=1.5, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=label, zorder=3)
        ax.errorbar(horizons, rates, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=2.0, capthick=0.85, alpha=0.9, zorder=2)

    # Set x-ticks to the union of all horizons across all traces.
    all_horizons = sorted({r.horizon for _, results, _ in experiments for r in results})
    if all_horizons:
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.set_xticks(all_horizons)
        ax.set_xticklabels([str(h) for h in all_horizons])

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

    if show_legend:
        handles, labels = ax.get_legend_handles_labels()
        if legend_reverse:
            handles = handles[::-1]
            labels = labels[::-1]
        ax.legend(handles, labels, loc=legend_loc, fontsize=LEGEND_FS, framealpha=0.9,
                  edgecolor="#4f4f4f", borderpad=0.3, labelspacing=0.25, handlelength=1.4,
                  handletextpad=0.4)


def make_comparison_plot(
    left_experiments: Sequence[Tuple[str, Sequence[CheckpointResult], str]],
    right_experiments: Sequence[Tuple[str, Sequence[CheckpointResult], str]],
    left_name: str,
    right_name: str,
    dpi: int,
    plot_name: Optional[str] = None,
    shared_legend: bool = False,
    legend_title: Optional[str] = None,
    legend_reverse_left: bool = False,
    legend_reverse_right: bool = False,
    legend_loc: str = "best",
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

    panel_legend = not shared_legend
    _draw_panel(ax_left, left_experiments, left_name,
                show_legend=panel_legend and len(left_experiments) > 1, legend_loc=legend_loc,
                legend_reverse=legend_reverse_left)
    _draw_panel(ax_right, right_experiments, right_name,
                show_legend=panel_legend and len(right_experiments) > 1, legend_loc=legend_loc,
                legend_reverse=legend_reverse_right)

    if shared_legend:
        # Collect handles+labels from both axes; deduplicate by label name,
        # keeping the first occurrence (left panel takes priority).
        seen: dict = {}
        for ax in (ax_left, ax_right):
            for handle, label in zip(*ax.get_legend_handles_labels()):
                if label not in seen:
                    seen[label] = handle
        if seen:
            handles = list(seen.values())
            labels = list(seen.keys())
            # The shared legend is a single merged list, so reverse it whenever
            # reversal is requested for either panel.
            if legend_reverse_left or legend_reverse_right:
                handles = handles[::-1]
                labels = labels[::-1]
            # Anchor to the top-right corner of ax_right (zero gap, top-aligned).
            leg = ax_right.legend(
                handles,
                labels,
                loc="upper left",
                bbox_to_anchor=(1.03, 1.0),
                bbox_transform=ax_right.transAxes,
                title=legend_title,
                title_fontsize=LEGEND_TITLE_FS,
                fontsize=LEGEND_FS,
                framealpha=0.9,
                edgecolor="#4f4f4f",
                borderpad=0.3,
                labelspacing=0.25,
                handlelength=1.4,
                handletextpad=0.4,
                borderaxespad=0.0,
            )
            if legend_title:
                leg.get_title().set_fontweight("bold")

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
        description="Compare two sets of experiments side-by-side with independent axes but shared axis titles."
    )
    parser.add_argument("--left-path", type=Path, nargs="+", required=True,
                        help="Path(s) to the left panel's experiment output directories (T_a_<N> sub-folders).")
    parser.add_argument("--right-path", type=Path, nargs="+", required=True,
                        help="Path(s) to the right panel's experiment output directories (T_a_<N> sub-folders).")
    parser.add_argument("--left-name", type=str, default=None,
                        help="Title for the left panel (defaults to directory name when a single path is given).")
    parser.add_argument("--right-name", type=str, default=None,
                        help="Title for the right panel (defaults to directory name when a single path is given).")
    parser.add_argument(
        "--experiment-name", type=str, nargs="+", default=None,
        help=(
            "Legend label for each experiment path, listed in order: left paths first, then right paths. "
            "Must match the total number of paths (len(--left-path) + len(--right-path)). "
            "Defaults to the directory name of each path."
        ),
    )
    preset_help = f"presets: {', '.join(PRESET_COLORS)}; or any matplotlib color"
    parser.add_argument("--left-color", type=str, default=NAVY,
                        help=f"Color for the left panel when a single path is given ({preset_help}; default {NAVY}). "
                             "Ignored when multiple left paths are supplied (a palette is generated instead).")
    parser.add_argument("--right-color", type=str, default=DARK_RED,
                        help=f"Color for the right panel when a single path is given ({preset_help}; default {DARK_RED}). "
                             "Ignored when multiple right paths are supplied (a palette is generated instead).")
    parser.add_argument("--plot-name", type=str, default=None, help="Overall figure title.")
    parser.add_argument(
        "--shared-legend", action="store_true",
        help=(
            "Place a single legend to the right of both panels instead of drawing "
            "independent per-panel legends. Duplicate labels across panels are merged."
        ),
    )
    parser.add_argument(
        "--legend-title", type=str, default=None,
        help="Title displayed above the shared legend (only used with --shared-legend).",
    )
    parser.add_argument(
        "--legend-reverse", action="store_true",
        help="Reverse the order of legend entries for BOTH panels (shortcut for "
             "--legend-reverse-left --legend-reverse-right).",
    )
    parser.add_argument(
        "--legend-reverse-left", action="store_true",
        help="Reverse the order of entries in the left panel's legend only.",
    )
    parser.add_argument(
        "--legend-reverse-right", action="store_true",
        help="Reverse the order of entries in the right panel's legend only.",
    )
    parser.add_argument(
        "--legend-loc", type=str, default="best",
        help=(
            "Matplotlib location for the per-panel legend, e.g. 'best', 'lower left', "
            "'lower right', 'upper right'. Use 'lower left'/'lower right' to snap the "
            "legend to the bottom corners. Ignored when --shared-legend is set."
        ),
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save the figure (PNG, PDF, etc.). Omit to skip saving.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving to disk.")
    return parser.parse_args()


def _load(exp_path: Path, label: str) -> Optional[Sequence[CheckpointResult]]:
    """Load results for one experiment, or return ``None`` (after printing an
    error) if the path is missing or contains no valid results."""
    if not exp_path.exists():
        print(f"ERROR: Experiment path '{exp_path}' does not exist. Skipping '{label}'.")
        return None
    results = collect_best_results(exp_path)
    if not results:
        print(f"ERROR: No valid results.pkl files found under {exp_path}. Skipping '{label}'.")
        return None
    print(f"\n{label} — best checkpoint per horizon:")
    for r in results:
        if r.num_checkpoints_available > 1:
            trials_str = ", ".join(str(t) for t in r.all_checkpoint_trials)
            n_tag = f" [{r.num_checkpoints_available} ckpts: {trials_str} trials]"
        else:
            n_tag = ""
        print(f"  T_a_{r.horizon}: {r.success_rate:.3f} ({r.num_trials} trials) -> {r.checkpoint_dir}{n_tag}")
    return results


def _build_experiments(
    paths: List[Path],
    labels: List[str],
    single_path_color: str,
    panel_title: str,
) -> List[Tuple[str, Sequence[CheckpointResult], str]]:
    """Load results for each path and assign colors."""
    print(f"\n{'='*60}")
    print(f"Panel: {panel_title}")
    print(f"{'='*60}")

    if len(paths) == 1:
        colors = [_resolve_color(single_path_color)]
    else:
        colors = _generate_color_palette(len(paths))

    experiments: List[Tuple[str, Sequence[CheckpointResult], str]] = []
    for path, label, color in zip(paths, labels, colors):
        results = _load(path, label)
        if results is None:
            continue
        experiments.append((label, results, color))
    return experiments


def main() -> None:
    args = parse_args()

    left_paths: List[Path] = args.left_path
    right_paths: List[Path] = args.right_path
    total_paths = len(left_paths) + len(right_paths)

    # Resolve experiment-level labels.
    if args.experiment_name is not None:
        if len(args.experiment_name) != total_paths:
            raise ValueError(
                f"--experiment-name count ({len(args.experiment_name)}) must match the total number of "
                f"paths ({total_paths} = {len(left_paths)} left + {len(right_paths)} right)."
            )
        all_labels = args.experiment_name
    else:
        all_labels = [p.name for p in left_paths + right_paths]

    left_labels = all_labels[: len(left_paths)]
    right_labels = all_labels[len(left_paths):]

    # Resolve panel titles.
    if args.left_name:
        left_panel_title = args.left_name
    elif len(left_paths) == 1:
        left_panel_title = left_labels[0]
    else:
        left_panel_title = "Left"

    if args.right_name:
        right_panel_title = args.right_name
    elif len(right_paths) == 1:
        right_panel_title = right_labels[0]
    else:
        right_panel_title = "Right"

    left_experiments = _build_experiments(left_paths, left_labels, args.left_color, left_panel_title)
    right_experiments = _build_experiments(right_paths, right_labels, args.right_color, right_panel_title)

    fig = make_comparison_plot(
        left_experiments,
        right_experiments,
        left_panel_title,
        right_panel_title,
        dpi=args.dpi,
        plot_name=args.plot_name,
        shared_legend=args.shared_legend,
        legend_title=args.legend_title,
        legend_reverse_left=args.legend_reverse or args.legend_reverse_left,
        legend_reverse_right=args.legend_reverse or args.legend_reverse_right,
        legend_loc=args.legend_loc,
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
