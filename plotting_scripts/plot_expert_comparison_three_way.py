"""
Combine three success-rate figures side-by-side to compare three types of experts.

Each expert is plotted in its own panel with independent x- and y-axes (the
ranges may differ between panels), but the three panels share common x-axis and
y-axis titles. By default the left panel is navy blue, the middle turquoise, and
the right dark red. Wilson confidence-interval error bars are included;
per-checkpoint (epoch) labels are intentionally omitted.

Each panel supports one *or more* experiment paths, which are overlaid as
separate lines within that panel. When multiple paths are supplied for a panel
the script auto-generates a lightness-spread palette anchored at the panel's
base color, and a per-panel legend is shown.

Reuses the data-loading logic from src/eval/plot_eval_results.py, which scans an
output directory with the structure:
    <experiment_path>/T_a_<horizon>/<checkpoint_stem>/results.pkl

Example usage
-------------

Human vs Non-Markovian vs Markovian Expert (single path per panel):

   python src/eval/plot_expert_comparison_three_way.py \
       --left-path outputs/2_obs_one_leg_teleop_r3m \
       --middle-path outputs/2_obs_one_leg_scripted_r3m_200_eps \
       --right-path outputs/2_obs_one_leg_scripted_r3m_non_markovian_200_eps \
       --left-name "Human (NM) Expert" \
       --middle-name "Scripted (M) Expert" \
       --right-name "Scripted (NM) Expert" \
       --left-color navy \
       --middle-color dark_red \
       --right-color turquoise \
       --output outputs/plots/comparison_human_expert_markovian_expert_non_markovian_expert_one_leg_v2.png

Context Length Ablation on 3 tasks:

   python src/eval/plot_expert_comparison_three_way.py \
       --left-path /data/locomotion/michzeng/push_t_results/2_obs_32_horizon_idle_frames_pruned_v2_clean /data/locomotion/michzeng/push_t_results/6_obs_32_horizon_idle_frames_pruned_v2_clean /data/locomotion/michzeng/push_t_results/10_obs_32_horizon_idle_frames_pruned_v2_clean /data/locomotion/michzeng/push_t_results/14_obs_32_horizon_idle_frames_pruned_v2_clean /data/locomotion/michzeng/push_t_results/18_obs_32_horizon_idle_frames_pruned_v2_clean \
       --left-name '$T_o=2$' '$T_o=6$' '$T_o=10$' '$T_o=14$' '$T_o=18$' \
       --left-panel-title "Push-T" \
       --middle-path /data/locomotion/michzeng/IsaacLab/outputs/v1_resets/2_obs_gear_assembly_human_expert_attention_double_enc /data/locomotion/michzeng/IsaacLab/outputs/v1_resets/4_obs_gear_assembly_human_expert_attention_double_enc /data/locomotion/michzeng/IsaacLab/outputs/v1_resets/6_obs_gear_assembly_human_expert_attention_double_enc /data/locomotion/michzeng/IsaacLab/outputs/v1_resets/8_obs_gear_assembly_human_expert_attention_double_enc \
       --middle-name '$T_o=2$' '$T_o=4$' '$T_o=6$' '$T_o=8$' \
       --middle-panel-title "Gear Assembly" \
       --right-path /data/locomotion/michzeng/relay-policy-learning/outputs/2_obs_human_expert_attention_double_enc /data/locomotion/michzeng/relay-policy-learning/outputs/4_obs_human_expert_attention_double_enc /data/locomotion/michzeng/relay-policy-learning/outputs/6_obs_human_expert_attention_double_enc /data/locomotion/michzeng/relay-policy-learning/outputs/8_obs_human_expert_attention_double_enc \
       --right-name '$T_o=2$' '$T_o=4$' '$T_o=6$' '$T_o=8$' \
       --right-panel-title "Kitchen" \
       --right-horizons 1 10 \
       --left-color navy \
       --middle-color dark_red \
       --right-color turquoise \
       --output outputs/plots/figure_3_tasks_context_length_ablation_v2.png

    python src/eval/plot_expert_comparison_three_way.py \
       --left-path /data/locomotion/michzeng/push_t_results/2_obs_32_horizon_idle_frames_pruned_v2_clean /data/locomotion/michzeng/push_t_results/6_obs_32_horizon_idle_frames_pruned_v2_clean /data/locomotion/michzeng/push_t_results/10_obs_32_horizon_idle_frames_pruned_v2_clean /data/locomotion/michzeng/push_t_results/14_obs_32_horizon_idle_frames_pruned_v2_clean /data/locomotion/michzeng/push_t_results/18_obs_32_horizon_idle_frames_pruned_v2_clean \
       --left-name '$T_o=2$' '$T_o=6$' '$T_o=10$' '$T_o=14$' '$T_o=18$' \
       --left-panel-title "Push-T" \
       --middle-path /data/locomotion/michzeng/IsaacLab/outputs/v1_resets/2_obs_gear_assembly_human_expert_attention_double_enc /data/locomotion/michzeng/IsaacLab/outputs/v1_resets/4_obs_gear_assembly_human_expert_attention_double_enc /data/locomotion/michzeng/IsaacLab/outputs/v1_resets/6_obs_gear_assembly_human_expert_attention_double_enc /data/locomotion/michzeng/IsaacLab/outputs/v1_resets/8_obs_gear_assembly_human_expert_attention_double_enc \
       --middle-name '$T_o=2$' '$T_o=4$' '$T_o=6$' '$T_o=8$' \
       --middle-panel-title "Gear Assembly" \
       --right-path /data/locomotion/michzeng/relay-policy-learning/outputs/2_obs_human_expert /data/locomotion/michzeng/relay-policy-learning/outputs/3_obs_human_expert /data/locomotion/michzeng/relay-policy-learning/outputs/4_obs_human_expert /data/locomotion/michzeng/relay-policy-learning/outputs/5_obs_human_expert /data/locomotion/michzeng/relay-policy-learning/outputs/6_obs_human_expert \
       --right-name '$T_o=2$' '$T_o=3$' '$T_o=4$' '$T_o=5$' '$T_o=6$' \
       --right-panel-title "Kitchen" \
       --right-horizons 1 10 \
       --left-color navy \
       --middle-color dark_red \
       --right-color turquoise \
       --right-ymin 0.73 \
       --output outputs/plots/figure_3_tasks_context_length_ablation_v3.png

Don't set --output to skip saving. Set --show to open an interactive window.
"""

from __future__ import annotations

import argparse
import colorsys
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.stats.proportion as smp
from matplotlib.ticker import FormatStrFormatter, MultipleLocator, ScalarFormatter
from matplotlib.transforms import ScaledTranslation

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

# Preset color names selectable via --left-color / --middle-color / --right-color.
# Any other string is passed straight through as a matplotlib color (hex, named, etc.).
PRESET_COLORS = {
    "navy": NAVY,
    "dark_red": DARK_RED,
    "turquoise": TURQUOISE,
}


def _resolve_color(color: str) -> str:
    """Map a preset name to its hex value, or pass the string through unchanged."""
    return PRESET_COLORS.get(color.lower(), color)


def _generate_panel_palette(base_color: str, n: int) -> List[str]:
    """Generate *n* colors by varying the lightness around *base_color*.

    For n=1 the base color is returned unchanged.  For n>1 the palette spreads
    from a darker shade to a lighter shade, keeping hue and saturation fixed.
    """
    if n <= 0:
        return []
    if n == 1:
        return [base_color]

    hex_str = base_color.lstrip("#")
    r, g, b = (int(hex_str[i:i + 2], 16) / 255.0 for i in (0, 2, 4))
    h, l, s = colorsys.rgb_to_hls(r, g, b)

    # Spread lightness symmetrically around the base, clamped to [0.15, 0.82].
    half = 0.22
    l_min = max(0.15, l - half)
    l_max = min(0.82, l + half)

    colors: List[str] = []
    for i in range(n):
        lv = l_min + (l_max - l_min) * i / (n - 1)
        r2, g2, b2 = colorsys.hls_to_rgb(h, lv, s)
        colors.append(f"#{round(r2 * 255):02x}{round(g2 * 255):02x}{round(b2 * 255):02x}")
    return colors


# -----------------------------------------------------------------------------
# CoRL pre-print layout
# -----------------------------------------------------------------------------
# CoRL uses a single-column US-letter layout with a text block ~6.0 in wide and
# 10 pt body text (captions ~9 pt). Sizing the figure to the full text width and
# keeping in-figure text at ~8-10 pt keeps everything legible without scaling.

FIG_WIDTH_IN = 6.0   # full CoRL text width
FIG_HEIGHT_IN = 2.2  # three side-by-side panels at a comfortable aspect ratio

AXIS_TITLE_FS = 8    # shared x/y axis titles
PANEL_TITLE_FS = 9   # per-panel titles (kept small so adjacent titles don't overlap)
SUPTITLE_FS = 10     # overall figure title
TICK_FS = 7          # tick labels (kept at the ~7 pt readability floor)
LEGEND_FS = 6        # per-panel legend labels


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------

# A panel's data is a list of (label, results, color) triples.
PanelData = List[Tuple[str, Sequence[CheckpointResult], str]]


def _draw_panel(
    ax: plt.Axes,
    experiments: PanelData,
    title: str,
    legend_loc: Optional[str] = None,
    ymin: Optional[float] = None,
) -> None:
    """Draw one or more experiment traces onto *ax*."""
    ax.set_facecolor("white")

    all_horizons: List[int] = []
    for _label, results, color in experiments:
        if not results:
            continue

        horizons = np.array([r.horizon for r in results], dtype=float)
        rates = np.array([r.success_rate for r in results], dtype=float)
        trials = np.array([r.num_trials for r in results], dtype=int)

        ci = np.array(
            [smp.proportion_confint(int(p * n), n, alpha=0.05, method="wilson")
             for p, n in zip(rates, trials)]
        )
        yerr = np.vstack([np.clip(rates - ci[:, 0], 0, 1), np.clip(ci[:, 1] - rates, 0, 1)])

        ax.plot(horizons, rates, color=color, linewidth=1.5, marker="o", markersize=4,
                markeredgecolor="white", markeredgewidth=0.8, label=_label, zorder=3)
        ax.errorbar(horizons, rates, yerr=yerr, fmt="none", ecolor=color, elinewidth=0.7,
                    capsize=2.0, capthick=0.85, alpha=0.9, zorder=2)
        all_horizons.extend(int(h) for h in horizons)

    if all_horizons:
        panel_horizons = sorted(set(all_horizons))
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(ScalarFormatter())
        ax.set_xticks(panel_horizons)
        ax.set_xticklabels([str(h) for h in panel_horizons])

    if ymin is not None:
        ax.set_ylim(bottom=ymin)

    ax.set_title(title, fontsize=PANEL_TITLE_FS, fontweight="bold", pad=3)

    ax.grid(True, which="major", color=GRID_COLOR, linestyle="-", linewidth=0.8, alpha=0.6)
    ax.grid(True, which="minor", color=GRID_COLOR, linestyle="-", linewidth=0.5, alpha=0.3)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#4f4f4f")

    ax.tick_params(axis="both", which="major", direction="in", labelsize=TICK_FS, length=2.5, width=0.8)
    ax.tick_params(axis="x", which="minor", direction="in", length=1.5, width=0.6)
    ax.tick_params(axis="y", which="minor", left=False)
    ax.yaxis.set_major_locator(MultipleLocator(0.1))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.tick_params(axis="y", which="major", pad=2, labelrotation=90)

    if len(experiments) > 1:
        ax.legend(loc=legend_loc or "best", fontsize=LEGEND_FS, framealpha=0.9,
                  edgecolor="#4f4f4f", borderpad=0.3, labelspacing=0.2,
                  handlelength=1.2, handletextpad=0.3)


# Candidate y-tick intervals, smallest first. Restricted to values that render
# cleanly with the single-decimal ("%.1f") formatter.
_Y_TICK_STEPS = (0.1, 0.2, 0.5, 1.0)

Y_TICKLABEL_RAISE_PT = 4  # upward nudge so rotated y labels sit on-center


def _apply_adaptive_yticks(ax: plt.Axes, renderer, gap_factor: float = 2.2) -> None:
    """Widen the y-tick interval when 0.1 spacing would crowd the rotated labels.

    Picks the smallest step in ``_Y_TICK_STEPS`` whose on-screen spacing clears
    the label's own extent (times ``gap_factor``), so wide-range panels thin out
    while narrow-range panels keep fine 0.1 ticks. Must be called after an
    initial draw so pixel extents are available.
    """
    labels = [t for t in ax.get_yticklabels() if t.get_text()]
    if not labels:
        return
    y0, y1 = ax.get_ylim()
    span = abs(y1 - y0)
    if span <= 0:
        return

    axis_height_px = ax.get_window_extent(renderer).height
    label_extent_px = max(lbl.get_window_extent(renderer).height for lbl in labels)
    needed_px = label_extent_px * gap_factor

    chosen = _Y_TICK_STEPS[-1]
    for step in _Y_TICK_STEPS:
        spacing_px = (step / span) * axis_height_px
        if spacing_px >= needed_px:
            chosen = step
            break
    ax.yaxis.set_major_locator(MultipleLocator(chosen))


def _raise_yticklabels(ax: plt.Axes, dpi_scale_trans, dy_pt: float = Y_TICKLABEL_RAISE_PT) -> None:
    """Nudge rotated y-tick labels up by ``dy_pt`` points so they read on-center."""
    offset = ScaledTranslation(0.0, dy_pt / 72.0, dpi_scale_trans)
    for lbl in ax.get_yticklabels():
        lbl.set_transform(lbl.get_transform() + offset)


def make_comparison_plot(
    left: PanelData,
    middle: PanelData,
    right: PanelData,
    left_title: str,
    middle_title: str,
    right_title: str,
    dpi: int,
    plot_name: Optional[str] = None,
    legend_loc: Optional[str] = None,
    left_ymin: Optional[float] = None,
    middle_ymin: Optional[float] = None,
    right_ymin: Optional[float] = None,
) -> plt.Figure:
    # sharex=False / sharey=False keeps the three panels' axes fully independent.
    # constrained_layout snugly packs the shared super-labels against the axes,
    # avoiding the large gap tight_layout leaves under a supxlabel.
    fig, (ax_left, ax_middle, ax_right) = plt.subplots(
        1, 3, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharex=False, sharey=False,
        layout="constrained",
    )
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.03, hspace=0.03)

    _draw_panel(ax_left, left, left_title, legend_loc=legend_loc, ymin=left_ymin)
    _draw_panel(ax_middle, middle, middle_title, legend_loc=legend_loc, ymin=middle_ymin)
    _draw_panel(ax_right, right, right_title, legend_loc=legend_loc, ymin=right_ymin)

    fig.supxlabel("Execution Horizon (steps)", fontsize=AXIS_TITLE_FS)
    fig.supylabel("Success Rate", fontsize=AXIS_TITLE_FS)

    if plot_name:
        fig.suptitle(plot_name, fontsize=SUPTITLE_FS, fontweight="bold")

    fig.set_dpi(dpi)

    # Initial draw so pixel extents are available, then widen any crowded y-tick
    # intervals, then redraw with the new ticks.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for ax in (ax_left, ax_middle, ax_right):
        _apply_adaptive_yticks(ax, renderer)
    fig.canvas.draw()

    # Freeze layout so the manual label nudge below is not overwritten by savefig.
    fig.set_layout_engine("none")
    for ax in (ax_left, ax_middle, ax_right):
        _raise_yticklabels(ax, fig.dpi_scale_trans)

    return fig


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare three panels of experts side-by-side. "
            "Each panel accepts one or more experiment paths (overlaid as lines)."
        )
    )
    parser.add_argument("--left-path", type=Path, nargs="+", required=True,
                        help="One or more experiment paths for the left panel.")
    parser.add_argument("--middle-path", type=Path, nargs="+", required=True,
                        help="One or more experiment paths for the middle panel.")
    parser.add_argument("--right-path", type=Path, nargs="+", required=True,
                        help="One or more experiment paths for the right panel.")
    parser.add_argument("--left-name", type=str, nargs="*", default=None,
                        help="Legend labels for the left panel (one per path; defaults to dir names). "
                             "When a single string is given for multiple paths it becomes the panel title only.")
    parser.add_argument("--middle-name", type=str, nargs="*", default=None,
                        help="Legend labels for the middle panel (one per path; defaults to dir names).")
    parser.add_argument("--right-name", type=str, nargs="*", default=None,
                        help="Legend labels for the right panel (one per path; defaults to dir names).")
    preset_help = f"presets: {', '.join(PRESET_COLORS)}; or any matplotlib color"
    parser.add_argument("--left-color", type=str, default=NAVY,
                        help=f"Base color for the left panel ({preset_help}; default {NAVY}).")
    parser.add_argument("--middle-color", type=str, default=TURQUOISE,
                        help=f"Base color for the middle panel ({preset_help}; default {TURQUOISE}).")
    parser.add_argument("--right-color", type=str, default=DARK_RED,
                        help=f"Base color for the right panel ({preset_help}; default {DARK_RED}).")
    parser.add_argument("--left-panel-title", type=str, default=None,
                        help="Title shown above the left panel. Defaults to the first (or only) --left-name.")
    parser.add_argument("--middle-panel-title", type=str, default=None,
                        help="Title shown above the middle panel. Defaults to the first (or only) --middle-name.")
    parser.add_argument("--right-panel-title", type=str, default=None,
                        help="Title shown above the right panel. Defaults to the first (or only) --right-name.")
    parser.add_argument("--left-horizons", type=int, nargs=2, default=None,
                        metavar=("LOWER", "UPPER"),
                        help="Inclusive [LOWER UPPER] execution-horizon range to display for the left panel.")
    parser.add_argument("--middle-horizons", type=int, nargs=2, default=None,
                        metavar=("LOWER", "UPPER"),
                        help="Inclusive [LOWER UPPER] execution-horizon range to display for the middle panel.")
    parser.add_argument("--right-horizons", type=int, nargs=2, default=None,
                        metavar=("LOWER", "UPPER"),
                        help="Inclusive [LOWER UPPER] execution-horizon range to display for the right panel.")
    parser.add_argument("--left-ymin", type=float, default=None,
                        help="Minimum y-axis (success rate) value for the left panel.")
    parser.add_argument("--middle-ymin", type=float, default=None,
                        help="Minimum y-axis (success rate) value for the middle panel.")
    parser.add_argument("--right-ymin", type=float, default=None,
                        help="Minimum y-axis (success rate) value for the right panel.")
    parser.add_argument("--plot-name", type=str, default=None, help="Overall figure title.")
    parser.add_argument(
        "--legend-loc", type=str, default=None,
        choices=["upper-left", "upper-right", "lower-left", "lower-right", "best"],
        help="Legend placement within each panel (default: matplotlib auto).",
    )
    parser.add_argument("--output", type=Path, default=None,
                        help="Path to save the figure (PNG, PDF, etc.). Omit to skip saving.")
    parser.add_argument("--show", action="store_true", help="Open an interactive window after saving.")
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI when saving to disk.")
    return parser.parse_args()


def _resolve_legend_loc(raw: Optional[str]) -> Optional[str]:
    mapping = {
        "upper-left": "upper left",
        "upper-right": "upper right",
        "lower-left": "lower left",
        "lower-right": "lower right",
        "best": "best",
    }
    return mapping.get(raw) if raw else None


def _filter_horizons(
    results: Sequence[CheckpointResult],
    horizons: Optional[Sequence[int]],
) -> Sequence[CheckpointResult]:
    """Keep only results whose horizon falls in the inclusive [lower, upper] range."""
    if not horizons:
        return results
    lower, upper = min(horizons), max(horizons)
    return [r for r in results if lower <= r.horizon <= upper]


def _build_panel(
    paths: List[Path],
    names: Optional[List[str]],
    base_color: str,
    panel_label: str,
    horizons: Optional[Sequence[int]] = None,
) -> Tuple[PanelData, str]:
    """Load results for all paths in a panel and return (experiments, panel_title).

    *panel_label* is the fallback title used when names has only one entry or is
    absent, and when paths has exactly one element.

    *horizons*, if given, is a (lower, upper) pair that filters results to the
    inclusive horizon range before plotting — identical to the ``--horizons``
    argument in ``generate_markovian_experts_figure.py``.
    """
    # Resolve labels: default to directory names when not provided.
    if names is None:
        labels = [p.name for p in paths]
    elif len(names) == len(paths):
        labels = names
    elif len(names) == 1 and len(paths) > 1:
        # Single name supplied for multiple paths → treat it as the panel title
        # and fall back to directory names as per-line labels.
        panel_label = names[0]
        labels = [p.name for p in paths]
    else:
        raise ValueError(
            f"--*-name count ({len(names)}) must match --*-path count ({len(paths)}) "
            f"or be exactly 1 (used as the panel title only)."
        )

    palette = _generate_panel_palette(_resolve_color(base_color), len(paths))
    experiments: PanelData = []
    for path, label, color in zip(paths, labels, palette):
        if not path.exists():
            print(f"WARNING: path does not exist, skipping: {path}")
            continue
        results = collect_best_results(path)
        if not results:
            print(f"WARNING: no valid results.pkl files found under {path}, skipping.")
            continue
        results = _filter_horizons(results, horizons)
        if not results:
            print(f"WARNING: no results remain after horizon filter for {path}, skipping.")
            continue
        print(f"\n{label} — best checkpoint per horizon:")
        for r in results:
            print(f"  T_a_{r.horizon}: {r.success_rate:.3f} ({r.num_trials} trials)")
        experiments.append((label, results, color))

    if not experiments:
        raise RuntimeError(f"No valid experiments loaded for panel '{panel_label}'.")

    title = panel_label
    return experiments, title


def main() -> None:
    args = parse_args()

    legend_loc = _resolve_legend_loc(args.legend_loc)

    left_experiments, left_default_title = _build_panel(
        args.left_path, args.left_name, args.left_color,
        (args.left_name[0] if args.left_name and len(args.left_name) == 1 else args.left_path[0].name),
        horizons=args.left_horizons,
    )
    middle_experiments, middle_default_title = _build_panel(
        args.middle_path, args.middle_name, args.middle_color,
        (args.middle_name[0] if args.middle_name and len(args.middle_name) == 1 else args.middle_path[0].name),
        horizons=args.middle_horizons,
    )
    right_experiments, right_default_title = _build_panel(
        args.right_path, args.right_name, args.right_color,
        (args.right_name[0] if args.right_name and len(args.right_name) == 1 else args.right_path[0].name),
        horizons=args.right_horizons,
    )

    left_title = args.left_panel_title or left_default_title
    middle_title = args.middle_panel_title or middle_default_title
    right_title = args.right_panel_title or right_default_title

    fig = make_comparison_plot(
        left_experiments, middle_experiments, right_experiments,
        left_title, middle_title, right_title,
        dpi=args.dpi, plot_name=args.plot_name,
        legend_loc=legend_loc,
        left_ymin=args.left_ymin,
        middle_ymin=args.middle_ymin,
        right_ymin=args.right_ymin,
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
