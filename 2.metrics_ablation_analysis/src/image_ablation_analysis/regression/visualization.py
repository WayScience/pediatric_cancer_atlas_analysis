"""
visualization.py

Generic visualization module for the nested regression analysis.

Functions: 
- Plots partial R² (for x2 term) against restricted R² (x1-only model),
    faceted by specified panel columns and colored by a hue column.
"""

from typing import Sequence, Optional, Tuple
from pathlib import Path
import textwrap

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_partial_r2_vs_r2(
    boot_res: pd.DataFrame,
    panel_cols: Sequence[str],
    hue_col: str,
    partial_col: str = "partial_r2_x2",
    r2_col: str = "r2_restricted",
    partial_label: Optional[str] = None,
    r2_label: Optional[str] = None,
    n_cols: int = 3,
    title_wrap_width: int = 48,
    title_fontsize: int = 11,
    title_pad: float = 10,
    legend_bbox_to_anchor: Tuple[float, float] = (0.82, 0.5),
    legend_fontsize: int = 10,
    legend_frameon: bool = True,
    subplot_left: float = 0.08,
    subplot_right: float = 0.78,
    subplot_top: float = 0.92,
    subplot_bottom: float = 0.10,
    subplot_wspace: float = 0.30,
    subplot_hspace: float = 0.45,
    save_dpi: int = 300,
    save_bbox_tight: bool = True,
    save_pad_inches: float = 0.20,
    save_path: Optional[Path] = None,
    show: bool = True,
):
    """
    Generic visualization of partial R² (for x2 term) vs restricted R² (x1-only model),
    faceted by panel_cols and colored by hue_col.

    :param boot_res: DataFrame with bootstrap regression results.
    :param panel_cols: Columns to facet the plots by (one subplot per unique combination).
    :param hue_col: Column to color the points by (different colors per unique value).
    :param partial_col: Column name for partial R² values (default: "partial_r2_x2").
    :param r2_col: Column name for restricted R² values (default: "r2_restricted").
    :param partial_label: Label for the partial R² axis. If None, a default label is used.
    :param r2_label: Label for the R² axis. If None, a default label is used.
    :param n_cols: Number of columns in the subplot grid (default: 3).
    :param title_wrap_width: Character width used to wrap long subplot titles (default: 48).
    :param title_fontsize: Font size for subplot titles (default: 11).
    :param title_pad: Padding for subplot titles in points (default: 10).
    :param legend_bbox_to_anchor: Figure-relative legend anchor as (x, y) (default: (0.82, 0.5)).
    :param legend_fontsize: Font size for legend text (default: 10).
    :param legend_frameon: Whether to draw legend frame (default: True).
    :param subplot_left: Left margin for subplot area (default: 0.08).
    :param subplot_right: Right margin for subplot area (default: 0.78).
    :param subplot_top: Top margin for subplot area (default: 0.92).
    :param subplot_bottom: Bottom margin for subplot area (default: 0.10).
    :param subplot_wspace: Horizontal spacing between subplots (default: 0.30).
    :param subplot_hspace: Vertical spacing between subplots (default: 0.45).
    :param save_dpi: DPI used for saving figure (default: 300).
    :param save_bbox_tight: Save with tight bounding box to avoid clipping (default: True).
    :param save_pad_inches: Extra padding in inches when saving (default: 0.20).
    :param save_path: Optional path to save the figure. If None, the figure is not saved.
    :param show: Whether to display the plot. If False, the plot is closed after creation.
    """

    # --------------------------
    # 1. Aggregate mean and 95% CI
    # --------------------------
    group_cols = list(panel_cols) + [hue_col]

    grouped_stats = (
        boot_res
        .groupby(group_cols)
        .agg({
            partial_col: [
                "mean",
                lambda x: x.quantile(0.025),
                lambda x: x.quantile(0.975),
            ],
            r2_col: [
                "mean",
                lambda x: x.quantile(0.025),
                lambda x: x.quantile(0.975),
            ],
        })
        .reset_index()
    )

    # Flatten multiindex columns
    grouped_stats.columns = (
        group_cols
        + [
            "partial_r2_mean",
            "partial_r2_lower",
            "partial_r2_upper",
            "r2_restricted_mean",
            "r2_restricted_lower",
            "r2_restricted_upper",
        ]
    )

    # --------------------------
    # 2. Error bar extents
    # --------------------------
    grouped_stats["partial_r2_err_lower"] = (
        grouped_stats["partial_r2_mean"] - grouped_stats["partial_r2_lower"]
    )
    grouped_stats["partial_r2_err_upper"] = (
        grouped_stats["partial_r2_upper"] - grouped_stats["partial_r2_mean"]
    )

    grouped_stats["r2_restricted_err_lower"] = (
        grouped_stats["r2_restricted_mean"] - grouped_stats["r2_restricted_lower"]
    )
    grouped_stats["r2_restricted_err_upper"] = (
        grouped_stats["r2_restricted_upper"] - grouped_stats["r2_restricted_mean"]
    )

    # --------------------------
    # 3. Unique panel combinations & colors
    # --------------------------
    unique_combinations = grouped_stats[panel_cols].drop_duplicates()

    unique_hues = grouped_stats[hue_col].unique()
    colors = sns.color_palette("tab10", n_colors=len(unique_hues))
    hue_colors = dict(zip(unique_hues, colors))

    threshold_specs = [
        {
            "ratio": 1.0,
            "color": "red",
            "label": "100% of restricted variance",
        },
        {
            "ratio": 0.5,
            "color": "orange",
            "label": "50% of restricted variance",
        },
        {
            "ratio": 0.1,
            "color": "green",
            "label": "10% of restricted variance",
        },
    ]

    # --------------------------
    # 4. Subplot grid
    # --------------------------
    n_plots = len(unique_combinations)
    n_rows = (n_plots + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6.5 * n_cols, 5.2 * n_rows))
    if isinstance(axes, plt.Axes):
        axes = [axes]
    else:
        axes = axes.flatten()

    # Axis labels
    if partial_label is None:
        partial_label = f"Partial R² ({partial_col})"
    if r2_label is None:
        r2_label = f"R² restricted ({r2_col})"

    # --------------------------
    # 5. Draw each panel
    # --------------------------
    # Track legend handles and labels for a single shared legend
    legend_handles = {}

    for idx, (_, combo_row) in enumerate(unique_combinations.iterrows()):
        ax = axes[idx]

        # mask for this panel (all panel_cols must match)
        mask = pd.Series(True, index=grouped_stats.index)
        for col in panel_cols:
            mask &= grouped_stats[col] == combo_row[col]

        combo_data = grouped_stats[mask]

        # Plot each hue category with error bars
        for hue_val in combo_data[hue_col].unique():
            metric_data = combo_data[combo_data[hue_col] == hue_val]

            errorbar = ax.errorbar(
                metric_data["r2_restricted_mean"],
                metric_data["partial_r2_mean"],
                xerr=[
                    metric_data["r2_restricted_err_lower"],
                    metric_data["r2_restricted_err_upper"],
                ],
                yerr=[
                    metric_data["partial_r2_err_lower"],
                    metric_data["partial_r2_err_upper"],
                ],
                fmt="o",
                label=hue_val,
                color=hue_colors[hue_val],
                alpha=0.7,
                markersize=6,
                capsize=3,
            )

            # Collect legend handles (only need one per hue value)
            if hue_val not in legend_handles:
                legend_handles[hue_val] = errorbar

        # --------------------------
        # Variance-equivalence threshold curve
        # --------------------------
        # General form:
        #   R²_full - R²_restricted = c * R²_restricted
        # which implies:
        #   partial R² = (c * x) / (1 - x)
        # where x = R²_restricted

        for spec in threshold_specs:
            x_max = 1.0 / (1.0 + spec["ratio"]) # solve for well defined domain for curve
            x_curve = np.linspace(0.001, x_max - 1e-3, 200)
            y_curve = (spec["ratio"] * x_curve) / (1 - x_curve)
            ax.plot(
                x_curve,
                y_curve,
                linestyle=":",
                color=spec["color"],
                linewidth=2,
                alpha=0.9,
                label=spec["label"] if idx == 0 else None,  # only label on first subplot
            )

        # Title: join panel values for this combo
        title_parts = [f"{col}={combo_row[col]}" for col in panel_cols]
        title_text = " | ".join(title_parts)
        wrapped_title = "\n".join(textwrap.wrap(title_text, width=title_wrap_width))
        ax.set_title(wrapped_title, fontsize=title_fontsize, pad=title_pad)

        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis("off")

    fig.supxlabel(r2_label, fontsize=12)
    fig.supylabel(partial_label, fontsize=12)

    # Create a single shared legend outside the plots (to the right)
    handles = [legend_handles[hue_val] for hue_val in unique_hues if hue_val in legend_handles]
    handles += [plt.Line2D([0], [0], color=spec["color"], linestyle=":", linewidth=2) for spec in threshold_specs]
    labels = [hue_val for hue_val in unique_hues if hue_val in legend_handles]
    labels += [spec["label"] for spec in threshold_specs]
    legend = fig.legend(
        handles, labels,
        loc="center left",
        bbox_to_anchor=legend_bbox_to_anchor,
        bbox_transform=fig.transFigure,
        fontsize=legend_fontsize,
        frameon=legend_frameon,
    )

    fig.subplots_adjust(
        left=subplot_left,
        right=subplot_right,
        top=subplot_top,
        bottom=subplot_bottom,
        wspace=subplot_wspace,
        hspace=subplot_hspace,
    )

    if save_path is not None:
        save_path = Path(save_path).resolve()
        save_root = save_path.parent
        save_root.mkdir(parents=True, exist_ok=True)
        save_kwargs = {
            "dpi": save_dpi,
        }
        if save_bbox_tight:
            save_kwargs.update(
                {
                    "bbox_inches": "tight",
                    "pad_inches": save_pad_inches,
                    "bbox_extra_artists": [legend],
                }
            )
        plt.savefig(save_path, **save_kwargs)

    if show:
        plt.show()
    else:
        plt.close(fig)
