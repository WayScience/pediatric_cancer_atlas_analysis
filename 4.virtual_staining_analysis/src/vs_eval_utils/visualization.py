"""
Visualization suite for model eval results
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
from matplotlib import cm as colorm, colors as mcolors
from PyComplexHeatmap import *


def _make_discrete_palette(levels, palette=None, cmap_name="Set2", fallback_colors=None):
    """
    Resolve a discrete palette dict for categorical annotations.

    Parameters
    ----------
    levels : sequence
        Ordered category levels.
    palette : None | dict | list/tuple | str
        - dict: explicit {level: color}
        - list/tuple: colors in level order
        - str: matplotlib colormap name
        - None: use cmap_name / fallback_colors
    cmap_name : str
        Default matplotlib colormap name if palette is None.
    fallback_colors : list[str] | None
        Optional fixed fallback colors.

    Returns
    -------
    dict
        Mapping from level -> color
    """
    levels = list(levels)

    if palette is None:
        if fallback_colors is not None and len(fallback_colors) >= len(levels):
            return dict(zip(levels, fallback_colors[: len(levels)]))
        cmap = colorm.get_cmap(cmap_name)
        colors = [mcolors.to_hex(c) for c in cmap(np.linspace(0, 1, len(levels)))]
        return dict(zip(levels, colors))

    if isinstance(palette, dict):
        missing = [lvl for lvl in levels if lvl not in palette]
        if missing:
            raise ValueError(f"Palette dict missing levels: {missing}")
        return {lvl: palette[lvl] for lvl in levels}

    if isinstance(palette, str):
        cmap = colorm.get_cmap(palette)
        colors = [mcolors.to_hex(c) for c in cmap(np.linspace(0, 1, len(levels)))]
        return dict(zip(levels, colors))

    if isinstance(palette, (list, tuple)):
        if len(palette) < len(levels):
            raise ValueError(
                f"Palette list has {len(palette)} colors but needs at least {len(levels)}."
            )
        return dict(zip(levels, palette[: len(levels)]))

    raise TypeError("palette must be None, dict, list/tuple, or matplotlib cmap name string")


def _make_sequential_value_palette(
    levels,
    cmap="Blues",
    vmin=None,
    vmax=None,
    as_str_keys=False,
    labels_as_str=True,
):
    """
    Shared sequential palette for ordered numeric annotation values like densities.
    """
    levels = list(levels)
    if vmin is None:
        vmin = min(levels)
    if vmax is None:
        vmax = max(levels)

    norm = Normalize(vmin=vmin, vmax=vmax)
    cmap_obj = colorm.get_cmap(cmap)

    palette = {
        (str(v) if as_str_keys else v): mcolors.to_hex(cmap_obj(norm(v)))
        for v in levels
    }
    label_map = {v: (str(v) if labels_as_str else v) for v in levels}
    return palette, label_map


def _build_adaptive_rwb_norm(
    values,
    center="auto",
    clip_quantiles=(0.02, 0.98),
    symmetric_span=True,
    min_center_quantile_gap=0.05,
):
    """
    Build an adaptive TwoSlopeNorm for a red-white-blue heatmap.

    Strategy
    --------
    1. Robustly clip extreme tails using quantiles.
    2. Pick a center:
       - if center is numeric: use it
       - if center == 'auto':
           * use 0.5 if data looks bounded in [0, 1]
           * otherwise use the median
    3. Expand vmin/vmax around center:
       - symmetric_span=True gives balanced visual contrast on both sides
       - False uses data-driven asymmetric bounds

    This preserves a red-white-blue map but adjusts mapping so contrast is not
    wasted by a few outliers or heavy skew.
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        raise ValueError("No finite values available for adaptive normalization.")

    q_low, q_high = np.quantile(arr, clip_quantiles)

    if center == "auto":
        # heuristic: if the metric behaves like a bounded similarity score
        # use 0.5 as semantic midpoint; otherwise use median
        arr_min = np.nanmin(arr)
        arr_max = np.nanmax(arr)
        if arr_min >= -1e-8 and arr_max <= 1 + 1e-8:
            vcenter = 0.5
        else:
            vcenter = float(np.median(arr))
    else:
        vcenter = float(center)

    # Guard against degenerate center placement after clipping
    if q_low >= vcenter:
        q_low = min(np.nanmin(arr), vcenter - max(min_center_quantile_gap, 1e-8))
    if q_high <= vcenter:
        q_high = max(np.nanmax(arr), vcenter + max(min_center_quantile_gap, 1e-8))

    if symmetric_span:
        span = max(vcenter - q_low, q_high - vcenter)
        if span <= 0:
            span = max(np.nanstd(arr), 1e-6)
        vmin = vcenter - span
        vmax = vcenter + span
    else:
        vmin = q_low
        vmax = q_high

    # final guardrails
    if not (vmin < vcenter < vmax):
        eps = max(np.nanstd(arr) * 1e-3, 1e-6)
        vmin = min(vmin, vcenter - eps)
        vmax = max(vmax, vcenter + eps)

    norm = TwoSlopeNorm(vmin=vmin, vcenter=vcenter, vmax=vmax)
    return norm, {"vmin": vmin, "vcenter": vcenter, "vmax": vmax}


def plot_cm_raw_metrics(
    df: pd.DataFrame,
    metric_name: str,
    value_col: str = "metric_value",
    arch_order: tuple = ("UNet", "wGAN", "ConvNeXtUNet"),

    # ---- annotation palette knobs ----
    architecture_palette = None,
    channel_palette = None,
    density_palette: dict = None,
    cell_palette = None,
    density_cmap: str = "Blues",

    # ---- boolean row annotation ----
    annotate_row: str = None,
    row_annotation_text: str = "Annotated",

    # ---- annotation legend / text knobs ----
    show_cell_legend: bool = True,
    show_density_legend: bool = True,
    show_architecture_legend: bool = False,
    show_channel_legend: bool = False,
    show_train_density_legend: bool = False,

    row_text: bool = True,
    col_text: bool = True,

    # ---- main heatmap color knobs ----
    main_center = "auto",
    main_clip_quantiles: tuple = (0.02, 0.98),
    main_symmetric_span: bool = True,
    main_cmap = None,
    force_unit_interval_center: bool = False,

    # ---- figure/layout knobs ----
    figsize: tuple = (18, 18),
    linewidths: float = 0.2,
    linecolor: str = "white",

    show: bool = True,
) -> dict:
    """
    Helper for raw metric aggregated metric heatmap visualization

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing metrics and metadata.
    metric_name : str
        Name of the metric being plotted (used in the legend).
    value_col : str, optional
        Column name in `df` representing the metric values, by default "metric_value".
    arch_order : tuple, optional
        Ordering for architecture models, by default ("UNet", "wGAN", "ConvNeXtUNet").
    architecture_palette : dict, list, str, optional
        Palette for architecture annotations, by default None.
    channel_palette : dict, list, str, optional
        Palette for channel annotations, by default None.
    density_palette : dict, optional
        Palette for density annotations, by default None.
    cell_palette : dict, list, str, optional
        Palette for cell line annotations, by default None.
    density_cmap : str, optional
        Colormap name for sequential density generation if `density_palette` is None, by default "Blues".
    annotate_row : str, optional
        Boolean column to add as an extra row annotation, by default None.
    row_annotation_text : str, optional
        Label for the boolean row annotation, by default "Annotated".
    show_cell_legend : bool, optional
        Whether to show the cell annotation legend, by default True.
    show_density_legend : bool, optional
        Whether to show the density annotation legend, by default True.
    show_architecture_legend : bool, optional
        Whether to show the architecture legend, by default False.
    show_channel_legend : bool, optional
        Whether to show the channel legend, by default False.
    show_train_density_legend : bool, optional
        Whether to show the train density legend, by default False.
    row_text : bool, optional
        Whether to overlay text on row annotations, by default True.
    col_text : bool, optional
        Whether to overlay text on column annotations, by default True.
    main_center : str or float, optional
        Center value for adaptive normalization ("auto" to infer), by default "auto".
    main_clip_quantiles : tuple, optional
        Quantiles to clip extrema for determining colormap spread, by default (0.02, 0.98).
    main_symmetric_span : bool, optional
        If True, the mapping around `main_center` is symmetric, by default True.
    main_cmap : Colormap, optional
        Colormap for the main heatmap. If None, uses a default red-white-blue, by default None.
    force_unit_interval_center : bool, optional
        Forces `main_center=0.5` if the data bounds fit into [0, 1], by default False.
    figsize : tuple, optional
        Tuple specifying matplotlib figure dimensions, by default (18, 18).
    linewidths : float, optional
        Width of lines that divide heatmap cells, by default 0.2.
    linecolor : str, optional
        Color of lines that divide heatmap cells, by default "white".
    show : bool, optional
        Whether to display the generated plot out-of-the-box, by default True.

    Returns
    -------
    dict
        Dictionary containing references to the figure, plotter, matrix, metadata, and colormap info.
    """
    row_keys = ["platemap_file", "cell_line", "seeding_density"]
    col_keys = [
        "Metadata_Model_architecture",
        "Metadata_Model_target_channels",
        "Metadata_Model_train_density",
    ]

    agg = (
        df.groupby(row_keys + col_keys, dropna=False, observed=False)[value_col]
        .mean()
        .reset_index()
    )

    agg["row_id"] = agg.apply(
        lambda r: f"{r['platemap_file']} | {r['cell_line']} | seed={r['seeding_density']}",
        axis=1,
    )
    agg["col_id"] = agg.apply(
        lambda r: (
            f"{r['Metadata_Model_architecture']} | "
            f"{r['Metadata_Model_target_channels']} | "
            f"train={r['Metadata_Model_train_density']}"
        ),
        axis=1,
    )

    mat = agg.pivot(index="row_id", columns="col_id", values=value_col)

    row_meta = (
        agg[["row_id"] + row_keys]
        .drop_duplicates()
        .set_index("row_id")
        .loc[mat.index]
        .copy()
    )
    row_meta["seeding_density"] = row_meta["seeding_density"].astype(int)

    col_meta = (
        agg[["col_id"] + col_keys]
        .drop_duplicates()
        .set_index("col_id")
        .loc[mat.columns]
        .copy()
    )
    col_meta["Metadata_Model_architecture"] = pd.Categorical(
        col_meta["Metadata_Model_architecture"],
        categories=list(arch_order),
        ordered=True,
    )

    # --------- resolve palettes ---------
    architecture_levels = list(pd.unique(col_meta["Metadata_Model_architecture"]))
    architecture_palette = _make_discrete_palette(
        architecture_levels,
        palette=architecture_palette,
        cmap_name="Set2",
        fallback_colors=["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2"],
    ) if not architecture_palette else architecture_palette

    channel_levels = list(pd.unique(col_meta["Metadata_Model_target_channels"]))
    channel_palette = _make_discrete_palette(
        channel_levels,
        palette=channel_palette,
        cmap_name="Set2",
    ) if not channel_palette else channel_palette

    cell_levels = list(pd.unique(row_meta["cell_line"]))
    cell_palette = _make_discrete_palette(
        cell_levels,
        palette=cell_palette,
        cmap_name="Set2",
    ) if not cell_palette else cell_palette

    density_levels = sorted(
        pd.unique(
            pd.concat(
                [
                    row_meta["seeding_density"].astype(int),
                    col_meta["Metadata_Model_train_density"].astype(int),
                ],
                axis=0,
            )
        )
    )

    if density_palette is None:
        density_palette, density_label_map = _make_sequential_value_palette(
            density_levels,
            cmap=density_cmap,
            as_str_keys=False,
            labels_as_str=True,
        )
    elif isinstance(density_palette, dict):
        missing = [d for d in density_levels if d not in density_palette]
        if missing:
            raise ValueError(f"density_palette missing density levels: {missing}")
        density_label_map = {d: str(d) for d in density_levels}
    else:
        raise TypeError("density_palette must be None or a dict keyed by density value")

    # --------- ordering ---------
    row_order = row_meta.sort_values(row_keys).index
    col_order = col_meta.sort_values(
        by=[
            "Metadata_Model_architecture",
            "Metadata_Model_target_channels",
            "Metadata_Model_train_density",
            "col_id",
        ],
        kind="stable",
    ).index

    mat = mat.loc[row_order, col_order]
    row_meta = row_meta.loc[row_order]
    col_meta = col_meta.loc[col_order]

    row_meta["density_label"] = row_meta["seeding_density"].map(density_label_map)
    col_meta["density_label"] = col_meta["Metadata_Model_train_density"].map(density_label_map)

    # --------- row boolean annotation agg ---------
    if annotate_row is not None:
        row_anno_agg = df.groupby(row_keys, dropna=False, observed=False)[annotate_row].all()
        row_meta = row_meta.join(row_anno_agg, on=row_keys, how="left")
        row_meta[annotate_row + "_label"] = row_meta[annotate_row].fillna(False).map({True: "*", False: " "})

    # --------- main adaptive red-white-blue heatmap ---------
    if main_cmap is None:
        # low -> blue, mid -> white, high -> red
        main_cmap = LinearSegmentedColormap.from_list(
            "adaptive_rwb",
            [(0.0, "blue"), (0.5, "white"), (1.0, "red")]
        )

    heat_values = mat.to_numpy().ravel()
    center_for_norm = main_center

    if force_unit_interval_center:
        finite_vals = heat_values[np.isfinite(heat_values)]
        if finite_vals.size > 0 and finite_vals.min() >= -1e-8 and finite_vals.max() <= 1 + 1e-8:
            center_for_norm = 0.5

    norm, norm_info = _build_adaptive_rwb_norm(
        heat_values,
        center=center_for_norm,
        clip_quantiles=main_clip_quantiles,
        symmetric_span=main_symmetric_span,
    )

    # --------- annotations ---------
    row_ha_dict = {
        "platemap": anno_simple(
            row_meta["platemap_file"],
            add_text=row_text,
            legend=False,
            text_kws={"fontsize": 8, "fontweight": "bold"},
        ),
        "cell": anno_simple(
            row_meta["cell_line"],
            add_text=row_text,
            legend=show_cell_legend,
            colors=cell_palette,
        ),
        "density": anno_simple(
            row_meta["density_label"],
            colors=density_palette,
            legend=show_density_legend,
        )
    }

    if annotate_row is not None:
        # pycomplexheatmap doesn't have anno_text
        row_ha_dict[row_annotation_text] = anno_simple(
            row_meta[annotate_row + "_label"],
            add_text=True,
            colors={"*": "white", " ": "white"}, # white for both to hide uneeded color
            legend=True,
            text_kws={"fontsize": 10, "fontweight": "bold"},
        )

    row_ha = HeatmapAnnotation(
        **row_ha_dict,
        axis=0,
        label_kws={"fontsize": 8},
        verbose=0,
    )

    col_ha = HeatmapAnnotation(
        architecture=anno_simple(
            col_meta["Metadata_Model_architecture"],
            add_text=col_text,
            colors=architecture_palette,
            legend=show_architecture_legend,
            text_kws={"fontsize": 8, "fontweight": "bold", "rotation": 0},
        ),
        channel=anno_simple(
            col_meta["Metadata_Model_target_channels"],
            add_text=col_text,
            colors=channel_palette,
            legend=show_channel_legend,
        ),
        train_density=anno_simple(
            col_meta["density_label"],
            colors=density_palette,
            legend=show_train_density_legend,
        ),
        axis=1,
        label_kws={"fontsize": 8},
        verbose=0,
    )

    row_split = row_meta["platemap_file"]

    fig = plt.figure(figsize=figsize)

    cm = ClusterMapPlotter(
        data=mat,
        left_annotation=row_ha,
        top_annotation=col_ha,

        row_split=row_split,
        row_split_gap=0.6,

        row_cluster=False,
        col_cluster=False,

        show_rownames=False,
        show_colnames=False,
        row_names_side="left",
        col_names_side="top",

        cmap=main_cmap,
        norm=norm,
        label=f"mean({metric_name})",
        legend=True,
        verbose=0,

        xticklabels_kws={"labelrotation": 90, "labelsize": 8},
        yticklabels_kws={"labelsize": 8},

        linewidths=linewidths,
        linecolor=linecolor,
    )

    if show:
        plt.show()

    return {
        "fig": fig,
        "plotter": cm,
        "matrix": mat,
        "row_meta": row_meta,
        "col_meta": col_meta,
        "main_norm_info": norm_info,
        "architecture_palette": architecture_palette,
        "channel_palette": channel_palette,
        "density_palette": density_palette,
    }
