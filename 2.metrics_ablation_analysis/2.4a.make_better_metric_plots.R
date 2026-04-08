# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .R
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: R
#     language: R
#     name: ir
# ---

# %% [markdown]
# # Generates metric ablation analysis figure and save as single panels
# Mirrors the visualization in 2.3, except drawn in R

# %% vscode={"languageId": "r"}
suppressPackageStartupMessages({
    suppressWarnings(library(readr))
    suppressWarnings(library(dplyr))
    suppressWarnings(library(ggplot2))
    suppressWarnings(library(tidyr))
    suppressWarnings(library(stringr))
    suppressWarnings(library(purrr))
    suppressWarnings(library(arrow))
    }
)

# %% vscode={"languageId": "r"}
bootstrap_results = list(
    plate1_u2os_nest_confluence = read_parquet("./results/boot_res_plate1_u2os_nest_confluence.parquet", show_col_types = FALSE),
    plate2_u2os_nest_confluence = read_parquet("./results/boot_res_plate2_u2os_nest_confluence.parquet", show_col_types = FALSE),
    u2os_conf8000_nest_plate = read_parquet("./results/boot_res_u2os_conf8000_nest_plate.parquet", show_col_types = FALSE),
    all_conf8000_nest_cell = read_parquet("./results/boot_res_all_conf8000_nest_cell_line_pool_plate.parquet", show_col_types = FALSE)
)

bootstrap_panel_cols = list(
    plate1_u2os_nest_confluence = c("cell_line", "ablation_type"),
    plate2_u2os_nest_confluence = c("cell_line", "ablation_type"),
    u2os_conf8000_nest_plate = c("cell_line", "ablation_type"),
    all_conf8000_nest_cell = c("ablation_type")
)

# %% vscode={"languageId": "r"}
summary(select(bootstrap_results[["plate1_u2os_nest_confluence"]], partial_r2_x2, r2_restricted))

# %% vscode={"languageId": "r"}
# unique ablation types
ablation_types <- unique(bootstrap_results[["plate1_u2os_nest_confluence"]]$ablation_type)
print(ablation_types)

# %% vscode={"languageId": "r"}
plot_partial_r2_vs_r2 <- function(
  boot_res,
  panel_cols,
  hue_col = "metric_name",
  partial_col = "partial_r2_x2",
  r2_col = "r2_restricted",
  partial_label = NULL,
  r2_label = NULL,
  n_cols = 3,
  title_wrap_width = 48,
  metric_palette = NULL,
  metric_order = NULL,
  band_fills = c(
    "<= 10% of restricted variance explained" = "#d9f0d3",
    "10-50% of restricted variance explained" = "#fee08b",
    "50-100% of restricted variance explained" = "#fcbba1"
  ),
  shade_alpha = 0.22,
  show_band_legend = TRUE,
  free_scales = "fixed"
) {
  # --------------------------
  # 1. Aggregate mean and 95% CI
  # --------------------------
  group_cols <- c(panel_cols, hue_col)

  grouped_stats <- boot_res %>%
    group_by(across(all_of(group_cols))) %>%
    summarise(
      partial_r2_mean  = mean(.data[[partial_col]], na.rm = TRUE),
      partial_r2_lower = quantile(.data[[partial_col]], 0.025, na.rm = TRUE),
      partial_r2_upper = quantile(.data[[partial_col]], 0.975, na.rm = TRUE),
      r2_restricted_mean  = mean(.data[[r2_col]], na.rm = TRUE),
      r2_restricted_lower = quantile(.data[[r2_col]], 0.025, na.rm = TRUE),
      r2_restricted_upper = quantile(.data[[r2_col]], 0.975, na.rm = TRUE),
      .groups = "drop"
    )

  # --------------------------
  # 2. Build facet labels from panel columns
  # --------------------------
  grouped_stats <- grouped_stats %>%
    mutate(
      panel_label = pmap_chr(
        select(., all_of(panel_cols)),
        function(...) {
          vals <- list(...)
          txt <- paste(
            purrr::map2_chr(panel_cols, vals, ~ paste0(.x, "=", .y)),
            collapse = " | "
          )
          str_wrap(txt, width = title_wrap_width)
        }
      )
    )

  if (!is.null(metric_order)) {
    grouped_stats[[hue_col]] <- factor(
      grouped_stats[[hue_col]],
      levels = metric_order
    )
  }

  # --------------------------
  # 3. Build shaded threshold regions only
  # partial R² = (c * x) / (1 - x)
  # --------------------------
  x_all <- seq(0.001, 0.999, length.out = 800)

  curve_fun <- function(r, x) {
    y <- (r * x) / (1 - x)
    x_max <- 1 / (1 + r)
    y[x >= x_max] <- NA_real_
    y
  }

  band_df <- tibble::tibble(
    x = x_all,
    y10 = curve_fun(0.1, x_all),
    y50 = curve_fun(0.5, x_all),
    y100 = curve_fun(1.0, x_all)
  )

  band_low <- band_df %>%
    transmute(
      x = x,
      ymin = 0,
      ymax = y10,
      band = "<= 10% of restricted variance explained"
    ) %>%
    filter(!is.na(ymax))

  band_mid <- band_df %>%
    transmute(
      x = x,
      ymin = y10,
      ymax = y50,
      band = "10-50% of restricted variance explained"
    ) %>%
    filter(!is.na(ymin), !is.na(ymax))

  band_high <- band_df %>%
    transmute(
      x = x,
      ymin = y50,
      ymax = y100,
      band = "50-100% of restricted variance explained"
    ) %>%
    filter(!is.na(ymin), !is.na(ymax))

  band_plot_df <- bind_rows(band_low, band_mid, band_high) %>%
    mutate(
      band = factor(
        band,
        levels = c(
          "<= 10% of restricted variance explained",
          "10-50% of restricted variance explained",
          "50-100% of restricted variance explained"
        )
      )
    )

  # --------------------------
  # 4. Default axis labels
  # --------------------------
  if (is.null(partial_label)) {
    partial_label <- paste0("Partial R² (", partial_col, ")")
  }
  if (is.null(r2_label)) {
    r2_label <- paste0("Restricted R² (", r2_col, ")")
  }

  # --------------------------
  # 5. Validate free_scales
  # --------------------------
  allowed_scales <- c("fixed", "free", "free_x", "free_y")
  if (!(free_scales %in% allowed_scales)) {
    stop("free_scales must be one of: ", paste(allowed_scales, collapse = ", "))
  }

  # --------------------------
  # 6. Base plot
  # --------------------------
  p <- ggplot(
    grouped_stats,
    aes(
      x = r2_restricted_mean,
      y = partial_r2_mean,
      color = .data[[hue_col]]
    )
  ) +
    geom_ribbon(
      data = band_plot_df,
      aes(x = x, ymin = ymin, ymax = ymax, fill = band),
      inherit.aes = FALSE,
      alpha = shade_alpha,
      color = NA
    ) +
    geom_errorbar(
      aes(
        ymin = partial_r2_lower,
        ymax = partial_r2_upper
      ),
      width = 0,
      alpha = 0.7,
      linewidth = 0.4
    ) +
    geom_segment(
      aes(
        x = r2_restricted_lower,
        xend = r2_restricted_upper,
        y = partial_r2_mean,
        yend = partial_r2_mean
      ),
      alpha = 0.7,
      linewidth = 0.4
    ) +
    geom_point(size = 2.4, alpha = 0.95) +
    facet_wrap(~ panel_label, ncol = n_cols, scales = free_scales) +
    labs(
      x = r2_label,
      y = partial_label,
      color = NULL,
      fill = NULL
    ) +
    coord_cartesian(
      xlim = c(0, NA),
      ylim = c(0, NA),
      expand = TRUE
    ) +
    theme_bw(base_size = 11) +
    theme(
      panel.grid.minor = element_blank(),
      panel.grid.major = element_line(color = "grey90", linewidth = 0.3),
      strip.text = element_text(size = 10, face = "bold"),
      legend.position = "right"
    )

  # --------------------------
  # 7. Metric color palette
  # --------------------------
  if (!is.null(metric_order)) {
    hue_levels <- metric_order
  } else {
    hue_levels <- sort(unique(as.character(grouped_stats[[hue_col]])))
  }

  if (!is.null(metric_palette)) {
    if (is.null(names(metric_palette))) {
      if (length(metric_palette) < length(hue_levels)) {
        stop("Unnamed metric_palette must have at least as many colors as hue levels.")
      }
      names(metric_palette) <- hue_levels
    }
    p <- p + scale_color_manual(
      values = metric_palette,
      breaks = hue_levels
    )
  } else {
    default_metric_cols <- scales::hue_pal()(length(hue_levels))
    names(default_metric_cols) <- hue_levels
    p <- p + scale_color_manual(
      values = default_metric_cols,
      breaks = hue_levels
    )
  }

  p <- p + scale_fill_manual(
    values = band_fills,
    drop = FALSE
  )

  if (!show_band_legend) {
    p <- p + guides(fill = "none")
  }

  return(p)
}

# %% vscode={"languageId": "r"}
metric_pal <- c(
  "lpips"   = "#5E3C99",  # deep purple
  "dists"   = "#1F78B4",  # strong blue

  "ssim"    = "#1B9E77",  # green
  "foreground_ssim" = "#66C2A5",  # lighter related green

  "psnr"    = "#E66101",  # orange
  "foreground_psnr" = "#FDB863",   # lighter related orange

  "mae" = "#CCCCCC"  # gray
)

for (analysis_name in names(bootstrap_results)) {

  for (ablation_type in ablation_types) {

    p <- plot_partial_r2_vs_r2(
      boot_res = bootstrap_results[[analysis_name]] %>% filter(ablation_type == !!ablation_type),
      panel_cols = bootstrap_panel_cols[[analysis_name]],
      hue_col = "metric_name",
      partial_col = "partial_r2_x2",
      r2_col = "r2_restricted",
      partial_label = "Partial R²",
      r2_label = "Restricted R²",
      n_cols = 1,
      shade_alpha = 0.4,
      free_scales = "free",
      metric_palette = metric_pal,
      metric_order = c("dists", "lpips", "foreground_ssim", "ssim", "foreground_psnr", "psnr", "mae")
    )

    ggsave(
      paste0(
        "./plots/fig_panels/",
        analysis_name,
        "_", 
        ablation_type, 
        ".pdf"
      ),
      p,
      width = 7.5,
      height = 4.5,
    )

  }

}
