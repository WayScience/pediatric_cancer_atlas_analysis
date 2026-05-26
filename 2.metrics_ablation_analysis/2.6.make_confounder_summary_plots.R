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

# %% vscode={"languageId": "r"}
suppressWarnings(suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(ggplot2)
  library(cowplot)
  library(ggtext)
  library(gt)
}))

# %% [markdown]
# ## Define plot parameters

# %% vscode={"languageId": "r"}
metric_order <- c(
  "dists",
  "lpips",
  "foreground_ssim",
  "ssim",
  "foreground_psnr",
  "psnr",
  "mae"
 )

metric_labels <- c(
  "dists" = "<span style='color:#3B4CC0'>DISTS</span>",
  "lpips" = "<span style='color:#3B4CC0'>LPIPS</span>",
  "foreground_ssim" = "<span style='color:#F1A340'>Foreground SSIM</span>",
  "ssim" = "<span style='color:#F1A340'>SSIM</span>",
  "foreground_psnr" = "<span style='color:#F1A340'>Foreground PSNR</span>",
  "psnr" = "<span style='color:#F1A340'>PSNR</span>",
  "mae" = "<span style='color:#7F7F7F'>MAE</span>"
 )

ablation_order <- c(
  "Dilate",
  "Erode",
  "GaussianBlur",
  "GaussNoise",
  "GridDistortion",
  "RandomGamma"
 )

confounder_order <- c(
  "Cell line",
  "Seeding density / confluence"
 )

# %% [markdown]
# ## Read in plot data

# %% vscode={"languageId": "r"}
cell_summary <- read_csv(
  "results/boot_nest_cell_line_summary.csv",
  show_col_types = FALSE
) %>%
  mutate(confounder = "Cell line")

confluence_summary <- read_csv(
  "results/boot_nest_confluence_summary.csv",
  show_col_types = FALSE
) %>%
  mutate(confounder = "Seeding density / confluence")

summary_df <- bind_rows(cell_summary, confluence_summary) %>%
  mutate(
    metric_name = factor(metric_name, levels = metric_order),
    confounder = factor(confounder, levels = confounder_order)
  )

# %% [markdown]
# ## Divide partial r squared by restricted r squared to yield surrogate of burden - ratio of metric measurement of confouding factor vs metric measurement of ablation extent

# %% vscode={"languageId": "r"}
eps <- 1e-4

compute_burden_df <- function(df, confounder_label) {
  df %>%
    mutate(
      confounder = confounder_label,
      burden = (partial_r2_mean + eps) / (restricted_r2_mean + eps),
      metric_name = factor(metric_name, levels = rev(metric_order)),
      ablation_type = factor(ablation_type, levels = ablation_order)
    )
}

cell_burden_df <- compute_burden_df(cell_summary, "Cell line")
confluence_burden_df <- compute_burden_df(confluence_summary, "Seeding density / confluence")

burden_df <- bind_rows(cell_burden_df, confluence_burden_df)
head(burden_df)

# %% vscode={"languageId": "r"}
max_abs_burden <- max(abs(burden_df$burden), na.rm = TRUE)
fill_limit <- ceiling(max_abs_burden)

# %% [markdown]
# ## Plotting helpers

# %% vscode={"languageId": "r"}
plot_burden_heatmap <- function(
  df,
  title,
  show_y = TRUE,
  show_legend = TRUE,
  fill_limit = NULL,
  plot_margin = margin(5.5, 5.5, 5.5, 5.5)
) {
  p <- ggplot(
    df,
    aes(
      x = ablation_type,
      y = metric_name,
      fill = burden
    )
  ) +
    geom_tile(color = "white", linewidth = 0.6) +
    scale_y_discrete(labels = metric_labels, position = "right") +
    labs(
      title = title,
      x = "Degradation transform",
      y = NULL,
      fill = expression("(partial R"^2~"/ restricted R"^2*")")
    ) +
    theme_bw(base_size = 11) +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(size = 11, face = "bold", hjust = 0),
      axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
      axis.text.y = element_text(size = 9.5),
      legend.position = "right",
      legend.title = element_text(size = 9),
      legend.text = element_text(size = 8.5),
      plot.margin = plot_margin
    )

  if (!is.null(fill_limit)) {
    p <- p +
      scale_fill_gradient2(
        low = "#2166AC",
        mid = "white",
        high = "#B2182B",
        midpoint = 0,
        limits = c(-fill_limit, fill_limit),
        oob = scales::squish
      )
  } else {
    p <- p +
      scale_fill_gradient2(
        low = "#2166AC",
        mid = "white",
        high = "#B2182B",
        midpoint = 0
      )
  }

  if (!show_y) {
    p <- p + theme(
      axis.text.y = element_blank(),
      axis.ticks.y = element_blank()
    )
  }

  if (!show_legend) {
    p <- p + guides(fill = "none")
  }

  p
}

# %% [markdown]
# ## Plot burden heatmap for two confounder factors and combine as single panel

# %% vscode={"languageId": "r"}
p_burden_facet_base <- ggplot(
  burden_df,
  aes(
    x = ablation_type,
    y = metric_name,
    fill = burden
  )
 ) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(
    aes(label = sprintf("%.2f", burden)),
    size = 2.8,
    color = "black"
  ) +
  scale_y_discrete(labels = metric_labels, position = "right") +
  facet_grid(confounder ~ ., switch = "y") +
  labs(
    title = "Burden (partial R^2 / restricted R^2)",
    x = "Degradation transform",
    y = NULL
  ) +
  theme_bw(base_size = 11) +
  theme(
    panel.grid = element_blank(),
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = ggtext::element_markdown(size = 9.5),
    plot.title = element_text(size = 11, face = "bold", hjust = 0),
    strip.placement = "outside",
    strip.background = element_blank(),
    strip.text.y.left = element_text(angle = 90, face = "bold", size = 10),
    panel.spacing.y = unit(0.1, "lines"),
    legend.position = "none",
    plot.margin = margin(2, 8, 2, 18)
  )

if (!is.null(fill_limit)) {
  p_burden_facet_base <- p_burden_facet_base +
    scale_fill_gradient2(
      low = "#2166AC",
      mid = "white",
      high = "#B2182B",
      midpoint = 0,
      limits = c(-fill_limit, fill_limit),
      oob = scales::squish
    )
} else {
  p_burden_facet_base <- p_burden_facet_base +
    scale_fill_gradient2(
      low = "#2166AC",
      mid = "white",
      high = "#B2182B",
      midpoint = 0
    )
}

p_burden_facet_labeled <- cowplot::ggdraw(p_burden_facet_base) +
  cowplot::draw_label(
    "Confounder",
    x = 0.02,
    y = 0.5,
    angle = 90,
    vjust = 0.5,
    hjust = 0,
    fontface = "bold",
    size = 10
  )

panel_C <- cowplot::ggdraw(p_burden_facet_labeled) +
  cowplot::draw_label(
    "Metric class",
    x = 0.98,
    y = 0.12,
    hjust = 1,
    size = 8,
    fontface = "bold"
  ) +
  cowplot::draw_label(
    "Deep learning",
    x = 0.98,
    y = 0.09,
    hjust = 1,
    size = 7.5,
    color = "#3B4CC0"
  ) +
  cowplot::draw_label(
    "Conventional",
    x = 0.98,
    y = 0.065,
    hjust = 1,
    size = 7.5,
    color = "#F1A340"
  ) +
  cowplot::draw_label(
    "Pixel-wise",
    x = 0.98,
    y = 0.04,
    hjust = 1,
    size = 7.5,
    color = "#7F7F7F"
  )

panel_C

# %% vscode={"languageId": "r"}
ggsave(
  filename = "plots/fig_panels/panel_C_confounder_burden_v.pdf",
  plot = panel_C,
  width = 6,
  height = 10,
  units = "in",
  device = cairo_pdf,
  create.dir = TRUE
)

ggsave(
  filename = "plots/fig_panels/panel_C_confounder_burden_v.png",
  plot = panel_C,
  width = 6,
  height = 10,
  units = "in",
  dpi = 300,
  create.dir = TRUE
)

# %% [markdown]
# ## Make high level summary plot for which metrics have lowest burden across the board

# %% vscode={"languageId": "r"}
min_burden_winners <- burden_df %>%
  group_by(confounder, ablation_type) %>%
  filter(burden == min(burden, na.rm = TRUE)) %>%
  ungroup()

min_burden_counts <- min_burden_winners %>%
  count(metric_name, name = "n_min") %>%
  right_join(tibble(metric_name = factor(metric_order, levels = metric_order)),
    by = "metric_name"
  ) %>%
  mutate(
    n_min = ifelse(is.na(n_min), 0, n_min),
    metric_label = metric_labels[as.character(metric_name)],
    stars = strrep("*", n_min)
  )

min_burden_table <- min_burden_counts %>%
  select(metric_label, stars) %>%
  as.data.frame()

row.names(min_burden_table) <- min_burden_table$metric_label
min_burden_table$metric_label <- NULL

min_burden_table

# %% vscode={"languageId": "r"}
min_burden_gt <- min_burden_table %>%
  tibble::rownames_to_column("metric_label") %>%
  gt::gt() %>%
  gt::fmt_markdown(columns = "metric_label") %>%
  gt::cols_label(
    metric_label = "Metric",
    stars = "Min burden count"
  ) %>%
  gt::tab_options(
    table.font.size = 10,
    data_row.padding = gt::px(2)
  )

min_burden_gt

# %% vscode={"languageId": "r"}
gt::gtsave(
  data = min_burden_gt,
  filename = "plots/fig_panels/panel_C_min_burden_table.html"
 )
