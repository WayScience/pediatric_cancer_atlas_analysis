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
# # First iteration of generating metric ablation plots with example panels,
# showing two possible variants of organizing ablation results (by analysis type vs by ablation type) 

# %% vscode={"languageId": "r"}
suppressPackageStartupMessages(library(magick))
suppressWarnings(library(ggplot2))
suppressWarnings(library(cowplot))

# %% vscode={"languageId": "r"}
# --------------------------
# Helper: read first page of PDF, trim it
# --------------------------
read_pdf_trim <- function(file, density = 400) {
  img <- magick::image_read_pdf(file, density = density)[1]
  magick::image_trim(img)
}


# --------------------------
# Helper: convert image to a cowplot canvas
# x/y/width/height let you tune inner padding if needed
# --------------------------
pdf_to_panel <- function(file, density = 400,
                         x = 0.02, y = 0.02,
                         width = 0.96, height = 0.96) {
  img <- read_pdf_trim(file, density = density)
  cowplot::ggdraw() +
    cowplot::draw_image(img, x = x, y = y, width = width, height = height)
}


# --------------------------
# Helper: make one row = metric panel + example panel
# metric panel narrower, example panel wider
# --------------------------
make_pair_row <- function(metric_file, eg_file,
                          metric_rel_width = 2,
                          gap_rel_width = 0.3,
                          eg_rel_width = 5,
                          density = 400) {

  metric_plot <- pdf_to_panel(metric_file, density = density)
  eg_plot     <- pdf_to_panel(eg_file, density = density)

  spacer <- cowplot::ggdraw()  # empty plot = gap

  cowplot::plot_grid(
    metric_plot,
    spacer,
    eg_plot,
    nrow = 1,
    rel_widths = c(metric_rel_width, gap_rel_width, eg_rel_width),
    align = "h",
    axis = "tb"
  )
}


# --------------------------
# Helper: blank panel occupying one row
# --------------------------
blank_panel <- function() {
  cowplot::ggdraw()
}

# --------------------------
# Helper: add per-panel letter labels
# --------------------------
labeled_panel <- function(panel, label,
                          label_x = 0.01,
                          label_y = 0.99,
                          label_size = 14) {
  cowplot::ggdraw(panel) +
    cowplot::draw_label(
      label,
      x = label_x, y = label_y,
      hjust = 0, vjust = 1,
      fontface = "bold",
      size = label_size
    )
}

# %% [markdown]
# ## Variant 1: Group by regression analysis, show all ablation plots on the left column and example panels on the right

# %% vscode={"languageId": "r"}
save_figure_dir <- "./plots/figures/"
if (!dir.exists(save_figure_dir)) {
    dir.create(save_figure_dir, recursive = TRUE)
}

# %% vscode={"languageId": "r"}
# --------------------------
# Input files
# --------------------------
blur_metric_panel_file <- "./plots/fig_panels/plate1_u2os_nest_confluence_GaussianBlur.pdf"
blur_eg_panel_file     <- "./plots/fig_panels/GaussianBlur_sigma_limit_U2OS_density=4000_seed=1.pdf"

dilate_metric_panel_file <- "./plots/fig_panels/plate1_u2os_nest_confluence_Dilate.pdf"
dilate_eg_panel_file     <- "./plots/fig_panels/Dilate_iterations_U2OS_density=4000_seed=1.pdf"

erode_metric_panel_file <- "./plots/fig_panels/plate1_u2os_nest_confluence_Erode.pdf"
erode_eg_panel_file     <- "./plots/fig_panels/Erode_iterations_U2OS_density=4000_seed=1.pdf"

noise_metric_panel_file <- "./plots/fig_panels/plate1_u2os_nest_confluence_GaussNoise.pdf"
noise_eg_panel_file     <- "./plots/fig_panels/GaussNoise_std_range_U2OS_density=4000_seed=1.pdf"

distort_metric_panel_file <- "./plots/fig_panels/plate1_u2os_nest_confluence_GridDistortion.pdf"
distort_eg_panel_file     <- "./plots/fig_panels/GridDistortion_distort_limit_U2OS_density=4000_seed=1.pdf"

gamma_metric_panel_file <- "./plots/fig_panels/plate1_u2os_nest_confluence_RandomGamma.pdf"
gamma_eg_panel_file     <- "./plots/fig_panels/RandomGamma_gamma_limit_U2OS_density=4000_seed=1.pdf"


# --------------------------
# Build the six rows
# --------------------------
row_blur <- make_pair_row(
  blur_metric_panel_file,
  blur_eg_panel_file
)

row_dilate <- make_pair_row(
  dilate_metric_panel_file,
  dilate_eg_panel_file
)

row_erode <- make_pair_row(
  erode_metric_panel_file,
  erode_eg_panel_file
)

row_noise <- make_pair_row(
  noise_metric_panel_file,
  noise_eg_panel_file
)

row_distort <- make_pair_row(
  distort_metric_panel_file,
  distort_eg_panel_file
)

row_gamma <- make_pair_row(
  gamma_metric_panel_file,
  gamma_eg_panel_file
)

# --------------------------
# Stack rows vertically
# equal row height
# --------------------------
final_fig <- cowplot::plot_grid(
  row_blur,
  row_dilate,
  row_erode,
  row_noise,
  row_distort,
  row_gamma,
  ncol = 1,
  rel_heights = rep(1, 6),
  labels = c("A", "B", "C", "D", "E", "F"),
  label_size = 14,
  label_fontface = "bold",
  label_x = 0.01,
  label_y = 0.98,
  hjust = 0,
  vjust = 1,
  align = "v",
  axis = "lr"
)

final_fig

# %% [markdown]
# ### Figure x. Quantifying metric response to magnitude of image degradation (ablation) and confounding biological factors. 
# All panels: Left - Scatter plot visualization of bootstrapped nested regression models fitting a restricted model of linear regression of metric values on ablation magnitude (normalized) and a full model which additionally includes a continuous or categorical biological confounder. For every metric, a separate nested bootstrap is performed to yield mean partial r^2 and restricted r^2 values alongside their 95% confidence intervals, which are visualized as single scatter points with error bars. Shaded areas represent regions of the scatter plot where the partial r^2 value is equal to or smaller than a certain percentage of restricted r^2 (explains less than % of variance relative to the restricted model). In general, lower the ratio of partial r^2 to restricted r^2 indicates less confounded metric. Right - Example patches for visually comparing ablations of various magnitude against un-transformed raw patch. 
# - A. Nested regression analysis with seeding-density as confounder. Ablation type is GaussianBlur varying sigma. Here all metrics apart from `dists` seemed insensitive to added bluriness of the image signified by close to 0 restricted r^2. `dists` is mildly sensitive to bluriness yet also considerably confounded by seeding density.  
# - B. Nested regression analysis with seeding-density as confounder. Ablation type is Dilation varying iteration. Here all metrics seemed mildly sensitive to added dilation of foreground objects in the image, confounded to differing extend with lpips being the most heavily confounded. `dists` is the most sensitive whereas foreground centric version of PSNR is the least confounded.
# - C. Nested regression analysis with seeding-density as confounder. Ablation type is Erosion varying iteration. Here all metrics apart from `dists` seemed mildly sensitive to erosion of foreground objects in the image with `lpips` being confounded the most. `dists` is much more sensitive compared to the rest but also with considerable confounding. 
# - D. Nested regression analysis with seeding-density as confounder. Ablation type is GaussianNoise varying std_range. Perhaps unsuprisingly, the metrics that are the most sensitive while being least confounded are the PSNR family due to their nature of being the ratio of the max possible intensity level against the image standard deviation, making them perfect for detecting salt and pepper style noise. The human perception based deep learning metrics `lpips` and `dists` apeared to be insensitive, interestingly. 
# - E. Nested regression analysis with seeding-density as confounder. Ablation type is GridDistortion varying distort_limit. Non-deep learning based metrics all seemed to be insensitive to the induced distortion. `lpips` is mildly sensitive yet heavily confounded. `dists` is the most sensitive while being least confounded.
# - F. Nested regression analysis with seeding-density as confounder. Ablation type is RandomGamma varying gamma_limit. Here all metrics fall under the insensitive to mildly sensitive range against adjusted Gamma, which is a monotonic function re-mapping pixel values to a different range. PSNR and `dists` are among the most sensitive metrics.   

# %% vscode={"languageId": "r"}
ggsave(
  paste0(save_figure_dir, "combined_ablation_panels.pdf"),
  final_fig,
  width = 16,
  height = 24,
  device = cairo_pdf,
  bg = "white"
)

# %% [markdown]
# ## Variant 2: Group by ablation type, show all analysis types on the left column and example panels across seeding densities on the right

# %% vscode={"languageId": "r"}
# --------------------------
# Panels
# --------------------------
blur_nest_confluence_metric_panel_file <- "./plots/fig_panels/plate1_u2os_nest_confluence_GaussianBlur.pdf"
blur_nest_plate_metric_panel_file      <- "./plots/fig_panels/u2os_conf8000_nest_plate_GaussianBlur.pdf"
blur_nest_cell_metric_panel_file       <- "./plots/fig_panels/all_conf8000_nest_cell_GaussianBlur.pdf"

blur_density1000_eg_panel_file  <- "./plots/fig_panels/GaussianBlur_sigma_limit_U2OS_density=1000_seed=1.pdf"
blur_density2000_eg_panel_file  <- "./plots/fig_panels/GaussianBlur_sigma_limit_U2OS_density=2000_seed=1.pdf"
blur_density4000_eg_panel_file  <- "./plots/fig_panels/GaussianBlur_sigma_limit_U2OS_density=4000_seed=1.pdf"
blur_density8000_eg_panel_file  <- "./plots/fig_panels/GaussianBlur_sigma_limit_U2OS_density=8000_seed=1.pdf"
blur_density12000_eg_panel_file <- "./plots/fig_panels/GaussianBlur_sigma_limit_U2OS_density=12000_seed=1.pdf"


# --------------------------
# Build left-column analysis panels
# --------------------------
panel_A <- labeled_panel(
  pdf_to_panel(blur_nest_confluence_metric_panel_file),
  "A"
)

panel_B <- labeled_panel(
  pdf_to_panel(blur_nest_plate_metric_panel_file),
  "B"
)

panel_C <- labeled_panel(
  pdf_to_panel(blur_nest_cell_metric_panel_file),
  "C"
)

# Two blank rows to align with 5-row right column
panel_blank_1 <- blank_panel()
panel_blank_2 <- blank_panel()

left_column <- cowplot::plot_grid(
  panel_A,
  panel_B,
  panel_C,
  panel_blank_1,
  panel_blank_2,
  ncol = 1,
  rel_heights = rep(1, 5),
  align = "v",
  axis = "lr"
)

# --------------------------
# Build right-column example panels
# --------------------------
panel_D <- labeled_panel(
  pdf_to_panel(blur_density1000_eg_panel_file),
  "D"
)

panel_E <- labeled_panel(
  pdf_to_panel(blur_density2000_eg_panel_file),
  "E"
)

panel_F <- labeled_panel(
  pdf_to_panel(blur_density4000_eg_panel_file),
  "F"
)

panel_G <- labeled_panel(
  pdf_to_panel(blur_density8000_eg_panel_file),
  "G"
)

panel_H <- labeled_panel(
  pdf_to_panel(blur_density12000_eg_panel_file),
  "H"
)

right_column <- cowplot::plot_grid(
  panel_D,
  panel_E,
  panel_F,
  panel_G,
  panel_H,
  ncol = 1,
  rel_heights = rep(1, 5),
  align = "v",
  axis = "lr"
)

# --------------------------
# Combine columns
# left = 3 metric rows + 2 blank rows
# right = 5 example rows
# --------------------------
final_fig <- cowplot::plot_grid(
  left_column,
  right_column,
  nrow = 1,
  rel_widths = c(2, 5),   # tune this if needed
  align = "h",
  axis = "tb"
)

final_fig

# %% [markdown]
# ### Figure x. Quantifying metric response to magnitude of image degradation (ablation) and confounding biological factors - GaussianBlur. 
# - A-C. Scatter plot visualization of bootstrapped nested regression models fitting a restricted model of linear regression of metric values on ablation magnitude (normalized) and a full model which additionally includes a continuous or categorical biological confounder. For every metric, a separate nested bootstrap is performed to yield mean partial r^2 and restricted r^2 values alongside their 95% confidence intervals, which are visualized as single scatter points with error bars. Shaded areas represent regions of the scatter plot where the partial r^2 value is equal to or smaller than a certain percentage of restricted r^2 (explains less than % of variance relative to the restricted model). In general, lower the ratio of partial r^2 to restricted r^2 indicates less confounded metric. 
# - A. Nested regression analysis with seeding-density as confounder. Ablation type is GaussianBlur varying sigma. Here all metrics apart from `dists` seemed insensitive to added bluriness of the image signified by close to 0 restricted r^2. `dists` is mildly sensitive to bluriness yet also considerably confounded by seeding density.  
# - B. Nested regression analysis with categorical plate id as confounder.  Ablation type is GaussianBlur varying sigma. Here all metrics are suprisingly not confounded by plate at all. Deep learning based metrics `lpips` and `dists` are mildly sensitive whereas all other metrics are insensitive to blur.
# - C. Nested regression analysis with categorical cell type as confounder. blation type is GaussianBlur varying sigma. Here all metrics apart from `dists` seemed to insensitive to mildly sensitive to blur and seemed heavily confounded by cell type, making them sub-optimal for evaluating virtual staining model generalization across cell types.  
# - Example patches for visually comparing ablations of various magnitude against un-transformed raw patch, showing U2-OS cell line plated with seeding densities D:1,000, E: 2,000, F: 4,000, G: 8,000, and H: 12,000. 

# %% vscode={"languageId": "r"}
ggsave(
  paste0(save_figure_dir, "blur_ablation_analysis_and_eg.pdf"),
  final_fig,
  width = 16,
  height = 24,
  device = cairo_pdf,
  bg = "white"
)
