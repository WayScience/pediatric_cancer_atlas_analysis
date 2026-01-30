#!/usr/bin/env python
# coding: utf-8

# # 2.3. Perform nested regression with bootstrapping of metric value against magnitude of ablation and biological covariate confluence
# 
# Assesses how each metrics is able to capture the variation in ablation magnitude while
# being unbiased across biological covariates.

# In[1]:


import pathlib
import ast

import pandas as pd
import pyarrow.parquet as pq

from image_ablation_analysis.indexing import ParquetIndex
from image_ablation_analysis.regression.nested_regression import (
    bootstrap_nested_regression,
    BootstrapConfig,
    ColumnSpec,
)
from image_ablation_analysis.regression.visualization import plot_partial_r2_vs_r2


# ## Pathing

# In[2]:


abl_root = pathlib.Path("/mnt/hdd20tb/alsf_ablated2/").resolve(strict=True)

metrics_dir = abl_root / "results" / "metrics3"
metrics_dir = metrics_dir.resolve(strict=True)


# ## Read in the raw metric evaluation result
# Has the metric name, metric value per pair of ablated images and its raw reference

# In[3]:


dataset = pq.ParquetDataset(str(metrics_dir))

table = dataset.read()
df = table.to_pandas()

print(len(df))
print(df.head())


# ## Read in the ablation index
# Contains ablation magnitude and type metadata

# In[4]:


index = ParquetIndex(index_dir=abl_root / "ablated_index")
index_df = index.read()
index_df['param_values'] = index_df['param_values'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
index_df['param_values'] = index_df['param_values'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x)
index_df['param_swept'] = index_df['param_swept'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
index_df['param_swept'] = index_df['param_swept'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x)
index_df[["ablation_package", "ablation_type", "hash"]] = (
    index_df["config_id"]
    .str.split(":", n=2, expand=True, regex=False)
)
print(index_df.head())


# ## Merge for regression analysis

# In[5]:


merged = pd.merge(
    index_df,
    df,
    on=["original_abs_path", "aug_abs_path", "variant"]
)
print(len(merged))


# In[ ]:


colspec = ColumnSpec(
    # these columns define the stratification for nested regression
    # i.e. for each unique combination of these columns, a separate nested
    # regression + bootstrap will be performed
    group_cols=("metric_name", "cell_line", "ablation_type"),

    # dependent variable
    # for image ablation analysis this will always be "metric_value"
    y="metric_value",

    # independent variables
    x1="param_values", # restricted regression parameter
    x2="seeding_density", # full regression parameters

    # optional; if omitted, (x1, x2) will be standardized when cfg.standardize=True
    standardize_cols=("param_values", "seeding_density"),
)

cfg = BootstrapConfig(
    n_boot=300, # number of bootstrap samples

    # fraction of group size to sample per regression instance, 
    # such sampling will be done `n_boot` times to yield `n_boot` bootstrap samples
    # which is then used to compute statistics (95% CI) to establish significance
    sample_frac=0.5,

    # whether to sample with replacement, conventionally with bootstrapping 
    # we always sample with replacement
    replace=True, 

    standardize=False,   # or True for within-group z-scoring
    robust_cov=None,     # or "HC3"
    min_group_size=25,   # prevent regression on tiny groups
)

boot_res = bootstrap_nested_regression(merged, colspec, cfg)


# ## Visualize

# In[7]:


plot_partial_r2_vs_r2(
    boot_res=boot_res,
    panel_cols=["cell_line", "ablation_type"],
    hue_col="metric_name",
    partial_col="partial_r2_x2",
    r2_col="r2_restricted",
    partial_label="Partial R² (confluence)",
    r2_label="R² Restricted (higher is better)",
)

