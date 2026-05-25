#!/usr/bin/env python
# coding: utf-8

# # 2.4. Perform nested regression with bootstrapping of metric value against magnitude of ablation and biological covariate confluence
# 
# Assesses how each metrics is able to capture the variation in ablation magnitude while
# being unbiased across biological covariates.

# In[1]:


import pathlib
from typing import Optional

import pandas as pd
import polars as pl

from image_ablation_analysis.regression.nested_regression import (
    bootstrap_nested_regression,
    BootstrapConfig,
    ColumnSpec,
)


# ## Pathing

# In[2]:


results_dir = pathlib.Path(".") / "results"
if not results_dir.exists():
    raise FileNotFoundError(f"Results directory not found at {results_dir.resolve()}")

regression_input_data_file = results_dir / "for_regression_subsampled.parquet"
if not regression_input_data_file.exists():
    raise FileNotFoundError(f"Regression input data not found at {regression_input_data_file.resolve()}")


# ## Regression helper

# In[3]:


def summarize_r2_scatter_bootstrap(
    boot_df: pd.DataFrame,
    output_csv: Optional[str | pathlib.Path] = None,
    group_cols: tuple[str, ...] = ("metric_name", "ablation_type"),
    restricted_col: str = "r2_restricted",
    partial_col: str = "partial_r2_x2",
    ci: float = 0.95,
) -> pd.DataFrame:

    required = set(group_cols) | {"boot_idx", restricted_col, partial_col}
    missing = sorted(required - set(boot_df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    lower_q = (1 - ci) / 2
    upper_q = 1 - lower_q

    df = boot_df.copy()
    df[restricted_col] = pd.to_numeric(df[restricted_col], errors="coerce")
    df[partial_col] = pd.to_numeric(df[partial_col], errors="coerce")

    summary = (
        df.groupby(list(group_cols), dropna=False)
        .agg(
            n_boot=("boot_idx", "nunique"),

            restricted_r2_mean=(restricted_col, "mean"),
            restricted_r2_lower=(restricted_col, lambda x: x.quantile(lower_q)),
            restricted_r2_upper=(restricted_col, lambda x: x.quantile(upper_q)),

            partial_r2_mean=(partial_col, "mean"),
            partial_r2_lower=(partial_col, lambda x: x.quantile(lower_q)),
            partial_r2_upper=(partial_col, lambda x: x.quantile(upper_q)),
        )
        .reset_index()
        .sort_values(list(group_cols))
        .reset_index(drop=True)
    )

    if output_csv is not None:
        output_csv = pathlib.Path(output_csv)
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(output_csv, index=False)

    return summary


# In[4]:


regression_input = pl.read_parquet(regression_input_data_file).to_pandas()
print(len(regression_input))
regression_input.head()


# ## Shared boostrap/regression parameters
# All regression analysis will share the same dependent variable, whichare the metric values as well as the first (restricted) independent variable which will be the parameter value. The full independent variable and the groupings of regression analysis will change based on the confounding variable being tested for.

# In[5]:


regression_config = {
    "y": "metric_value",    # dependent variable, always metric value for this analysis
    "x1": "param_values",   # independent variable 1, always the ablation parameter values for this analysis
}

bootstrap_config = {
    "n_boot": 300,
    "sample_frac": 0.5,
    "replace": True,
    "standardize": False,
    "robust_cov": None,     # or "HC3"
    "min_group_size": 25,   # prevent regression on tiny groups
}


# ## Regression Analysis 1: Assessing confounding by seeding density

# In[6]:


colspec = ColumnSpec(
    group_cols=("metric_name", "ablation_type"),
    x2="seeding_density", # full regression parameters
    x2_categorical=False,
    standardize_cols=("param_values", "seeding_density"),
    **regression_config
)

cfg = BootstrapConfig(
    **bootstrap_config
)

boot_res = bootstrap_nested_regression(regression_input, colspec, cfg)
boot_res.to_parquet(results_dir / "boot_nest_confluence.parquet", index=False)


# In[7]:


summarize_r2_scatter_bootstrap(
    boot_res,
    output_csv=results_dir / "boot_nest_confluence_summary.csv",
)


# ## Regression Analysis 2: Assessing confounding by cell lines

# In[8]:


colspec = ColumnSpec(
    group_cols=("metric_name", "ablation_type"),
    x2="cell_line", # categorical var
    x2_categorical=True,
    standardize_cols=("param_values",),
    **regression_config
)

cfg = BootstrapConfig(
    **bootstrap_config
)

boot_res = bootstrap_nested_regression(regression_input, colspec, cfg)
boot_res.to_parquet(results_dir / "boot_nest_cell_line.parquet", index=False)


# In[9]:


summarize_r2_scatter_bootstrap(
    boot_res,
    output_csv=results_dir / "boot_nest_cell_line_summary.csv",
)

