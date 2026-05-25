#!/usr/bin/env python
# coding: utf-8

# # 2.3. Wrangle metric evaluation results with metadata for shared used by all downstream analysis

# In[1]:


import pathlib
import yaml
import ast

import pandas as pd
import polars as pl

from image_ablation_analysis.indexing import ParquetIndex


# ## Pathing

# In[2]:


module_config_path = pathlib.Path("..") / '2.metrics_ablation_analysis' / 'config.yml'
if not module_config_path.exists():
    raise FileNotFoundError(f"Module config file not found: {module_config_path}")
config = yaml.safe_load(module_config_path.read_text())
results_dir = pathlib.Path(".") / "results"
results_dir.mkdir(exist_ok=True) 

abl_root = pathlib.Path(config['ablation_output_path']).resolve(strict=True)

metrics_dir = abl_root / "results" / "metrics"
if not metrics_dir.exists():
    raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")


# ## Read in the raw metric evaluation result
# Has the metric name, metric value plus filepaths to the pair of ablated images and its raw reference

# In[3]:


# Load lazy here and only display schema and head to confirm the structure
lf = pl.scan_parquet(str(metrics_dir / '*.parquet'), parallel="columns")
print(lf.collect_schema().names())
print(lf.head())


# ## Read in the ablation index & some wrangling
# Contains ablation magnitude and type metadata needing for regression

# In[4]:


def wrangle_data_for_regression(df: pd.DataFrame) -> pd.DataFrame:
    """
    Post pandas materialization data wrangling helper
    """

    df['param_values'] = df['param_values'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['param_values'] = df['param_values'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x)
    df['param_swept'] = df['param_swept'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    df['param_swept'] = df['param_swept'].apply(lambda x: x[0] if isinstance(x, (list, tuple)) and len(x) == 1 else x)

    return df


# In[5]:


index = ParquetIndex(index_dir=abl_root / "ablated_index")
index_lf = index.read_lazy()

# Extract ablation package, type, and hash from config_id
# should be doable in lazy whereas those that require literal_eval or ast parsing should be done post materialization
index_lf = index_lf.with_columns(
    pl.col("config_id").str.split_exact(":", 2).alias("config_parts")
).with_columns(
    pl.col("config_parts").struct.field("field_0").alias("ablation_package"),
    pl.col("config_parts").struct.field("field_1").alias("ablation_type"),
    pl.col("config_parts").struct.field("field_2").alias("hash"),
).drop("config_parts")

print(lf.collect_schema().names())
print(lf.head())


# ## Merge metric eval output dataframe with ablation metadata and materialize to produce dataframe shared by downstream analysis

# In[6]:


for_regression_lf = index_lf.join(
    lf,
    on=["original_abs_path", "aug_abs_path", "variant"],
    how="inner",
)

for_analysis_df = for_regression_lf.collect().to_pandas()
for_analysis_df = wrangle_data_for_regression(for_analysis_df)
print(len(for_analysis_df))
for_analysis_df.to_parquet(results_dir / "for_analysis.parquet", index=False)
for_analysis_df.head()


# ### Also produce subsampled dataframe for analysis that benefit from plate + well level equal representation 

# In[ ]:


group_col = ["Metadata_Plate", "Metadata_Well", "Metadata_Site"]

# subsample to smallest group size to ensure equal representation of all conditions in regression
min_group_size = for_analysis_df.groupby(group_col).size().min()
print(f"Minimum group size across {group_col}: {min_group_size}")
max_samp_size = 200
samp_size = min(min_group_size, max_samp_size)

for_analysis_subsampled = (
    for_analysis_df.groupby(group_col)
    .sample(n=samp_size, random_state=42).reset_index(drop=True)
)
print(len(for_analysis_subsampled))

# for_analysis_subsampled.to_parquet(results_dir / "for_analysis_subsampled.parquet", index=False)
for_analysis_subsampled.head()

