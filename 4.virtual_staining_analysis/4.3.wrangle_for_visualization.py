#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import yaml

import pandas as pd
import polars as pl
import glasbey


# ## Pathing

# In[2]:


ANALYSIS_REPO_ROOT = (pathlib.Path(".") / '..').resolve() 
LOADDATA_FILE_PATH = ANALYSIS_REPO_ROOT / '0.data_preprocessing' / 'data_split_loaddata'
if not LOADDATA_FILE_PATH.exists():
    raise FileNotFoundError(f"Loaddata path does not exist at {LOADDATA_FILE_PATH}")

EVAL_DIR = pathlib.Path('/mnt/hdd20tb/vsf_eval2/')


# ## Platemap Info & wrangling
# PlateID to cell_line and seeding density

# In[3]:


platemap_dir = pathlib.Path("..") / "0.data_preprocessing" / "platemaps"
platemaps = {
    file.stem: pd.read_csv(file) for file in platemap_dir.glob("*.csv")
}
for platemap, df in platemaps.items():
    platemaps[platemap]['platemap_file'] = platemap

platemap_df = pd.concat(platemaps.values(), ignore_index=True)
platemap_df.drop(columns=["barcode", "time_point"], inplace=True)
platemap_df.rename(columns={"well": "Metadata_Well"}, inplace=True)
platemap_df.head()


# ## Model Info & wrangling
# Model ID to model details 

# In[4]:


checked_run_path = pathlib.Path(
    'checked_model_runs.csv'
)
if not checked_run_path.exists():
    raise RuntimeError(f'Checked run info file not found at {checked_run_path}')

all_run_info_df = pd.read_csv(checked_run_path)
print(f"Total model runs in checked run info: {all_run_info_df.shape[0]}")

# rename columns for later merging
all_run_info_df = all_run_info_df.loc[
    :,
    [
        "run_id", "path",
        "architecture", "input_channels", "target_channels", "density", 
    ]
].copy().rename(
    columns={
        "run_id": "Metadata_Model_run_id", 
        "path": "Metadata_Model_path",
        "density": "Metadata_Model_train_density",
        "architecture": "Metadata_Model_architecture",
        "input_channels": "Metadata_Model_input_channels",
        "target_channels": "Metadata_Model_target_channels",
    }
)
# parse channel
all_run_info_df["Metadata_Model_input_channels"] = all_run_info_df[
    "Metadata_Model_input_channels"].apply(lambda x: x.strip("[]").replace(",", "").replace("'", "").strip())
all_run_info_df["Metadata_Model_target_channels"] = all_run_info_df[
    "Metadata_Model_target_channels"].apply(lambda x: x.strip("[]").replace(",", "").replace("'", "").strip())
all_run_info_df.head()


# ## Metric eval results per image
# PlateID, Model ID with metric values

# In[5]:


eval_subdirs = list(EVAL_DIR.glob("*"))
eval_subdirs = [dir for dir in eval_subdirs if dir.is_dir()]
eval_subdirs.sort() 

eval_parquet_files = []

for subdir in eval_subdirs:

    squashed_file = subdir / f"{subdir.name}_squashed.parquet"
    if squashed_file.exists():
        print(f"Squashed file already exists for {subdir.name}, skipping.")
        eval_parquet_files.append(squashed_file)
        continue

    if not any(list(subdir.glob('*'))):
        print(f"Skipping empty dir {subdir.name}")
        # also remove empty dir
        subdir.rmdir()
        continue

    # Temporarily comment out to prevent previously un-squashed files
    # from being included here because some re-runs are still in progress.

    # df = pl.scan_parquet(str(subdir / '*.parquet'), parallel="columns").collect()
    # n_rows = df.shape[0]
    # # squash to single parquet and write to disk
    # df.write_parquet(squashed_file)
    # print(f"Squashed {n_rows} rows to {squashed_file.name}")
    # del df
    # gc.collect()

    # eval_parquet_files.append(squashed_file)

print(f"Total squashed parquet files: {len(eval_parquet_files)}")

eval_df = pl.scan_parquet([str(file) for file in eval_parquet_files], parallel="columns").collect().to_pandas()
print(f"Total rows in eval_df: {eval_df.shape[0]}")

eval_df_minimal = eval_df.loc[
    :,
    [
        "Metadata_Plate", "Metadata_Well", "Metadata_Site", "platemap_file", 
        "Metadata_Model_path", "Metadata_Model_run_id", 
        "inference_file",
        "metric_name", "metric_value"
    ]
].copy()


# ## Merge as dataframe and write for later visualization

# In[6]:


eval_df_merge_platemap = pd.merge(
    eval_df_minimal,
    platemap_df,
    on=["platemap_file", "Metadata_Well"],
    how="inner"
)

eval_df_merge_model = pd.merge(
    eval_df_merge_platemap,
    all_run_info_df,
    on=["Metadata_Model_path", "Metadata_Model_run_id"],
    how="inner"
)

print(f"Total rows after merging with model info: {eval_df_merge_model.shape[0]}")

eval_df_merge_model.to_parquet(
    EVAL_DIR / "eval_enriched_merged.parquet"
)


# ### Define palettes once and load when plotting

# In[7]:


# determinisitically define cell palette using glasbey and sorted unique cell line names
unique_cell_lines = sorted(eval_df_merge_model['cell_line'].dropna().unique())
cell_palette_colors = glasbey.create_palette(palette_size=len(unique_cell_lines))
cell_palette = dict(zip(unique_cell_lines, cell_palette_colors))
with open(EVAL_DIR / "cell_palette.yaml", "w") as f:
    yaml.dump(cell_palette, f)


# manually define model colormap
architecture_palette = {
    "UNet": "#1f77b4",
    "wGAN": "#ff7f0e",
    "ConvNeXtUNet": "#2ca02c",
}

with open(EVAL_DIR / "architecture_palette.yaml", "w") as f:
    yaml.dump(architecture_palette, f)

# manually define 5 categorical channel colormap
channel_palette = {
    "OrigAGP": "#A10807",
    "OrigDNA": "#00BCD4",
    "OrigER": "#32CD32",
    "OrigMito": "#FF00FF",
    "OrigRNA": "#F9E076"
}
with open(EVAL_DIR / "channel_palette.yaml", "w") as f:
    yaml.dump(channel_palette, f)

# define 5 step ordinal seeding density colormap
unique_densities = sorted(eval_df_merge_model["seeding_density"].dropna().unique())
# Ordinal sequential palette (shades of green: light to dark)
density_colors = ["#edf8e9", "#bae4b3", "#74c476", "#31a354", "#006d2c"]
density_palette = {float(d): density_colors[i % len(density_colors)] for i, d in enumerate(unique_densities)}
with open(EVAL_DIR / "density_palette.yaml", "w") as f:
    yaml.dump(density_palette, f)

