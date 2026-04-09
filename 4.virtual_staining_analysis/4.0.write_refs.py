#!/usr/bin/env python
# coding: utf-8

# # Crop reference (Cell Painting ground truth) and write to disk with file index to facilitate downstream metric evaluation

# In[1]:


import pathlib
import sys
import yaml

import tifffile as tiff
import pandas as pd
import torch

# utils
# temporary path hack before the package dependencies is determined and solved
utils_path = pathlib.Path('.') / 'src'
sys.path.append(str(utils_path))

from vs_eval_utils.nb_utils import find_git_root
from vs_eval_utils.ds_utils import prep_crop_dataset
from virtual_stain_flow.datasets.crop_cell_dataset import CropCellImageDataset


# In[2]:


REFERENCE_DIR = pathlib.Path('/mnt/hdd20tb/vsf_reference')
REFERENCE_DIR.mkdir(exist_ok=True) 


# In[3]:


ANALYSIS_REPO_ROOT = find_git_root()
CONFIG_PATH = ANALYSIS_REPO_ROOT / 'config.yml'

config_file = ANALYSIS_REPO_ROOT / 'config.yml'
if not config_file.exists():
    raise FileNotFoundError(f"Config file not found at {config_file}")
config = yaml.safe_load(config_file.read_text())

LOADDATA_FILE_PATH = ANALYSIS_REPO_ROOT / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_eval.csv'
if not LOADDATA_FILE_PATH.exists():
    raise FileNotFoundError(f"Loaddata file not found at {LOADDATA_FILE_PATH}")

SC_FEATURES_DIR = pathlib.Path(config['paths']['sc_features_path'])
if not SC_FEATURES_DIR.exists():
    raise FileNotFoundError(f"Single-cell features directory not found at {SC_FEATURES_DIR}")


# In[4]:


PLATES: list[str] | None = ["BR00143976", "BR00143977"]

loaddata_df = pd.read_csv(LOADDATA_FILE_PATH)
sc_feature_files = list(
        SC_FEATURES_DIR.glob('*_sc_normalized.parquet')
    )

sc_features = pd.DataFrame()
for plate in loaddata_df['Metadata_Plate'].unique():
    sc_features_parquet = SC_FEATURES_DIR / f'{plate}_sc_normalized.parquet'
    if not sc_features_parquet.exists():
        print(f'{sc_features_parquet} does not exist, skipping...')
        continue 
    else:
        sc_features = pd.concat([
            sc_features, 
            pd.read_parquet(
                sc_features_parquet,
                columns=['Metadata_Plate', 'Metadata_Well', 'Metadata_Site', 'Metadata_Cells_Location_Center_X', 'Metadata_Cells_Location_Center_Y']
            )
        ])

sc_multid = pd.MultiIndex.from_frame(
    sc_features[[
        'Metadata_Plate',
        'Metadata_Well',
        'Metadata_Site',
    ]]
)

if PLATES is not None:
    loaddata_df_sub = loaddata_df[
        loaddata_df['Metadata_Plate'].isin(PLATES)
    ].reset_index(drop=True)
else:
    loaddata_df_sub = loaddata_df

print(f"Subset loaddata_df to {len(loaddata_df_sub)} rows based on PLATES filter.")

loaddata_multiid = pd.MultiIndex.from_frame(
    loaddata_df_sub[[
        'Metadata_Plate',
        'Metadata_Well',
        'Metadata_Site',
    ]]
)

loaddata_df_sub = loaddata_df_sub.loc[
    loaddata_multiid.isin(sc_multid),:
]
print (f"Subset loaddata_df_sub to {len(loaddata_df_sub)} rows based on intersection with sc_features.")

def prep_crop_ds_wrap(loaddata_df: pd.DataFrame) -> CropCellImageDataset:
    """
    Wrapped helper to specify the sc_features argument for dataset preparation
    """
    try:
        return prep_crop_dataset(
            loaddata_df,
            sc_features # good for all batch 1
        )
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        raise e


# In[5]:


devices = {}

# List all CUDA devices
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        devices[name] = torch.device(f'cuda:{i}')
else:
    print("No CUDA devices available")

DEVICE = devices['NVIDIA RTX A6000']


# In[6]:


ref_index_rows = []

for conds, group in loaddata_df_sub.groupby(
        ['Metadata_Plate', 'row']
    ): 

    plate, row = conds

    write_sub_dir = REFERENCE_DIR / plate / row
    write_sub_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = prep_crop_ds_wrap(
            group
        )
    except Exception as e:
        print(f"Error processing group {conds}: {e}")
        continue

    try:
        dataset.input_channel_keys = ['OrigBrightfield']

        metadata = dataset.metadata.copy()
        metadata['patch_idx'] = metadata.groupby(
            ['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']).cumcount()
        meta_row_dicts = metadata.to_dict(orient='records')
    except Exception as e:
        print(f"Error preparing dataset for group {conds}: {e}")
        continue

    try:
        for channel in config['data']['target_channel_keys']:

            written_paths = []
            dataset.target_channel_keys = [channel]

            for i, meta in zip(
                range(len(dataset)),
                meta_row_dicts
            ):
                _, target_patch = dataset._get_raw_item(i)
                write_file = write_sub_dir /\
                    (
                        f"{meta['Metadata_Plate']}_"
                        f"{meta['Metadata_Well']}_"
                        f"{meta['Metadata_Site']}_"
                        f"{channel}_{meta['patch_idx']}.tif"
                    )

                try:
                    tiff.imwrite(
                            write_file,
                            target_patch
                        )
                    written_paths.append(write_file)

                except Exception as e:

                    err_file = write_file.with_suffix('.err')
                    err_file.touch(exist_ok=True)
                    written_paths.append(err_file)

            for path, meta in zip(written_paths, meta_row_dicts):
                _meta = meta.copy()
                _meta.update({
                    'out_path': str(path),
                    'channel': channel
                })
                ref_index_rows.append(_meta)

    except Exception as e:
        print(f"Error processing group {conds}: {e}")
        continue

ref_index_df = pd.DataFrame(ref_index_rows)
ref_index_df.to_parquet(REFERENCE_DIR / 'reference_index.parquet', index=False)

