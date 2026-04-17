#!/usr/bin/env python
# coding: utf-8

# # 4.1a. Run Model Inferences (UNet)
# 
# This notebook runs batch inference for trained virtual staining models (UNet architecture) on the evaluation split of the pediatric cancer atlas imaging data.
# 
# **Workflow:**
# 1. Load the evaluation loaddata and single-cell feature tables, filtering to plates of interest.
# 2. Initialise the checkpoint index so that already-completed inference tasks are skipped on re-runs.
# 3. For each model run, load weights → iterate over (plate, row) groups → crop single-cell patches → run inference → write per-cell TIFF outputs and update the checkpoint index.
# 4. Tear down the checkpoint session, cleaning up any empty run directories.

# In[1]:


import pathlib
import sys
import yaml

import pandas as pd
import torch

# utils
# temporary path hack before the package dependencies is determined and solved
utils_path = pathlib.Path('.') / 'src'
sys.path.append(str(utils_path))
from vs_eval_utils.inference_checkpointing import ( # type: ignore
    set_checkpoint_index,
    get_checkpoint_index,
    teardown_checkpoint_index,
)
from vs_eval_utils.model_loader import load_model_weights
from vs_eval_utils.model_inference import inference_and_checkpoint
from vs_eval_utils.ds_utils import prep_crop_dataset
from vs_eval_utils.nb_utils import find_git_root


# In[2]:


# virtual staining model and dataset
from virtual_stain_flow.models.unet import UNet
from virtual_stain_flow.datasets.crop_cell_dataset import CropCellImageDataset


# In[3]:


INFERENCE_DIR = pathlib.Path('/mnt/hdd20tb/vsf_inference')
INFERENCE_DIR.mkdir(exist_ok=True) 


# In[4]:


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


# In[5]:


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
            sc_features, # good for all batch 1
            target_channel_keys=config['data']['target_channel_keys']
        )
    except Exception as e:
        print(f"Error preparing dataset: {e}")
        raise e


# In[6]:


devices = {}

# List all CUDA devices
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        print(f"Device {i}: {torch.cuda.get_device_name(i)}")
        devices[name] = torch.device(f'cuda:{i}')
else:
    print("No CUDA devices available")

DEVICE = devices['NVIDIA GeForce RTX 3090']


# In[7]:


checked_run_path = pathlib.Path(
    'checked_model_runs.csv'
)
if not checked_run_path.exists():
    raise RuntimeError(f'Checked run info file not found at {checked_run_path}')

all_run_info_df = pd.read_csv(checked_run_path)
unet_run_info_df = all_run_info_df[all_run_info_df['architecture'] == 'UNet']
print(f"Total UNet runs: {len(unet_run_info_df)}")
unet_run_info_df.head()


# In[8]:


set_checkpoint_index(
    checkpoint_root=pathlib.Path(
        INFERENCE_DIR
    )
)


# In[9]:


df = get_checkpoint_index()
df.head()


# In[10]:


for i, run_row in unet_run_info_df.reset_index(drop=True,inplace=False).iterrows():

    run_id = run_row['run_id']
    run_path = pathlib.Path(run_row["path"])
    print(f"{i}. Processing run_id {run_id} at path {run_path}...")
    try:
        model = load_model_weights(
            run_path,
            device=DEVICE,
            model_handle=UNet,
            model_config={
                "init": {
                    "in_channels": 1,
                    "out_channels": 1,
                    "depth": 4,
                }
            },
            compile_model=False
        )
    except Exception:
        model = None
        print(f"Failed to load model for run_id {run_id} at path {run_path}, skipping...")

    for conds, group in loaddata_df_sub.groupby(
        ['Metadata_Plate', 'row']
    ): 

        try:
            inference_and_checkpoint(
                model=model,
                model_metadata=run_row,
                tasks=group,
                dataset=None,
                dataset_fn=prep_crop_ds_wrap,
                output_root=pathlib.Path(INFERENCE_DIR),
                output_flat=False,
                device=DEVICE,
            )
        except Exception as e:
            print(f"Error during inference for conditions {conds}: {e}")
            continue        


# In[11]:


teardown_checkpoint_index()

