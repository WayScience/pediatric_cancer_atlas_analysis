#!/usr/bin/env python
# coding: utf-8

# # 4.1b. Run Model Inferences (wGAN)
# 
# This notebook runs batch inference for trained virtual staining models (wGAN architecture) on the evaluation split of the pediatric cancer atlas imaging data.
# 
# **Workflow:**
# 1. Load the evaluation loaddata and single-cell feature tables, filtering to plates of interest.
# 2. Initialise the checkpoint index so that already-completed inference tasks are skipped on re-runs.
# 3. For each model run, load weights → iterate over (plate, row) groups → crop single-cell patches → run inference → write per-cell TIFF outputs and update the checkpoint index.
# 4. Tear down the checkpoint session, cleaning up any empty run directories.

# In[ ]:


import pathlib
import sys

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


# In[ ]:


# virtual staining model and dataset
from virtual_stain_flow.models.unet import UNet
from virtual_stain_flow.datasets.cp_loaddata_dataset import CPLoadDataImageDataset
from virtual_stain_flow.datasets.crop_cell_dataset import CropCellImageDataset
from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize


# In[ ]:


INFERENCE_DIR = pathlib.Path('/mnt/hdd20tb/vsf_inference')
INFERENCE_DIR.mkdir(exist_ok=True) 


# In[ ]:


ANALYSIS_REPO_ROOT = pathlib.Path(
    '/home/weishanli/Waylab'
    ) / 'pediatric_cancer_atlas_analysis'
CONFIG_PATH = ANALYSIS_REPO_ROOT / 'config.yml'
config = yaml.safe_load(CONFIG_PATH.read_text())

LOADDATA_FILE_PATH = ANALYSIS_REPO_ROOT / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_eval.csv'
assert LOADDATA_FILE_PATH.exists(), f"File not found: {LOADDATA_FILE_PATH}" 

SC_FEATURES_DIR = pathlib.Path(config['paths']['sc_features_path'])
assert SC_FEATURES_DIR.exists(), f"Directory not found: {SC_FEATURES_DIR}"


# In[ ]:


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


# In[ ]:


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


# In[ ]:


checked_run_path = pathlib.Path(
    '/mnt/hdd20tb/alpine_eval/checked_model_runs.csv'
)
if not checked_run_path.exists():
    raise RuntimeError(f'Checked run info file not found at {checked_run_path}')

all_run_info_df = pd.read_csv(checked_run_path)
wgan_run_info_df = all_run_info_df[all_run_info_df['architecture'] == 'wGAN']
print(f"Total wGAN runs: {len(wgan_run_info_df)}")
wgan_run_info_df.head()


# In[ ]:


set_checkpoint_index(
    checkpoint_root=pathlib.Path(
        INFERENCE_DIR
    )
)


# In[ ]:


df = get_checkpoint_index()
df.head()


# In[ ]:


def prep_crop_dataset(
    loaddata_df: pd.DataFrame,
    sc_features: pd.DataFrame = sc_features, # good for all batch 1
) -> CropCellImageDataset:
    """
    Helper invoking the virtual_stain_flow dataset initialization steps
        to prepare a CropCellImageDataset from a loaddata DataFrame.

    :param loaddata_df: DataFrame containing the loaddata information for 
        the samples to be included in the dataset
    :param sc_features: DataFrame containing the single-cell features 
        for the samples. The notebook initializes a global sc_features DataFrame
        that is used as default argument here. 
    :return: Initialized CropCellImageDataset ready for inference
    """
    cp_ids = CPLoadDataImageDataset(
            loaddata=loaddata_df,
            sc_feature=sc_features,
            pil_image_mode="I;16",
        )
    crop_ds = CropCellImageDataset.from_dataset(
        cp_ids,
        patch_size=256,
        object_coord_x_field="Metadata_Cells_Location_Center_X",
        object_coord_y_field="Metadata_Cells_Location_Center_Y",
        fov=(1080, 1080),
    )
    crop_ds.transform = MaxScaleNormalize(
        p=1,
        normalization_factor=2**16 - 1,  # normalize to [0, 1]
    )
    crop_ds.input_channel_keys = ["OrigBrightfield"]
    # Any target channel is fine for eval as we only need input BF
    crop_ds.target_channel_keys = ["OrigBrightfield"]

    return crop_ds


# In[ ]:


for i, run_row in wgan_run_info_df.reset_index(drop=True,inplace=False).iterrows():

    run_id = run_row['run_id']
    run_path = pathlib.Path(run_row["path"])
    print(f"{i}. Processing run_id {run_id} at path {run_path}...")
    try:
        model = load_model_weights(
            run_path,
            device=DEVICE,
            model_handle=UNet,
            model_config=None,
            compile_model=False
        )
    except Exception:
        model = None
        print(f"Failed to load model for run_id {run_id} at path {run_path}, skipping...")

    for conds, group in loaddata_df_sub.groupby(
        ['Metadata_Plate', 'row']
    ):  
        try:  
            dataset = prep_crop_dataset(group)
        except Exception as e:
            print(f"Error preparing dataset for conditions {conds}: {e}")
            continue    

        try:
            inference_and_checkpoint(
                model=model,
                model_metadata=run_row,
                tasks=group,
                dataset=dataset,
                output_root=pathlib.Path(INFERENCE_DIR),
                output_flat=False,
                device=DEVICE,
            )
        except Exception as e:
            print(f"Error during inference for conditions {conds}: {e}")
            continue        


# In[ ]:


teardown_checkpoint_index()

