#!/usr/bin/env python
# coding: utf-8

# ## This notebook runs optimization experiments on different combination of input/target normalization image transforms using the FNet model architecture by fixing hyper parameters during training to understand the effect of normalization methods on model performance.

# In[ ]:


import pathlib
import sys
import yaml
import gc

import pandas as pd
import torch
import torch.optim as optim
import mlflow
import optuna
import joblib


# ## Read config
# Paths to image data/metadata and dependent software location, as well as channel information are obtained from the config

# In[2]:


with open(pathlib.Path('.').absolute().parent.parent / "config.yml", "r") as file:
    config = yaml.safe_load(file)


# ## Import virtual_stain_flow software 

# In[3]:


sys.path.append(config['paths']['software_path'])
print(str(pathlib.Path('.').absolute().parent.parent))

## Dataset
from virtual_stain_flow.datasets.PatchDataset import PatchDataset
from virtual_stain_flow.datasets.CachedDataset import CachedDataset

## FNet training
from virtual_stain_flow.models.fnet import FNet
from virtual_stain_flow.trainers.Trainer import Trainer

from virtual_stain_flow.transforms.MinMaxNormalize import MinMaxNormalize
from virtual_stain_flow.transforms.PixelDepthTransform import PixelDepthTransform
from virtual_stain_flow.transforms.ZScoreNormalize import ZScoreNormalize

## Metrics
from virtual_stain_flow.metrics.MetricsWrapper import MetricsWrapper
from virtual_stain_flow.metrics.PSNR import PSNR
from virtual_stain_flow.metrics.SSIM import SSIM

## callback
from virtual_stain_flow.callbacks.MlflowLogger import MlflowLogger


# ## Define paths and other train parameters

# In[4]:


## Loaddata split for train is also used for optimization
LOADDATA_FILE_PATH = pathlib.Path('.').absolute().parent.parent \
    / '0.data_preprocessing' / 'data_split_loaddata' / 'loaddata_train.csv'
assert LOADDATA_FILE_PATH.exists(), f"File not found: {LOADDATA_FILE_PATH}"

SC_FEATURES_DIR = pathlib.Path(config['paths']['sc_features_path'])
assert SC_FEATURES_DIR.exists(), f"Directory not found: {SC_FEATURES_DIR}"

## Output directories
MLFLOW_DIR = pathlib.Path('.').absolute() / 'optuna_mlflow'
MLFLOW_DIR.mkdir(parents=True, exist_ok=True)

# dump directory for optuna studies
OPTUNA_JOBLIB_DIR = pathlib.Path('.').absolute() / 'optuna_joblib'
OPTUNA_JOBLIB_DIR.mkdir(parents=True, exist_ok=True)

## Basic data generation, model convolutional depth, optimizer param and max epoch definition
# Will be using these fixed values for the normalization method optimization
PATCH_SIZE = 256
CONV_DEPTH = 5
LR = 1e-4
BETAS = (0.5, 0.9)
BATCH_SIZE = 16
EPOCHS = 1_000
PATIENCE = 20

## Channels for input and target are read from config
INPUT_CHANNEL_NAMES = config['data']['input_channel_keys']
TARGET_CHANNEL_NAMES = config['data']['target_channel_keys']


# ## Configure Normalization Transforms

# In[5]:


## Define transforms and parameters
NORM_METHODS = {
    "z_score": {
        "class": ZScoreNormalize,
        "args": {"_mean": None, "_std": None, "_always_apply": True, "_p": 1.0}
    },
    "8bit": {
        "class": PixelDepthTransform,
        "args": {"src_bit_depth": 16, "target_bit_depth": 8, "_always_apply": True, "_p": 1.0}
    },
    "min_max": {
        "class": MinMaxNormalize,
        "args": {"_normalization_factor": (2 ** 16) - 1, "_always_apply": True, "_p": 1.0}
    }
}

## Define the model output activation to be used with each output normalization
NORM_METHOD_ACTIVATION = {
    "z_score": "linear",
    "8bit": "linear",
    "min_max": "sigmoid"
}


# ## Define optimization objective functions

# In[ ]:


CACHE_DATA = True

def free_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def objective(trial, dataset, channel_name):

    # Suggest an input and targettransform
    input_transform = trial.suggest_categorical("input_transform", list(NORM_METHODS.keys()))
    target_transform = trial.suggest_categorical("target_transform", list(NORM_METHODS.keys()))

    ## Configure the dataset with normalization methods
    dataset.set_input_transform(NORM_METHODS[input_transform]["class"](**NORM_METHODS[input_transform]["args"]))
    dataset.set_target_transform(NORM_METHODS[target_transform]["class"](**NORM_METHODS[target_transform]["args"]))

    ## Cache dataset
    # Caching PatchDatasets (into RAM) can substantially improve training speed, mostly
    # due to speeding up the data shuffling process that can be slow with dynamically
    # cropping patches from large images. However, to really benefit from caching it is
    # necessary to use a cache size that fits the entire dataset (or close to doing so).
    # Consider not using the Cached Dataset if memory is limited. Training/optimization is 
    # completely functional with the dynamic PatchDataset.
    if CACHE_DATA:
        dataset = CachedDataset(
                dataset=dataset,
                prefill_cache=True
            )
    else:
        # uses the dynamic PatchDataset
        pass

    ## Setup model and optimizer
    model = FNet(depth=CONV_DEPTH, 
                 # output activation paired with target/output normalization
                 output_activation=NORM_METHOD_ACTIVATION[target_transform])
    optimizer = optim.Adam(model.parameters(), lr=LR, betas=BETAS)
    
    ## Metrics to be computed (and logged)
    metric_fns = {
        "mse_loss": MetricsWrapper(_metric_name='mse', module=torch.nn.MSELoss()),
        "ssim_loss": SSIM(_metric_name="ssim"),
        "psnr_loss": PSNR(_metric_name="psnr"),
    }

    ## Params to log with mlflow
    params = {
            "lr": LR,
            "beta0": BETAS[0],
            "beta1": BETAS[1],
            "depth": CONV_DEPTH,
            "patch_size": PATCH_SIZE,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "patience": PATIENCE,
            "input_norm": input_transform,
            "target_norm": target_transform,
            "channel_name": channel_name,
        }

    ## mlflow logger callback
    mlflow_logger_callback = MlflowLogger(
        name='mlflow_logger',
        mlflow_uri=MLFLOW_DIR / 'mlruns',
        mlflow_experiment_name='FNet_optimize_io_norm',
        mlflow_start_run_args={'run_name': f'FNet_optimize_io_norm_{channel_name}', 'nested': True},
        mlflow_log_params_args=params
    )
    
    ## Trainer
    trainer = Trainer(
        model = model,
        optimizer = optimizer,
        backprop_loss = torch.nn.L1Loss(), # MAE loss for backpropagation
        dataset = dataset,
        batch_size = BATCH_SIZE,
        epochs = EPOCHS,
        patience = PATIENCE,
        callbacks=[mlflow_logger_callback],
        metrics=metric_fns,
        device = 'cuda',
        early_termination_metric='L1Loss'
    )

    # Train the model and log validation loss
    trainer.train()
    val_loss = trainer.best_loss

    del model
    del optimizer
    del metric_fns
    del mlflow_logger_callback
    del trainer
    
    free_gpu_memory()

    return val_loss


# ## Optimize for I/O normalizationm method per Channel

# In[ ]:


N_TRIALS = 50

## Loaddata for optimization
loaddata_df = pd.read_csv(LOADDATA_FILE_PATH)
sc_features = pd.DataFrame()

## Retrieve relevant sc features by assemblying them from parquet files
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

## Create patch dataset
pds = PatchDataset(
        _loaddata_csv=loaddata_df,
        _sc_feature=sc_features,
        _input_channel_keys=INPUT_CHANNEL_NAMES,
        _target_channel_keys=TARGET_CHANNEL_NAMES,
        _input_transform=None,
        _target_transform=None,
        patch_size=PATCH_SIZE,
        verbose=False,
        patch_generation_method="random_cell",
        n_expected_patches_per_img=50,
        patch_generation_random_seed=42
    )

for channel_name in TARGET_CHANNEL_NAMES:

    ## Configure dataset channel
    pds.set_input_channel_keys(INPUT_CHANNEL_NAMES)
    pds.set_target_channel_keys(channel_name)
    ## Caching of dataset is handled within the objective function due 
    ## to the need to change normalization methods for each trial

    print(f"Beginning optimization for channel: {channel_name} for io normalization methods")

    # Load the existing study
    study_path = OPTUNA_JOBLIB_DIR / f"FNet_optimize_{channel_name}_io_norm.joblib"
    if study_path.exists():
        study = joblib.load(study_path)
    else:
        # Or create if not already existing
        study = optuna.create_study(
            direction="minimize",
            study_name=f"FNet_optimize_{channel_name}_io_norm",
            sampler=optuna.samplers.TPESampler(seed=42)
        )

    # Resume optimization and run up until N_TRIALS
    while len(study.trials) < N_TRIALS:
        study.optimize(lambda trial: objective(trial, pds, channel_name), n_trials=1)
        joblib.dump(study, study_path)
        print(f"Saved study after trial {len(study.trials)}/{N_TRIALS}")
    
    print(f"{N_TRIALS} of Normalization Method Optimization for {channel_name} completed.")

    # Print best trial results
    print(f"Best trial for channel {channel_name}:")
    print(f"  Validation Loss: {study.best_trial.value}")
    print(f"  Hyperparameters: {study.best_trial.params}")

