#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pathlib
from typing import Dict

import tifffile as tiff
from tqdm.auto import tqdm
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from image_ablation_analysis.indexing import ParquetIndex
from image_ablation_analysis.hooks.normalization import BitDepthNormalizer
from image_ablation_analysis.eval.metrics import (
    MetricSpec,
    metric_factory,
    to_rgb_space,
    identity,
    functional_dists,
    functional_mae,
    functional_psnr
)
from image_ablation_analysis.eval.masked_metrics import (
    ForegroundPSNR,
    ForegroundSSIM,
)
from image_ablation_analysis.eval.eval_utils import validate_orig_abl, ImagePairDataset
from image_ablation_analysis.eval.eval_runner import EvalRunner


# ## Pathing

# In[2]:


abl_root = pathlib.Path("/mnt/hdd20tb/alsf_ablated2/")

out_dir = abl_root / "results" / "metrics"
out_dir.mkdir(parents=True, exist_ok=True)

if not abl_root.resolve(strict=True).exists():
    raise FileNotFoundError(f"Ablation root path {abl_root} does not exist.")

abl_images = list(abl_root.rglob("*.tiff"))
if not abl_images:
    raise FileNotFoundError(f"No ablated images found in {abl_root}.")
else:
    print(f"Found {len(abl_images)} ablated images in {abl_root}.")


# ## Set cuda device used for accelarating metrics computation

# In[3]:


if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()
    print(f"Number of available CUDA devices: {n_devices}")
    device_dict = {}
    for i in range(n_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"CUDA Device {i}: {device_name}")
        device_dict[device_name] = torch.device(f'cuda:{i}')
else:
    print("No CUDA devices available.")
    device_dict = {"cpu": torch.device('cpu')}

try:
    device = device_dict["NVIDIA GeForce RTX 3090"]
except Exception:
    device = device_dict["cpu"]
print(f"Using device: {device}: {torch.cuda.current_device()}")


# ## Define metric functions

# In[4]:


# Due to the inconsistent way certain metrics support/handle aggregation
# some metrics need special wrappers that ensures metrics are computed per
# image (sample) within the same batch. These wrappers are imported as
# functional_{metric_name} here and used. 
metric_fns = {
    "mae": functional_mae,
    "ssim": StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none").to(device),
    "psnr": functional_psnr,
    "lpips": LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True, reduction="none").to(device),
    "dists": functional_dists,
    "foreground_ssim": ForegroundSSIM(
        kernel_size=11,
        data_range=(0.0, 1.0),
        eps=1e-12,
        dtype=torch.float32
    ).to(device),
    "foreground_psnr": ForegroundPSNR(
        data_range=(0.0, 1.0),
        eps=1e-12,
        dtype=torch.float32
    ).to(device)
}

# Deep learning metrics DISTS and LPIPS are trained on RGB images and thus 
# needs a special preprocessing step to duplicate the grayscale channels
# to RGB channels. Other metrics can use the identity function.
METRICS: Dict[str, MetricSpec] = {
    "mae": MetricSpec(
        fn=metric_fns["mae"],
        preprocess=identity,
    ),
    "ssim": MetricSpec(
        fn=metric_fns["ssim"],
        preprocess=identity,
    ),
    "psnr": MetricSpec(
        fn=metric_fns["psnr"],
        preprocess=identity,
    ),
    "lpips": MetricSpec(
        fn=metric_fns["lpips"],
        preprocess=to_rgb_space,
    ),
    "dists": MetricSpec(
        fn=metric_fns["dists"],
        preprocess=to_rgb_space,
    ),
    "foreground_ssim": MetricSpec(
        fn=metric_fns["foreground_ssim"],
        preprocess=identity,
    ),
    "foreground_psnr": MetricSpec(
        fn=metric_fns["foreground_psnr"],
        preprocess=identity,
    ),
}


# ## Checking the generated ablated images

# In[5]:


index = ParquetIndex(index_dir=abl_root / "ablated_index")
index_df = index.read()
index_df.head()


# ## Setting up the evaluation dataset from the ablated image index
# The dataset would return pairs of ablatied image and the non-ablated raw version for convenient metric computation

# In[6]:


normalizer = BitDepthNormalizer(bit_depth=16)

dataset = ImagePairDataset(index_df, normalizer, metadata_cols=[
    'variant', 'run_id', 'config_id', 
    'Metadata_Plate', 'cell_line', 'seeding_density',
    'original_abs_path', 'aug_abs_path',
    'param_fixed', 'param_swept', 'param_values'
])

loader = DataLoader(
    dataset,
    batch_size=8, 
    shuffle=False,
    num_workers=8,  # tune for I/O
    pin_memory=torch.cuda.is_available(),

)


# ## Invoke metrics evaluation runner

# In[ ]:


runner = EvalRunner(
    index_df=index_df,
    normalizer=normalizer,
)

runner.run(
    out_dir=out_dir,
    metrics=METRICS,
    device=device,
    batch_size=8,
    num_workers=4,
    force_overwrite=True,
)

