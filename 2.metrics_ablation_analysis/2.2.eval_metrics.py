#!/usr/bin/env python
# coding: utf-8

# # 2.2. Evaluate image quality assessment metrics on ablated images generated from 2.1 against their raw reference
# 
# Metrics included in this notebook
# - MAE
# - SSIM
# - PSNR
# - LPIPS
# - DISTS
# - foreground aware SSIM and PSNR (see `metrics.py` implementation)
# 
# 

# In[1]:


import pathlib
from typing import Dict

import torch
# from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from image_ablation_analysis.indexing import ParquetIndex
from image_ablation_analysis.hooks.normalization import BitDepthNormalizer
from image_ablation_analysis.eval.metrics import (
    MetricSpec,
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
from image_ablation_analysis.eval.eval_utils import ImagePairDataset
from image_ablation_analysis.eval.eval_runner import EvalRunner


# ## Pathing

# In[ ]:


abl_root = pathlib.Path("/mnt/hdd20tb/alsf_ablated2/").resolve(strict=True)

out_dir = abl_root / "results" / "metrics"
out_dir.mkdir(parents=True, exist_ok=True)

abl_images = list(abl_root.rglob("*.tiff"))
if not abl_images:
    raise FileNotFoundError(f"No ablated images found in {abl_root}.")
else:
    print(f"Found {len(abl_images)} ablated images in {abl_root}.")


# ## Set cuda device used for accelarating metrics computation

# In[3]:


# Prefer the second GPU for this evaluation analysis by PCIE order
# I had my secondary GPU installed on the second slot, adjust as needed per your setup

if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()
    print(f"Number of available CUDA devices: {n_devices}")
    for i in range(n_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"CUDA Device {i}: {device_name}")    

    if n_devices >= 2:
        # Attempt using the second GPU (index 1) in PCIe order
        device = torch.device('cuda:1')
        print(f"Selected device: {torch.cuda.get_device_name(device.index)}")
    else:
        # fallback to the first GPU if only one is available
        device = torch.device('cuda:0') 
        print(f"Only one CUDA device available. Using device: {torch.cuda.get_device_name(device.index)}")
else:
    raise RuntimeError("No CUDA devices available.")

print(f"Using device: {device}")


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

# In[ ]:


# normalize 16-bit ablated images to [0, 1] float32
# this helps the evaluation with metrics that assume inputs are in [0, 1]
# like PSNR that is configured with data_range=1.0
normalizer = BitDepthNormalizer(bit_depth=16)

dataset = ImagePairDataset(
    # This image pair dataset takes the index dataframe from 2.1.generate_ablation
    # that tracks every single saved ablated image and its corresponding original image.
    # and returns pairs of (original, ablated) images for convenient evaluation
    index_df, 
    # this makes the dataset class call the normalizer 
    # before finally yielding the image pair
    # please see the ImagePairDataset implementation for details
    normalizer,  
    metadata_cols=[
        # append these metadata to output metrics eval for more 
        # convenient analysis downstream
        'variant', 'run_id', 'config_id', 
        'Metadata_Plate', 'cell_line', 'seeding_density',
        'original_abs_path', 'aug_abs_path',
        'param_fixed', 'param_swept', 'param_values'
    ]
)


# ## Invoke metrics evaluation runner

# In[7]:


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

