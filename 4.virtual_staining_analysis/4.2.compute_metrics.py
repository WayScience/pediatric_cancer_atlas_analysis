#!/usr/bin/env python
# coding: utf-8

# # Using written reference crops from 4.0 and inference crops from 4.1*, compute metrics per reference-inference pair

# In[1]:


import pathlib
from typing import Dict

import tifffile as tiff
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from torch import Tensor
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

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


# ## Pathing

# In[2]:


EVAL_DIR = pathlib.Path('/mnt/hdd20tb/vsf_eval/')
EVAL_DIR.mkdir(exist_ok=True, parents=True)
EVAL_INDEX_FILE = EVAL_DIR / 'eval_index.parquet'

INFERENCE_DIR = pathlib.Path('/mnt/hdd20tb/vsf_inference') / 'checkpoint_index'
if not INFERENCE_DIR.is_dir():
    raise NotADirectoryError(f'Inference directory not found at {INFERENCE_DIR}')

CHECKED_RUNS_FILE = pathlib.Path(
    'checked_model_runs.csv'
)
if not CHECKED_RUNS_FILE.is_file():
    raise FileNotFoundError(f'Checked run info file not found at {CHECKED_RUNS_FILE}')

REFERENCE_INDEX_FILE = pathlib.Path('/mnt/hdd20tb/vsf_reference') / 'reference_index.parquet'
if not REFERENCE_INDEX_FILE.is_file():
    raise FileNotFoundError(f'Reference index parquet file not found: {REFERENCE_INDEX_FILE}')


# In[3]:


unique_id_data_cols = [
    'Metadata_Plate',
    'Metadata_Well',
    'Metadata_Site',
]

unique_id_model_cols = [
    'Metadata_Model_run_id',
    'Metadata_Model_architecture',    
    'Metadata_Model_path',
]

model_chan_map = {
    'Metadata_Model_channel': 'channel'
}


# ## Inference index
# Maps data & model identifying information to model inference output paths (path to predicted tiffs)

# In[4]:


parquet_glob = str(INFERENCE_DIR / '**/*.parquet')

# collect parquet parts as lazy frame
# while lazy do column renaming and patch_id extract in one go 
inference_lf = (
    pl.scan_parquet(parquet_glob, glob=True)
    .rename({'output_file': 'inference_file'})
    .with_columns(
        pl.col('inference_file')
        .str.extract(r'([^/\\\\]+)\.[^.]+$', group_index=1)
        .cast(pl.Int64)
        .alias('patch_idx')
    )
)

n_inferences = inference_lf.select(pl.len().alias('n')).collect().item()
print(f"Number of inferences: {n_inferences}")

inference_cols = set(inference_lf.collect_schema().names())
missing_inference_model = [c for c in unique_id_model_cols if c not in inference_cols]
if missing_inference_model:
    raise ValueError(
        f"Missing model ID columns in inference lazy frame: {missing_inference_model}. "
        f"Expected: {unique_id_model_cols}"
    )

missing_inference_data = [c for c in unique_id_data_cols if c not in inference_cols]
if missing_inference_data:
    raise ValueError(
        f"Missing data ID columns in inference lazy frame: {missing_inference_data}. "
        f"Expected: {unique_id_data_cols}"
    )


# ## Model index
# Maps model identifying information to model metadata

# In[5]:


# this is a small index file for all trained model so load as materialized
# pd dataframe and do renaming and channel extraction in one go, then 
# move back to polars lf for later merging in lazy
all_run_info_df = pd.read_csv(CHECKED_RUNS_FILE)
all_run_info_df['channel'] = all_run_info_df['target_channels'].copy().apply(
    lambda x: eval(x)[0]
)
all_run_info_df = all_run_info_df.loc[:, ['run_id', 'architecture', 'channel', 'path']]
all_run_info_df.columns = [f'Metadata_Model_{col}' for col in all_run_info_df.columns]
print(f"Total runs: {len(all_run_info_df)}")

if not all(col in all_run_info_df for col in unique_id_model_cols):
    raise ValueError(f'Not all unique model ID columns found in run info dataframe. Expected columns: {unique_id_model_cols}')

all_run_info_lf = pl.from_pandas(all_run_info_df).lazy()


# ## Reference index
# Maps data identifying information to reference file paths

# In[6]:


reference_lf = (
    pl.scan_parquet(str(REFERENCE_INDEX_FILE))
    .rename({'out_path': 'reference_file'})
)

n_references = reference_lf.select(pl.len().alias('n')).collect().item()
print(f"Number of references: {n_references}")

reference_cols = set(reference_lf.collect_schema().names())
missing_reference_data = [c for c in unique_id_data_cols if c not in reference_cols]
if missing_reference_data:
    raise ValueError(
        f"Missing data ID columns in reference lazy frame: {missing_reference_data}. "
        f"Expected: {unique_id_data_cols}"
    )


# ## Build metric eval dataset
# Data wrangling to get to a dataframe with mapped reference path to inference path

# In[7]:


# merge run info into inference df for model pred - channel mapping
inference_enriched_lf = inference_lf.join(
    all_run_info_lf,
    on=unique_id_model_cols,
    how="inner",
)

eval_lf = reference_lf.join(
    inference_enriched_lf,
    left_on=unique_id_data_cols + ["patch_idx", "channel"],
    right_on=unique_id_data_cols + ["patch_idx", "Metadata_Model_channel"],
    how="inner",
)

# materialize here
eval_df = eval_lf.collect().to_pandas()
print(eval_df.shape)
eval_df.head()


# ### Visualize a couple of reference predict pairs

# In[8]:


n = 10
seed = 42

sampled = eval_df.sample(n=n, random_state=seed)

fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))

if n == 1:
    axes = np.array([axes])

for i, (_, row) in enumerate(sampled.iterrows()):
    ref_img = tiff.imread(row["reference_file"])
    inf_img = tiff.imread(row["inference_file"])

    axes[i, 0].imshow(ref_img[0], cmap="gray")
    axes[i, 0].set_title("reference")
    axes[i, 0].axis("off")

    axes[i, 1].imshow(inf_img[0], cmap="gray")
    axes[i, 1].set_title("inference")
    axes[i, 1].axis("off")

plt.tight_layout()
plt.show()


# ## Configure Evaluation

# In[9]:


if not EVAL_INDEX_FILE.exists():
    eval_index = pd.DataFrame()
else:
    eval_index = pd.read_parquet(EVAL_INDEX_FILE)


# ### Select CUDA device

# In[10]:


CUDA_PREF = 1 # second GPU

if torch.cuda.is_available():
    n_devices = torch.cuda.device_count()
    print(f"Number of available CUDA devices: {n_devices}")
    for i in range(n_devices):
        device_name = torch.cuda.get_device_name(i)
        print(f"CUDA Device {i}: {device_name}")    

    if n_devices >= 2:
        # Attempt using the preferred device index
        device = torch.device(f'cuda:{CUDA_PREF}')
        print(f"Selected device: {torch.cuda.get_device_name(device.index)}")
    else:
        # fallback to the first GPU if only one is available
        device = torch.device('cuda:0') 
        print(f"Only one CUDA device available. Using device: {torch.cuda.get_device_name(device.index)}")
else:
    raise RuntimeError("No CUDA devices available.")

print(f"Using device: {device}")


# ### Configure metrics

# In[11]:


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


# ### Eval helpers

# In[12]:


def _load_tiff_as_float_hw(
    path: str | pathlib.Path,
    dtype: np.dtype = np.float16
) -> np.ndarray:
    """
    Returns a numpy array (H, W) float32.
    """
    arr = tiff.imread(str(path))
    # handle common shapes: (H,W) or (1,H,W) or (H,W,1)
    if arr.ndim == 3:
        # (1,H,W)
        if arr.shape[0] == 1:
            arr = arr[0]
        # (H,W,1)
        elif arr.shape[-1] == 1:
            arr = arr[..., 0]
        else:
            raise ValueError(f"Unexpected TIFF shape {arr.shape} for {path}")
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D TIFF, got shape {arr.shape} for {path}")
    return np.expand_dims(arr.astype(dtype, copy=False), axis=0) # add channel dim


class ModelEvalDataset(torch.utils.data.Dataset):
    """
    Loads pairs of reference and inference TIFF images based on the provided index DataFrame.
    """

    def __init__(
        self, 
        index: pd.DataFrame,
        gt_col: str = 'ref_path',
        pred_col: str = 'pred_path',
        dtype: np.dtype = np.float16
    ):
        self.index = index
        self.metadata_rows: list[dict] = index.to_dict(orient='index')
        if not gt_col in index.columns:
            raise ValueError(f"Ground truth column '{gt_col}' not found in index DataFrame.")
        if not pred_col in index.columns:
            raise ValueError(f"Prediction column '{pred_col}' not found in index DataFrame.")

        self.gt_col = gt_col
        self.pred_col = pred_col
        self.dtype = dtype

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> tuple[np.ndarray, np.ndarray, dict]:
        return (
            _load_tiff_as_float_hw(
                self.index.iloc[idx][self.gt_col], dtype=self.dtype),
            _load_tiff_as_float_hw(
                self.index.iloc[idx][self.pred_col], dtype=self.dtype),
            self.metadata_rows[idx]
        )


@torch.no_grad()
def _run_metric_spec(
    preprocess_fn,
    metric_fn,
    gt: Tensor, 
    pred: Tensor
) -> Tensor:
    """
    Runs a metric function on a batch of ground truth and predicted images.
    """

    gt_p = preprocess_fn(gt)
    pred_p = preprocess_fn(pred)

    out = metric_fn(gt_p, pred_p)

    if not torch.is_tensor(out):
        raise TypeError(f"Metric returned non-tensor: {type(out)}")

    if out.ndim == 0:
        out = out.view(1, 1).expand(gt.shape[0], 1)
    elif out.ndim == 1:
        out = out.unsqueeze(1)

    if out.ndim != 2 or out.shape[0] != gt.shape[0] or out.shape[1] != 1:
        raise ValueError(f"Metric output must be (B,1); got {tuple(out.shape)}")

    return out


# In[13]:


eval_ds = ModelEvalDataset(
    eval_df, 
    gt_col='reference_file', 
    pred_col='inference_file', 
    dtype=np.float16
)
print(f"Dataset length: {len(eval_ds)}")

eval_loader = torch.utils.data.DataLoader(
    eval_ds,
    batch_size=64,
    shuffle=False,
    num_workers=16,
)


# In[ ]:


eval_subdir = EVAL_DIR / f"eval_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
eval_subdir.mkdir(parents=True, exist_ok=False)
print(f"Created evaluation subdirectory: {eval_subdir}")

for metric_name, metric in METRICS.items():
    print(f"Computing {metric_name}...")

    metric_fn = metric.fn.eval().to(device) if isinstance(metric.fn, torch.nn.Module) else metric.fn
    metric_preprocess = (
        metric.preprocess.eval().to(device)
        if isinstance(metric.preprocess, torch.nn.Module)
        else metric.preprocess
    )

    out_file = eval_subdir / f"metric_{metric_name}.parquet"
    writer = None

    try:
        with torch.no_grad():
            for batch_idx, (gt_imgs, pred_imgs, meta) in enumerate(eval_loader):
                batch_size = gt_imgs.shape[0]
                gt_imgs = gt_imgs.to(device)
                pred_imgs = pred_imgs.to(device)

                metric_values = _run_metric_spec(
                    metric_preprocess, metric_fn, gt_imgs, pred_imgs
                )

                meta["metric_name"] = [metric_name] * batch_size
                meta["metric_value"] = metric_values.cpu().numpy().ravel().tolist()

                batch_df = pd.DataFrame(meta)
                table = pa.Table.from_pandas(batch_df, preserve_index=False)

                if writer is None:
                    writer = pq.ParquetWriter(
                        out_file,
                        table.schema,
                        compression="zstd",  # smaller than snappy
                    )

                writer.write_table(table)  # one row-group per batch
    finally:
        if writer is not None:
            writer.close()

if any(eval_subdir.iterdir()):
    print(f"Evaluation completed. Results saved in: {eval_subdir}")
else:
    print(f"Evaluation completed but no results found in: {eval_subdir}")
    eval_subdir.rmdir()

