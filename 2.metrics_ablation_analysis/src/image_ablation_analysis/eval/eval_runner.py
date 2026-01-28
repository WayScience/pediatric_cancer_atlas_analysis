"""
eval_metrics.py

Evaluate metrics on image ablation analysis results.
"""

import pathlib
from typing import Optional, Dict

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pyarrow as pa
import pyarrow.parquet as pq

import torch
from torch.utils.data import DataLoader

from .eval_utils import ImagePairDataset
from .metrics import MetricSpec
from ..indexing import ParquetIndex
from ..hooks.normalization import BitDepthNormalizer


class EvalRunner:
    """
    Helper class to evaluate metrics on image ablation analysis results.
    
    Initializes with a file index or directory containing the index file.
    Internally creates a dataset from the index to load ablated and original images
    Provides a run() method to compute specified metrics on the image pairs    
    """
    def __init__(
        self, 
        index_df: Optional[pd.DataFrame] = None,
        index_dir: Optional[pathlib.Path] = None,
        normalizer: BitDepthNormalizer = BitDepthNormalizer(),
    ):
        """
        Initialize EvalRunner with the given file index.
        
        :param index_df: DataFrame containing the file index (provide index or index_dir).
        :param index_dir: Directory containing the index file (provide index or index_dir).
        """

        if index_df is not None:

            self.index_df: pd.DataFrame = index_df

            if not all(col in index_df.columns for col in [
                'original_abs_path', 'aug_abs_path', 'variant']):
                raise ValueError(
                    "Index DataFrame must contain 'original_abs_path', 'aug_abs_path', "
                    "and 'variant' columns.")
        
        elif index_dir is not None:

            index_dir = index_dir.resolve(strict=True)
            
            try:
                pidx = ParquetIndex(index_dir=index_dir)
            except Exception as e:
                raise RuntimeError(f"Failed to load Parquet index from {index_dir}: {e}")
            
            self.index_df: pd.DataFrame = pidx.read()

        else:

            raise ValueError("Either index or index_dir must be provided.")
        
        self.normalizer = normalizer
        self.dataset = ImagePairDataset(
            index_df=self.index_df,
            normalizer=self.normalizer,
        )

    def run(
        self, 
        out_dir: pathlib.Path,
        metrics: Dict[str, MetricSpec],
        device: torch.device = torch.device("cpu"),
        batch_size: int = 1,
        num_workers: int = 1,
        force_overwrite: bool = False,
    ):
        """
        Run evaluation of the specified metrics on the image pairs.

        :param out_dir: Directory to save the metric results as parquet files.
        :param metrics: Dictionary of metric names to MetricSpec defining the metrics to compute.
        :param device: Torch device to run the computations on (CPU or CUDA).
        :param batch_size: Batch size for data loading.
        :param num_workers: Number of worker threads for data loading.
        :param force_overwrite: If True, overwrite existing results in out_dir.
        """
        
        loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available() if device.type == "cuda" else False,
        )

        results = []
        batch_idx = 0

        with torch.no_grad():

            pbar = tqdm(loader, total=len(loader), desc="Evaluating metrics")
            
            for batch_num, (orig_batch, abl_batch, metadata) in enumerate(pbar):
                
                # Read in batch progress
                batch_parquet_file = out_dir / f"metrics_{batch_idx:06d}.parquet"
                # Index metric evaluation progress by unique identifying columns
                # - original_abs_path: the original/reference image path
                # - aug_abs_path: the augmented/ablated image path
                # - metric_name: the name of the metric being evaluated
                batch_progress_multi_idx = None
                if batch_parquet_file.exists():                    
                    try:
                        batch_progress_df = pd.read_parquet(str(
                            batch_parquet_file
                        ))
                    except Exception as e:
                        if not force_overwrite:
                            raise RuntimeError(
                                f"Failed to read existing metrics from {batch_parquet_file}: {e}. "
                                "To force evaluation, please delete the existing files."
                            )
                    batch_progress_multi_idx = pd.MultiIndex.from_frame(
                        batch_progress_df[
                            ['original_abs_path', 'aug_abs_path', 'metric_name']
                        ]
                    )               
                
                # build scaffold for queries
                queries = pd.DataFrame({
                    "original_abs_path": metadata['original_abs_path'], 
                    "aug_abs_path": metadata['aug_abs_path'],
                    "metric_name": None # placeholder to be filled in loop below
                })                
                    
                # orig_batch/abl_batch: (B, 1, H, W), float32 CPU
                orig_batch = orig_batch.to(device, non_blocking=True)
                abl_batch  = abl_batch.to(device, non_blocking=True)
                
                B = orig_batch.shape[0]
                
                metric_values = {}
                block = metadata if metadata is not None else {}
                
                for name, spec in metrics.items():

                    # complete the queries for this metric
                    # get mask of already computed entries
                    queries['metric_name'] = name
                    query_idx = pd.MultiIndex.from_frame(queries)
                    if batch_progress_multi_idx is None:
                        exists_mask = pd.Series([False]*B)
                    else:
                        exists_mask = query_idx.isin(batch_progress_multi_idx) 
                    
                    if exists_mask.all():
                        pbar.set_postfix({'Skipping computed metrics': name})
                        continue  # All entries already computed
                    
                    incompleted = np.where(~exists_mask)
                    
                    # subset the batches to compute
                    orig_batch_sub = orig_batch[incompleted]
                    abl_batch_sub  = abl_batch[incompleted]
                    B_sub = orig_batch_sub.shape[0]                                    

                    x = spec.preprocess(orig_batch_sub)
                    y = spec.preprocess(abl_batch_sub)
                    val = spec.fn(x, y)
                    
                    if isinstance(val, torch.Tensor):
                        if val.ndim == 0:
                            raise ValueError(f"Metric {name} returned a scalar value for batch size {B}, expected per-sample values.")
                        else:
                            metric_values[name] = val.view(-1)
                    else:
                        metric_values[name] = torch.full((B,), float(val), device=device)
                
                for name, vals in metric_values.items():
                    block[name] = vals.cpu().numpy()
                
                # Convert block to long format DataFrame and update parquet file
                wide_df = pd.DataFrame(block)
                # Due to the possibility of missing metrics (skipped above),
                # we need to only melt the computed metrics
                metric_cols = list(metrics.keys())
                metric_cols = [col for col in wide_df.columns if col in metric_cols]
                id_vars = [c for c in wide_df.columns if c not in metric_cols]
                long_df = wide_df.melt(
                    id_vars=id_vars,
                    value_vars=metric_cols,
                    var_name="metric_name",
                    value_name="metric_value",
                )
                
                # Write long_df to parquet file immediately after each batch
                table = pa.Table.from_pandas(long_df, preserve_index=False)
                pq.write_table(table, str(batch_parquet_file))
                
                batch_idx += 1
                pbar.set_postfix({'written_batches': batch_idx})
