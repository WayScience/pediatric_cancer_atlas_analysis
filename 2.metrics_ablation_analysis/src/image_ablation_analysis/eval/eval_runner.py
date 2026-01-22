"""
eval_metrics.py

Evaluate metrics on image ablation analysis results.
"""

import pathlib
from typing import Optional, Dict

import pandas as pd
from tqdm import tqdm
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
    Class to evaluate metrics on image ablation analysis results.
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
            
            if not index_dir.resolve(strict=True).exists():
                raise FileNotFoundError(f"Index directory {index_dir} does not exist.")
            
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
    ):
        """
        Run evaluation of the specified metrics on the image pairs.
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
                
                # orig_batch/abl_batch: (B, 1, H, W), float32 CPU
                orig_batch = orig_batch.to(device, non_blocking=True)
                abl_batch  = abl_batch.to(device, non_blocking=True)
                
                B = orig_batch.shape[0]
                
                metric_values = {}
                block = metadata if metadata is not None else {}
                
                for name, spec in metrics.items():
                    x = spec.preprocess(orig_batch)
                    y = spec.preprocess(abl_batch)
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
                
                wide_df = pd.DataFrame(block)
                metric_cols = list(metrics.keys())
                id_vars = [c for c in wide_df.columns if c not in metric_cols]
                long_df = wide_df.melt(
                    id_vars=id_vars,
                    value_vars=metric_cols,
                    var_name="metric_name",
                    value_name="metric_value",
                )
                
                # Write long_df to parquet file immediately after each batch
                parquet_file = out_dir / f"metrics_{batch_idx:06d}.parquet"
                table = pa.Table.from_pandas(long_df, preserve_index=False)
                pq.write_table(table, str(parquet_file))
                
                batch_idx += 1
                pbar.set_postfix({'written_batches': batch_idx})
