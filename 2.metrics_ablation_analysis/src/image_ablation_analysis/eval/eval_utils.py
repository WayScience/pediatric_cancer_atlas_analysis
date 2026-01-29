"""
eval_utils.py

Utility functions for evaluation in image ablation analysis.
"""

import pathlib
from typing import Tuple, Sequence, Dict, Any, Callable

import pandas as pd
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


def validate_orig_abl(original: np.ndarray, ablated: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate original and ablated images to have the same shape and channel dimensions.
    """

    o_shape = original.shape
    a_shape = ablated.shape

    if len(o_shape) == 2:
        # insert channel dimension
        original = original[np.newaxis, :, :]
    
    if len(a_shape) == 2:
        # insert channel dimension
        ablated = ablated[np.newaxis, :, :]

    if original.shape != ablated.shape:
        raise ValueError(
            f"Original and ablated images must have the same shape after coercion. "
            f"Got original shape {original.shape} and ablated shape {ablated.shape}."
        )
    
    return original, ablated


class ImagePairDataset(Dataset):
    """
    Dataset to load pairs of original and ablated images from a given index DataFrame.
    Useful for evaluating metrics on image ablation analysis results.
    """
    def __init__(
        self, 
        index_df: pd.DataFrame, 
        normalizer: Callable, # function to normalize images, should work with (H, W) or (C, H, W) numpy arrays
        # minimal unique identifying metadata columns to return
        metadata_cols: Sequence[str] = ['variant', 'original_abs_path', 'aug_abs_path'],
    ):
        
        self.df = index_df.reset_index(drop=True)
        self.normalizer = normalizer

        missing = [c for c in (metadata_cols or []) if c not in self.df.columns]
        if missing:
            raise ValueError(f"Metadata columns {missing} not found in index dataframe.")
        
        self.metadata_cols = list(metadata_cols)

    @staticmethod
    def _serialize_metadata_value(v: Any) -> Any:
        
        if v is None or isinstance(v, (str, int, float, bool)):
            return v
        
        try:
            import numpy as np
            import pandas as pd

            if isinstance(v, (np.generic, pd.Timestamp)):
                return v.item() if hasattr(v, "item") else str(v)
        except ImportError:
            pass

        # Fallback: just stringify
        return str(v)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(
        self, 
        idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]: # type: ignore

        row = self.df.iloc[idx]
        
        orig_path = pathlib.Path(row["original_abs_path"]).resolve(strict=True)
        aug_path = pathlib.Path(row["aug_abs_path"]).resolve(strict=True)

        orig_img = tiff.imread(str(orig_path))
        abl_img = tiff.imread(str(aug_path))

        # validate/possibly crop/resize/etc
        orig_img, abl_img = validate_orig_abl(orig_img, abl_img)

        orig_img, _ = self.normalizer.normalize(orig_img, path=str(orig_path))
        abl_img, _ = self.normalizer.normalize(abl_img, path=str(aug_path))

        # np.float32 (1, H, W) → torch.float32 (1, H, W)
        orig_t = torch.from_numpy(orig_img).float()
        abl_t  = torch.from_numpy(abl_img).float()

        metadata: Dict[str, Any] = {
            col: self._serialize_metadata_value(row[col])
            for col in self.metadata_cols
        }

        return orig_t, abl_t, metadata
