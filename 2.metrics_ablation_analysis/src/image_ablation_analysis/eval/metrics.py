"""
metrics.py

Metric definitions and wrappers for image ablation analysis evaluation.
"""

from typing import Callable, Dict, List
from dataclasses import dataclass

import torch
from torch import Tensor

from torchmetrics.functional.image.dists import deep_image_structure_and_texture_similarity
from torchmetrics.functional.image import peak_signal_noise_ratio


def identity(x: torch.Tensor) -> torch.Tensor:
    """
    Identity preprocessing function.
    No-op
    """
    return x


def to_rgb_space(x: torch.Tensor) -> torch.Tensor:
    """
    Convert single channel images to 3-channel by repeating channels.
    Useful for deep learning based metrics adapted from computer vision networks
        that expects 3-channel input.

    :param x: (B, C, H, W) float32 tensor, no-op if C==3, else C must be 1
    :raises ValueError: if C not in {1, 3}
    :return: (B, 3, H, W) float32 tensor 
    """
    if x.shape[1] == 3:
        return x
    if x.shape[1] != 1:
        raise ValueError(f"Can't safely repeat from C={x.shape[1]} to 3")
    return x.repeat(1, 3, 1, 1)


@dataclass
class MetricSpec:
    """
    Specification for a metric including the function and any preprocessing.
    """
    fn: Callable[[Tensor, Tensor], Tensor]
    preprocess: Callable[[Tensor], Tensor] = identity


def metric_factory(
    metric_fns: List[Callable],
    preprocess: Callable[[Tensor], Tensor] = identity,
) -> Dict[str, MetricSpec]:
    """
    Create a dictionary of MetricSpec from a list of metric functions.

    :param metric_fns: List of metric functions taking (x, y) tensors and 
        returning per-sample tensors
    :param preprocess: Preprocessing function to apply to inputs
        before metric computation
    :return: Dictionary mapping metric function names to MetricSpec
    """
    
    metrics = {}

    for fn in metric_fns:
        name = fn.__name__ if hasattr(fn, '__name__') else str(fn)
        metrics[name] = MetricSpec(fn=fn, preprocess=preprocess)
    
    return metrics


# Specially wrapped metrics to ensure correct reduction behavior

def functional_mae(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Mean Absolute Error per image in batch, reduces across C, H, W dimensions.

    :param x: (B, C, H, W) float32 tensor
    :param y: (B, C, H, W) float32 tensor
    :return: (B,) tensor of MAE values per image in batch
    """
    return torch.nn.functional.l1_loss(x, y, reduction='none').mean(dim=(1, 2, 3))   


def functional_psnr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Wrapper of functional PSNR from lightning, specifies image wise reduction.

    :param x: (B, C, H, W) float32 tensor with values in [0, 1]
    :param y: (B, C, H, W) float32 tensor with values in [0, 1]
    :return: (B,) tensor of PSNR values per image in batch
    """
    return peak_signal_noise_ratio(x, y, data_range=1.0, reduction='none', dim=(1,2,3))


def functional_dists(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Deep Image Structure and Texture Similarity per image in batch.

    :param x: (B, C, H, W) float32 tensor with values in [0, 1]
    :param y: (B, C, H, W) float32 tensor with values in [0, 1]
    :return: (B,) tensor of DISTS values per image in batch
    """
    return deep_image_structure_and_texture_similarity(x, y, reduction='none')
