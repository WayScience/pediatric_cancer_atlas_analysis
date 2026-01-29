"""
Module defining masked image quality metrics for image ablation analysis.
"""

from typing import Tuple, Optional
from abc import ABC, abstractmethod

import torch
from torch.nn import Module
import torch.nn.functional as F
from kornia.filters.otsu_thresholding import otsu_threshold
from torchmetrics.image import StructuralSimilarityIndexMeasure


class MaskedMetric(Module, ABC):
    
    def __init__(
        self,
        data_range: Tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()

        self._data_range = data_range
        self._i_max = data_range[1]
        self._eps = eps
        self._dtype = dtype

    def _validate_input(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ):
        if pred.device != target.device:
            raise ValueError(
                "'pred' and 'target' must be on the same device.")
        if pred.ndim == 3:
            pred = pred.unsqueeze(0)
        elif pred.ndim != 4:
            raise ValueError("'pred' must be CHW or BCHW.")
        if target.ndim == 3:
            target = target.unsqueeze(0)
        elif target.ndim != 4:
            raise ValueError("'target' must be CHW or BCHW.")

        if pred.shape[1] != 1 or target.shape[1] != 1:
            raise ValueError(
                f"{self.__name__} expects single-channel (C=1) tensors.")

        pred = pred.to(dtype=self._dtype)
        target = target.to(dtype=self._dtype)

        return pred, target

    def _get_mask(
        self,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        
        if mask is None:
            mask, _ = otsu_threshold(target, return_mask=True)
        else:
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            elif mask.ndim != 4:
                raise ValueError("'mask' must be CHW or BCHW.")
            
        if mask.shape != target.shape:
            raise ValueError(
                f"mask shape {mask.shape} must match "
                f"target shape {target.shape}.")   

        return mask.to(device=target.device, dtype=target.dtype)
    
    @abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        raise NotImplementedError()


class ForegroundPSNR(MaskedMetric):
    
    def __init__(
        self, 
        data_range: Tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
    ):
        
        super().__init__(
            data_range=data_range,
            eps=eps,
            dtype=dtype,
        )

        self.register_buffer("_i_max_buf", 
                             torch.tensor(data_range[1], dtype=dtype))


    def forward(
        self,
        pred: torch.Tensor,            # (B,1,H,W) or (1,H,W)
        target: torch.Tensor,          # (B,1,H,W) or (1,H,W)
        mask: Optional[torch.Tensor] = None,
    ):
        
        pred, target = self._validate_input(pred, target)
        mask = self._get_mask(target, mask)
        
        # --- masked MSE ---
        diff2 = (pred - target) ** 2
        num = (diff2 * mask).sum(dim=(1, 2, 3))       # (B,)
        den = mask.sum(dim=(1, 2, 3))                 # (B,)        
        mse = (num / den.clamp_min(self._eps)).clamp_min(self._eps)
        
        i_max = self._i_max_buf.to(
            device=pred.device, dtype=pred.dtype
            )

        return 20.0 * torch.log10(i_max) - 10.0 * torch.log10(mse) # type: ignore


class ForegroundSSIM(MaskedMetric):

    def __init__(
        self, 
        kernel_size: int = 11,
        data_range: Tuple[float, float] = (0.0, 1.0),
        eps: float = 1e-12,
        dtype: torch.dtype = torch.float32,
        **kwargs
    ):
        
        super().__init__(
            data_range=data_range,
            eps=eps,
            dtype=dtype,
        )

        self._kernel_size = kernel_size

        kwargs.update(
            {
                'kernel_size': kernel_size,
                'data_range': data_range,
                'return_full_image': True
            }
        )            

        self._ssim = StructuralSimilarityIndexMeasure(
            **kwargs
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ):
        
        pred, target = self._validate_input(pred, target)
        mask = self._get_mask(target, mask) 

        # masked SSIM as a feature map
        _, ssim_map = self._ssim(pred, target)

        self._ssim.reset() # needed to free vram
        
        # determine valid pixels to apply mask using the kernel size
        # from the SSIM computation
        k = torch.ones(
            (1, 1, self._kernel_size, self._kernel_size), 
            device=ssim_map.device, 
            dtype=ssim_map.dtype
        )

        try:
            pad = self._kernel_size // 2
            mask_single_chan = mask[:, :1, :, :]
            covered = F.conv2d(mask_single_chan, k, padding=pad)
            valid = (covered >= (
                self._kernel_size * self._kernel_size - self._eps)).to(ssim_map.dtype)
            valid = valid.expand_as(ssim_map)

            num = (ssim_map * valid).sum(dim=(1,2,3))
            den = valid.sum(dim=(1,2,3)).clamp_min(self._eps)

            result = num / den

            # cleanup stuff in cuda to prevent vram leaks
            del k, pad, mask_single_chan, covered, valid, num, den

            return result

        finally:
            if 'k' in locals(): del k
