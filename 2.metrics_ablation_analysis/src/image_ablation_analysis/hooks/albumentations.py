"""
albumentations.py

Albumentations backend hook for image ablation analysis.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
try:
    import albumentations as A
except Exception:
    A = None

from .generic_hook import GenericTransformHook
from ..utils import to_hwc, to_chw


@dataclass
class AlbumentationsBackend:
    """
    Albumentations backend for GenericTransformHook. 
    Composes and applies provided Albumentations transform with optional seed
        for reproducibility.
    Also internally handles the chw>hwc and hwc>chw conversions because
        Albumentations expects HxWxC format but the base generic hook
        and downstream analysis produce/consume CxHxW format.
    """
    transform: Any  # Transform
    name: str = "albumentations"

    def apply(self, img: np.ndarray, *, seed: Optional[int] = None) -> np.ndarray:
        """
        Apply the Albumentations transform to the input image.

        :param img: Input image in CxHxW format.
        :param seed: Optional seed for reproducibility
        :return: Transformed image in CxHxW format.
        """
        
        if A is None:
            raise RuntimeError("Albumentations is not installed.")

        img_hwc = to_hwc(img)

        # Had to wrap in Compose per call to configure seed.
        # This backend is invoked by the GenericTransformHook per image,
        # receiving seeds from hashing the image path + salt for reproducible
        # yet varied transformations.
        out = A.Compose([self.transform], seed=seed)(image=img_hwc)["image"]
        
        return to_chw(out)

    def describe(self) -> Tuple[str, Dict[str, Any]]:
        """
        Try to serialize transform config.
        One source of identity for the ablation so the runner can
        determine if the transform configured the same was applied and can be skipped.

        :return: Tuple of (transform name, parameters dict).
        """
        name = self.transform.__class__.__name__
        params: Dict[str, Any] = {}
        try:
            # Compose has .to_dict(); individual transforms often stringify well
            if hasattr(self.transform, "to_dict"):
                params = self.transform.to_dict()
            else:
                params = {"repr": repr(self.transform)}
        except Exception:
            params = {"repr": repr(self.transform)}
        return name, params
    

def make_albumentations_hook(
    transform: Any,
    **kwargs: Any,
) -> GenericTransformHook:
    """
    Convenience function to create a GenericTransformHook with AlbumentationsBackend.

    :param transform: Albumentations transform to apply.
    :param kwargs: Additional arguments to GenericTransformHook.
    :return: Configured GenericTransformHook instance.
    """
    return GenericTransformHook(
        backend=AlbumentationsBackend(transform=transform),
        **kwargs
    )
