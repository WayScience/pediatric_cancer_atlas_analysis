"""
generic_hook.py

Hooks to pre-define image transformations (ablations) that can be
evaluated by the runner. 
This generic hook modularization is designed to allow for future
    expandability to other transformation backends beyond Albumentations
    (e.g. kornia). 

Helpers:
- TransformBackend: Protocol defining the interface for transformation backends.
- _normalize_for_hash: Normalize a dict for stable hashing.
- _stable_hash_from_params: Generate a stable hash from transformation parameters.
- _seed_from_path: Generate a seed from a file path and optional salt.
Classes:
- GenericTransformHook: A generic image transformation hook defining behavior
    when loading tiff images up to applying the transformations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Protocol, Tuple
import json
import hashlib

import numpy as np
import tifffile as tiff

from ..ablation_runner import AugVariant
from .normalization import BitDepthNormalizer
from ..utils import sanitize_for_json


class TransformBackend(Protocol):
    name: str
    def apply(self, img: np.ndarray, *, seed: Optional[int] = None) -> np.ndarray:
        ...
    def describe(self) -> Tuple[str, Dict[str, Any]] :
        ...


def _normalize_for_hash(obj):
    """
    Recursively normalize to a hash-stable structure:
       - sort dict keys
       - convert tuples to lists
       - round floats to 8 dp
       - drop volatile albumentations keys
    """
    # keep 'transforms' only if it's the explicit list of sub-transforms; 
    # otherwise drop metadata
    VOLATILE = {
        "__class_fullname__", "id", "random_state", "replay", "save_key", 
        "applied", "transforms"
    }  
    if isinstance(obj, dict):
        return {k: _normalize_for_hash(v) 
                for k, v in sorted(obj.items()) 
                if k not in VOLATILE}
    if isinstance(obj, (list, tuple)):
        return [_normalize_for_hash(v) for v in obj]
    if isinstance(obj, float):
        return round(obj, 8)
    return obj


def _stable_hash_from_params(params_dict: dict) -> str:
    norm = _normalize_for_hash(params_dict)
    s = json.dumps(norm, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:16]


def _seed_from_path(path: Path, salt: str = "") -> int:
    h = hashlib.blake2b((str(path) + salt).encode("utf-8"), digest_size=8).hexdigest()
    return int(h, 16) % (2**31 - 1)


class GenericTransformHook:
    """
    Generic ablation hook modularized for future expandability with
    packages other than albumentations that performs image transformation
    """

    def __init__(
        self,
        backend: TransformBackend,
        *,
        variant_prefix: str = "xform",
        fixed_seed: Optional[int] = None,
        per_chan: bool = False,
        variant_name: Optional[str] = None,
        normalizer: Optional[BitDepthNormalizer] = None,
        return_original_dtype: bool = False,
    ):
        """
        :param backend: TransformBackend instance defining the transformation.
        :param variant_prefix: Prefix for the variant id.
        :param fixed_seed: If provided, use this fixed seed for all images.
            Otherwise, seed is generated per-image based on path hash.
        :param per_chan: If True, apply transform per channel (for multi-channel images).
        :param variant_name: If provided, override the variant name used in the id.
        :param normalizer: Optional normalizer to apply before transformation.
            Expected to be of type BitDepthNormalizer now to cater to
            albumentations' need for 0-1 float images for ablation to work
            expectedly. In the future other normalizers may be supported.
        :param return_original_dtype: If True, include original dtype and norm_info in AugVariant
        """
        
        self.backend = backend
        self.variant_prefix = variant_prefix
        self.fixed_seed = fixed_seed
        self.per_chan = per_chan
        self.variant_name_override = variant_name

        self.normalizer = normalizer or BitDepthNormalizer()
        self.return_original_dtype = return_original_dtype

        # capture a static description for variant id
        tname, tparams = self.backend.describe()
        self._tname = tname
        self._tparams = tparams
        self._param_hash = _stable_hash_from_params(tparams)
        self.config_id = f"{self.backend.name}:{self._tname}:{self._param_hash}"

    def __call__(self, src_path: Path) -> Iterable[AugVariant]:
        """
        Apply the transformation to the image at src_path. 
        Generates per image seed to ensure reproducibility while allowing
        variation of transformations across images. 

        Only works for hw or chw images for now. 
        By default normalizes images by their bit depth to [0, 1],
            as most transformation packages can handle a normalized float.

        :param src_path: Path to source tiff image.
        :yield: AugVariant with transformed image and metadata. 
        """

        # pick seed
        if self.fixed_seed is not None:
            seed = int(self.fixed_seed)
        else:
            seed = _seed_from_path(
                Path(src_path).stem, # only use stem
                salt=self.backend.name + self._param_hash
            )
        
        # normalize and cast to float32 by default to minimize compatibility issues
        raw = tiff.imread(str(src_path))

        # normalize via configured normalizer
        img, norm_info = self.normalizer.normalize(
            raw, path=src_path
        )

        # apply per-channel or full-image transform
        if self.per_chan and img.ndim == 3 and img.shape[0] > 1:
            slices = []
            for z in range(img.shape[0]):
                chan_img = img[z, ...]
                out = self.backend.apply(chan_img, seed=seed + z)
                slices.append(out)
            out = np.stack(slices, axis=0) # back to chw
        else:
            out = self.backend.apply(img, seed=seed) # all backends should return chw

        # optional inverse normalization to original dtype
        if self.return_original_dtype:
            out_for_save = self.normalizer.denormalize(out, norm_info)
            image_payload = out_for_save
        else:
            image_payload = out

        # build variant id: prefix-backend-transform-shortHash
        base_name = self.variant_name_override or f"{self.backend.name}-{self._tname}"
        variant = f"{self.variant_prefix}_{base_name}_{self._param_hash}"

        params = {
            "backend": self.backend.name,
            "transform_name": self._tname,
            "transform_params": sanitize_for_json(self._tparams),
            "param_hash": self._param_hash,
            "config_id": self.config_id,
            "seed_strategy": "fixed" if self.fixed_seed is not None else "per_image_hash",
            "per_chan": self.per_chan,
            "normalizer": getattr(self.normalizer, "name", type(self.normalizer).__name__),
        }

        # Sanitize norm_info to ensure numpy types are converted to Python native types
        sanitized_norm_info = sanitize_for_json(norm_info) if norm_info else None

        yield AugVariant(
            variant=variant, 
            image=image_payload, 
            params=params,
            orig_dtype=str(raw.dtype),
            norm_info=sanitized_norm_info,
        )
