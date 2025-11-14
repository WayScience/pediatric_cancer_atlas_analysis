"""
normalization.py

Normalization module for consistent image preprocessing
prior to applying augmentations in ablation analysis, as well as for
manageable inverting of normalization post-augmentation if needed.
"""

from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import warnings
from contextlib import contextmanager

import numpy as np

# Custom warning, formatting and context manager for cleaner warning output
class BitDepthWarning(UserWarning):
    pass


def minimal_formatwarning(message, category, filename, lineno, line=None):
    return f"{category.__name__}: {message}\n"


@contextmanager
def minimal_warnings():
    old = warnings.formatwarning
    warnings.formatwarning = minimal_formatwarning
    try:
        yield
    finally:
        warnings.formatwarning = old


class BitDepthNormalizer:
    """
    Default: scale by full bit-depth range to [0, 1] as float32.
    Assumes full dynamic range may be used; does NOT inspect image max.
    """

    name = "bitdepth"

    def __init__(self, bit_depth: Optional[int] = None):
        """
        :param bit_depth: Bit depth to assume for normalization.
            If None, inferred from image dtype when normalizing.
            Can be dangerous if image does not use full dynamic range.
        """
        self.bit_depth = bit_depth

    def normalize(
        self, 
        img: np.ndarray, 
        *, 
        path: Path
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        
        """
        Normalize image to [0, 1] float32 based on bit depth.

        :param img: Input image as numpy array.
        :param path: Path to the image file (for logging/debugging purposes).
        :return: Tuple of (normalized image, normalization info dict).
        """

        orig_dtype = img.dtype

        # some edge case dtype handling
        if orig_dtype == np.bool_:
            norm = img.astype(np.float32)
            norm_info = {
                "strategy": self.name,
                "orig_dtype": str(orig_dtype),
                "bit_depth": 1,
                "scale": 1.0,
            }
            return norm, norm_info
        
        elif np.issubdtype(orig_dtype, np.floating):
            img_min = float(np.nanmin(img))
            img_max = float(np.nanmax(img))

            if 0.0 <= img_min and img_max <= 1.0:
                # Treat as already normalized; no-op
                norm = img.astype(np.float32, copy=False)
                norm_info = {
                    "strategy": f"{self.name}_passthrough_float01",
                    "orig_dtype": str(orig_dtype),
                    "bit_depth": None,
                    "scale": 1.0,
                }
                return norm, norm_info

            if self.bit_depth is None:
                raise ValueError("Float input without clear normalization semantics.")

            # If user insists on treating it as bit-depth based:
            # e.g. 12-bit stored as float.
            if not isinstance(self.bit_depth, int) or self.bit_depth <= 0:
                raise ValueError(f"Invalid bit_depth: {self.bit_depth}")

            scale = float(2**self.bit_depth - 1)
            norm = (img.astype(np.float32) / scale)
            norm_info = {
                "strategy": f"{self.name}_float_as_int",
                "orig_dtype": str(orig_dtype),
                "bit_depth": self.bit_depth,
                "scale": scale,
            }
            return norm, norm_info
        
        # 3) Unsigned integers: main path
        elif np.issubdtype(orig_dtype, np.unsignedinteger):
            inferred = orig_dtype.itemsize * 8

            # all warnings raised are with stacklevel=0 because the stack
            # trace is not helpful at all and only drowns the actual message.
            if self.bit_depth is None:
                bit_depth = inferred
                with minimal_warnings():
                    warnings.warn(
                        f"[{path}] Bit depth not specified, auto-inferred as {bit_depth} "
                        f"from dtype {orig_dtype}. This may be incorrect if image "
                        f"doesn't use full dynamic range (e.g. 12-bit in 16-bit).",
                        BitDepthWarning,
                )
            else:
                bit_depth = self.bit_depth
                if not isinstance(bit_depth, int) or bit_depth <= 0:
                    raise ValueError(f"Invalid bit_depth: {bit_depth}")
                # Allow bit_depth < inferred (e.g. 12-bit in 16-bit) but warn.
                if bit_depth < inferred:
                    with minimal_warnings():
                        warnings.warn(
                            f"[{path}] bit_depth={bit_depth} < dtype bits={inferred}. "
                            f"Assuming high bits unused (e.g. 12-bit packed in 16-bit).",
                            BitDepthWarning,
                        )
                # If user gives > inferred, that's almost certainly wrong.
                if bit_depth > inferred:
                    with minimal_warnings():
                        warnings.warn(
                            f"[{path}] bit_depth={bit_depth} > dtype bits={inferred}. "
                            f"This is inconsistent; using {bit_depth} anyway.",
                            BitDepthWarning,
                        )

            scale = float(2**bit_depth - 1)
            norm = img.astype(np.float32) / scale

            norm_info = {
                "strategy": self.name,
                "orig_dtype": str(orig_dtype),
                "bit_depth": bit_depth,
                "scale": scale,
            }
            return norm, norm_info
        
        else:

            raise TypeError(
                f"[{path}] Unsupported image dtype for BitDepthNormalizer: {orig_dtype}. "
                f"Use unsigned ints/bool/float in [0,1], or implement a custom normalizer."
            )

    def denormalize(
        self, 
        img: np.ndarray, 
        norm_info: Dict[str, Any]
    ) -> np.ndarray:
        """
        Denormalize image back to original dtype based on stored norm_info.
        """

        orig_dtype = np.dtype(norm_info["orig_dtype"])
        scale = float(norm_info.get("scale", 1.0))
        bit_depth = norm_info.get("bit_depth", None)

        if norm_info["strategy"].endswith("passthrough_float01"):
            # Caller wants original float dtype back
            return img.astype(orig_dtype)

        if bit_depth is not None:
            max_val = float(2**bit_depth - 1)
        else:
            max_val = scale

        out = np.clip(img * scale, 0, max_val).round().astype(orig_dtype)
        return out
