"""
conftest.py

Pytest fixtures for normalization tests.
"""

from pathlib import Path

import pytest
import numpy as np
import albumentations as A
import tifffile as tiff

from image_ablation_analysis.hooks.normalization import BitDepthNormalizer
from image_ablation_analysis.hooks.albumentations import AlbumentationsBackend


"""
normalization testing fixtures
"""
@pytest.fixture
def normalizer():
    """BitDepthNormalizer instance with auto-inferred bit depth."""
    return BitDepthNormalizer()


@pytest.fixture
def normalizer_8bit():
    """BitDepthNormalizer instance with explicit 8-bit depth."""
    return BitDepthNormalizer(bit_depth=8)


@pytest.fixture
def normalizer_12bit():
    """BitDepthNormalizer instance with explicit 12-bit depth."""
    return BitDepthNormalizer(bit_depth=12)


@pytest.fixture
def normalizer_16bit():
    """BitDepthNormalizer instance with explicit 16-bit depth."""
    return BitDepthNormalizer(bit_depth=16)


@pytest.fixture
def dummy_path():
    """Dummy path for testing (normalization uses it for logging)."""
    return Path("/fake/test/image.tiff")


@pytest.fixture
def uint8_image():
    """8-bit unsigned integer test image with full range."""
    return np.array([[0, 127, 255], [64, 128, 192]], dtype=np.uint8)


@pytest.fixture
def uint16_image_8bit():
    """16-bit array containing 8-bit data (0-255 range)."""
    return np.array([[0, 127, 255], [64, 128, 192]], dtype=np.uint16)


@pytest.fixture
def uint16_image_12bit():
    """16-bit array containing 12-bit data (0-4095 range)."""
    return np.array([[0, 2047, 4095], [1024, 2048, 3072]], dtype=np.uint16)


@pytest.fixture
def uint16_image_16bit():
    """16-bit unsigned integer test image with full range."""
    return np.array([[0, 32767, 65535], [16384, 32768, 49152]], dtype=np.uint16)


@pytest.fixture
def float32_normalized():
    """Float32 image already in [0, 1] range."""
    return np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.125]], dtype=np.float32)


@pytest.fixture
def float32_unnormalized():
    """Float32 image with values outside [0, 1] range."""
    return np.array([[0.0, 127.5, 255.0], [64.0, 128.0, 192.0]], dtype=np.float32)


@pytest.fixture
def float16_normalized():
    """Float16 image already in [0, 1] range."""
    return np.array([[0.0, 0.5, 1.0], [0.25, 0.75, 0.125]], dtype=np.float16)


@pytest.fixture
def bool_image():
    """Boolean test image."""
    return np.array([[True, False, True], [False, True, False]], dtype=np.bool_)

"""
albumentation backend testing fixtures
"""


@pytest.fixture
def synthetic_image():
    """
    Create a synthetic normalized float32 single-channel image (CxHxW format).
    Values in [0, 1] range as expected by the backend after normalization.
    """
    np.random.seed(42)
    # Create a 1x64x64 image (single channel)
    img = np.random.rand(1, 64, 64).astype(np.float32)
    return img


@pytest.fixture
def backend_no_noise():
    """Backend with GaussNoise that should produce near-identical output (std=0)."""
    transform = A.GaussNoise(
        std_range=(0.0, 0.0),  # zero standard deviation
        mean_range=(0.0, 0.0),  # zero mean
        noise_scale_factor=1.0,
        p=1.0  # always apply
    )
    return AlbumentationsBackend(transform=transform)


@pytest.fixture
def backend_with_noise():
    """Backend with GaussNoise that should produce different output (std>0)."""
    transform = A.GaussNoise(
        std_range=(0.1, 0.1),  # std of 0.1
        mean_range=(0.0, 0.0),  # zero mean
        noise_scale_factor=1.0,
        p=1.0  # always apply
    )
    return AlbumentationsBackend(transform=transform)


"""
generic hook testing fixtures
"""
@pytest.fixture
def temp_uint16_image(tmp_path):
    """
    Create a synthetic uint16 16-bit image and save it as a TIFF file.
    Returns the path to the temporary TIFF file.
    """
    # Create a 16-bit image with values spanning the full range
    np.random.seed(42)
    img = np.random.randint(0, 65536, size=(2, 64, 64), dtype=np.uint16)
    
    # Save to temporary path
    temp_file = tmp_path / "test_image_uint16.tiff"
    tiff.imwrite(str(temp_file), img)
    
    return temp_file
