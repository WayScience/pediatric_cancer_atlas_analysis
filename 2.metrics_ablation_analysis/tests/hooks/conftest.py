"""
conftest.py

Pytest fixtures for normalization tests.
"""

from pathlib import Path

import pytest
import numpy as np

from image_ablation_analysis.hooks.normalization import BitDepthNormalizer


@pytest.fixture
def normalizer():
    """BitDepthNormalizer instance."""
    return BitDepthNormalizer()


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
