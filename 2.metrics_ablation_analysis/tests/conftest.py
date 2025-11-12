"""
conftest.py

Shared pytest fixtures for ablation runner tests.
"""

import tempfile
from pathlib import Path

import pytest
import numpy as np
import pandas as pd
import tifffile as tiff


@pytest.fixture
def temp_images_root(tmp_path):
    """Create temporary images root directory."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()
    return images_dir


@pytest.fixture
def temp_ablation_root(tmp_path):
    """Create temporary ablation output directory."""
    ablation_dir = tmp_path / "ablations"
    ablation_dir.mkdir()
    return ablation_dir


@pytest.fixture
def synthetic_uint16_image():
    """Create a synthetic uint16 16-bit test image."""
    # Create a 3-channel uint16 image with known values
    img = np.array([
        [[1000, 2000, 3000], [4000, 5000, 6000]],
        [[10000, 20000, 30000], [40000, 50000, 60000]],
        [[100, 200, 300], [400, 500, 600]]
    ], dtype=np.uint16)
    return img


@pytest.fixture
def temp_tiff_image(temp_images_root, synthetic_uint16_image):
    """Create a temporary TIFF file with synthetic uint16 data."""
    img_path = temp_images_root / "test_image.tiff"
    tiff.imwrite(str(img_path), synthetic_uint16_image)
    return img_path


@pytest.fixture
def temp_loaddata_csv(temp_images_root, temp_tiff_image):
    """Create a temporary loaddata CSV file in CellProfiler format."""
    csv_path = temp_images_root / "loaddata.csv"
    
    # CellProfiler LoadData format requires FileName_* and PathName_* columns
    df = pd.DataFrame({
        "FileName_OrigDNA": [temp_tiff_image.name],
        "PathName_OrigDNA": [str(temp_tiff_image.parent)],
        "Metadata_Plate": ["TestPlate001"],
        "Metadata_Well": ["A01"],
        "Metadata_Site": [1],
    })
    df.to_csv(csv_path, index=False)
    return csv_path
