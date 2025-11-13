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


# ===== LoadDataIndex Fixtures =====

@pytest.fixture
def channel_names():
    """Standard channel names for the pediatric cancer atlas."""
    return ["OrigBrightfield", "OrigER", "OrigAGP", "OrigMito", "OrigDNA", "OrigRNA"]


@pytest.fixture
def multi_channel_images(temp_images_root, channel_names, synthetic_uint16_image):
    """
    Create actual TIFF files for multiple channels, plates, wells, and sites.
    Returns a dict mapping (plate, well, site, channel) -> Path.
    """
    images = {}
    
    # Create images for 2 plates, 2 wells each, 2 sites each
    plates = ["BR00143976", "BR00143977"]
    wells = ["A01", "B02"]
    sites = [1, 2]
    
    for plate in plates:
        plate_dir = temp_images_root / plate
        plate_dir.mkdir(exist_ok=True)
        
        for well in wells:
            for site in sites:
                for channel in channel_names:
                    # Create filename following typical pattern
                    filename = f"{plate}_{well}_Site{site}_{channel}.tiff"
                    img_path = plate_dir / filename
                    
                    # Write a unique image (vary intensity by site number)
                    img_data = synthetic_uint16_image * site
                    tiff.imwrite(str(img_path), img_data)
                    
                    images[(plate, well, site, channel)] = img_path
    
    return images


@pytest.fixture
def comprehensive_loaddata_csv(temp_images_root, multi_channel_images, channel_names):
    """
    Create a comprehensive LoadData CSV with all required metadata columns
    matching the pediatric cancer atlas format.
    """
    csv_path = temp_images_root / "comprehensive_loaddata.csv"
    
    rows = []
    
    # Extract unique (plate, well, site) combinations
    unique_combinations = set()
    for (plate, well, site, channel) in multi_channel_images.keys():
        unique_combinations.add((plate, well, site))
    
    for plate, well, site in sorted(unique_combinations):
        row = {
            "Metadata_Plate": plate,
            "Metadata_Well": well,
            "Metadata_Site": site,
            "Metadata_AbsPositionZ": 100.5 + site * 10,
            "Metadata_ChannelID": f"CH{site}",
            "Metadata_Col": int(well[1:]),
            "Metadata_FieldID": f"Field{site}",
            "Metadata_PlaneID": f"Plane{site}",
            "Metadata_PositionX": 1000.0 + site * 100,
            "Metadata_PositionY": 2000.0 + site * 100,
            "Metadata_PositionZ": 50.0 + site * 5,
            "Metadata_Row": well[0],
            "Metadata_Reimaged": site % 2,  # 0 or 1
        }
        
        # Add FileName_* and PathName_* for each channel
        for channel in channel_names:
            img_path = multi_channel_images[(plate, well, site, channel)]
            row[f"FileName_{channel}"] = img_path.name
            row[f"PathName_{channel}"] = str(img_path.parent)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    return csv_path


@pytest.fixture
def multi_csv_loaddata(temp_images_root, multi_channel_images, channel_names):
    """
    Create multiple LoadData CSVs (simulating split datasets).
    Returns a list of CSV paths.
    """
    csv_paths = []
    
    # Split images by plate
    plates = ["BR00143976", "BR00143977"]
    
    for plate in plates:
        csv_path = temp_images_root / f"loaddata_{plate}.csv"
        rows = []
        
        for (p, well, site, channel) in multi_channel_images.keys():
            if p != plate:
                continue
            
            # Check if we already have this (plate, well, site) combination
            if not any(r.get("Metadata_Plate") == p and 
                      r.get("Metadata_Well") == well and 
                      r.get("Metadata_Site") == site for r in rows):
                
                row = {
                    "Metadata_Plate": p,
                    "Metadata_Well": well,
                    "Metadata_Site": site,
                    "Metadata_AbsPositionZ": 100.5 + site * 10,
                    "Metadata_ChannelID": f"CH{site}",
                    "Metadata_Col": int(well[1:]),
                    "Metadata_FieldID": f"Field{site}",
                    "Metadata_PlaneID": f"Plane{site}",
                    "Metadata_PositionX": 1000.0 + site * 100,
                    "Metadata_PositionY": 2000.0 + site * 100,
                    "Metadata_PositionZ": 50.0 + site * 5,
                    "Metadata_Row": well[0],
                    "Metadata_Reimaged": site % 2,
                }
                
                # Add channel paths
                for ch in channel_names:
                    img_path = multi_channel_images[(p, well, site, ch)]
                    row[f"FileName_{ch}"] = img_path.name
                    row[f"PathName_{ch}"] = str(img_path.parent)
                
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
        csv_paths.append(csv_path)
    
    return csv_paths


@pytest.fixture
def missing_channel_loaddata_csv(temp_images_root, multi_channel_images):
    """
    Create a LoadData CSV with missing channel columns (only subset of channels).
    """
    csv_path = temp_images_root / "missing_channel_loaddata.csv"
    
    # Only include DNA and RNA channels
    channels_subset = ["OrigDNA", "OrigRNA"]
    
    rows = []
    unique_combinations = set()
    for (plate, well, site, channel) in multi_channel_images.keys():
        if channel in channels_subset:
            unique_combinations.add((plate, well, site))
    
    for plate, well, site in sorted(unique_combinations):
        row = {
            "Metadata_Plate": plate,
            "Metadata_Well": well,
            "Metadata_Site": site,
        }
        
        for channel in channels_subset:
            img_path = multi_channel_images[(plate, well, site, channel)]
            row[f"FileName_{channel}"] = img_path.name
            row[f"PathName_{channel}"] = str(img_path.parent)
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)
    
    return csv_path
