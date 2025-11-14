"""
Tests for the LoadDataIndex class in image_ablation_analysis.indexing module.
"""

import pytest
from pathlib import Path
import pandas as pd

from image_ablation_analysis.indexing import LoadDataIndex


class TestLoadDataIndexInit:
    """Test LoadDataIndex initialization and CSV parsing."""
    
    def test_init_single_csv(self, comprehensive_loaddata_csv):
        """Test initialization with a single CSV file."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        assert index is not None
        assert hasattr(index, 'df')
        assert hasattr(index, 'long_df')
        assert hasattr(index, 'path_to_meta')
        assert len(index.df) > 0
    
    def test_init_multiple_csvs(self, multi_csv_loaddata):
        """Test initialization with multiple CSV files."""
        index = LoadDataIndex(multi_csv_loaddata)
        
        assert index is not None
        assert len(index.df) > 0
        # Should have combined data from both CSVs
        assert len(index.df) >= len(multi_csv_loaddata)
    
    def test_init_missing_csv_raises_error(self, temp_images_root):
        """Test that initialization raises FileNotFoundError for missing CSV."""
        missing_csv = temp_images_root / "nonexistent.csv"
        
        with pytest.raises(FileNotFoundError) as exc_info:
            LoadDataIndex([missing_csv])
        
        assert "not found" in str(exc_info.value).lower()
    
    def test_init_preserves_loaddata_source(self, multi_csv_loaddata):
        """Test that __loaddata_source column is added and tracked internally."""
        index = LoadDataIndex(multi_csv_loaddata)
        
        # Internal df should have __loaddata_source
        assert "__loaddata_source" in index.df.columns
        # But it shouldn't be in the public columns property
        assert "__loaddata_source" not in index.columns


class TestLoadDataIndexChannelDetection:
    """Test channel stem detection from FileName_* and PathName_* columns."""
    
    def test_detects_all_channels(self, comprehensive_loaddata_csv, channel_names):
        """Test that all channel stems are correctly detected."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        assert set(index.channel_stems) == set(channel_names)
    
    def test_only_paired_channels_detected(self, temp_images_root):
        """Test that only channels with both FileName_* and PathName_* are detected."""
        # Create CSV with mismatched columns
        csv_path = temp_images_root / "mismatched.csv"
        df = pd.DataFrame({
            "FileName_DNA": ["test.tiff"],
            "PathName_DNA": [str(temp_images_root)],
            "FileName_RNA": ["test2.tiff"],  # Missing PathName_RNA
            "PathName_Mito": [str(temp_images_root)],  # Missing FileName_Mito
            "Metadata_Plate": ["TestPlate"],
        })
        df.to_csv(csv_path, index=False)
        
        index = LoadDataIndex([csv_path])
        
        # Only DNA should be detected (has both FileName and PathName)
        assert "DNA" in index.channel_stems
        assert "RNA" not in index.channel_stems
        assert "Mito" not in index.channel_stems
    
    def test_channel_stems_sorted(self, comprehensive_loaddata_csv):
        """Test that channel stems are sorted alphabetically."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        assert index.channel_stems == sorted(index.channel_stems)


class TestLoadDataIndexAbsolutePathConstruction:
    """Test absolute path construction for each channel."""
    
    def test_abspath_columns_created(self, comprehensive_loaddata_csv, channel_names):
        """Test that AbsPath_* columns are created for each channel."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        for channel in channel_names:
            abs_col = f"AbsPath_{channel}"
            assert abs_col in index.df.columns
    
    def test_abspath_format(self, comprehensive_loaddata_csv):
        """Test that absolute paths are correctly formatted (forward slashes)."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        for channel in index.channel_stems:
            abs_col = f"AbsPath_{channel}"
            for path in index.df[abs_col]:
                # Should use forward slashes, not backslashes
                assert "\\" not in path
                assert "/" in path
    
    def test_abspath_combines_path_and_filename(self, comprehensive_loaddata_csv):
        """Test that absolute paths correctly combine PathName and FileName."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        for channel in index.channel_stems:
            file_col = f"FileName_{channel}"
            path_col = f"PathName_{channel}"
            abs_col = f"AbsPath_{channel}"
            
            for _, row in index.df.iterrows():
                expected_end = row[file_col]
                actual_path = row[abs_col]
                
                # Path should end with filename
                assert actual_path.endswith(expected_end)


class TestLoadDataIndexLongFormat:
    """Test conversion to long format (one row per image-channel pair)."""
    
    def test_long_df_created(self, comprehensive_loaddata_csv):
        """Test that long_df is created with expected columns."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        assert "__channel" in index.long_df.columns
        assert "__abs_path" in index.long_df.columns
        assert "__meta_row" in index.long_df.columns
    
    def test_long_df_row_count(self, comprehensive_loaddata_csv, channel_names):
        """Test that long_df has correct number of rows (sites × channels)."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        num_sites = len(index.df)
        num_channels = len(channel_names)
        expected_rows = num_sites * num_channels
        
        assert len(index.long_df) == expected_rows
    
    def test_long_df_no_null_paths(self, comprehensive_loaddata_csv):
        """Test that long_df has no null absolute paths."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        assert not index.long_df["__abs_path"].isna().any()
    
    def test_long_df_meta_row_preserved(self, comprehensive_loaddata_csv):
        """Test that metadata row is preserved in long format."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        for _, row in index.long_df.iterrows():
            meta_row = row["__meta_row"]
            assert isinstance(meta_row, dict)
            assert "Metadata_Plate" in meta_row
            assert "Metadata_Well" in meta_row


class TestLoadDataIndexPathToMeta:
    """Test the path_to_meta lookup dictionary."""
    
    def test_path_to_meta_populated(self, comprehensive_loaddata_csv):
        """Test that path_to_meta dictionary is populated."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        assert len(index.path_to_meta) > 0
    
    def test_path_to_meta_uses_resolved_paths(self, comprehensive_loaddata_csv, multi_channel_images):
        """Test that path_to_meta uses resolved absolute paths as keys."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        # All keys should be resolved absolute paths (no relative components)
        for key in index.path_to_meta.keys():
            path = Path(key)
            assert path.is_absolute()
            # Resolved paths shouldn't have .. or .
            assert ".." not in str(path)
    
    def test_path_to_meta_maps_to_correct_metadata(self, comprehensive_loaddata_csv, multi_channel_images):
        """Test that path_to_meta correctly maps paths to their metadata."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        # Pick a known image path
        for (plate, well, site, channel), img_path in list(multi_channel_images.items())[:3]:
            resolved_path = str(img_path.resolve())
            
            if resolved_path in index.path_to_meta:
                meta = index.path_to_meta[resolved_path]
                assert meta["Metadata_Plate"] == plate
                assert meta["Metadata_Well"] == well
                assert meta["Metadata_Site"] == site


class TestLoadDataIndexMetadataFor:
    """Test the metadata_for() method."""
    
    def test_metadata_for_existing_path(self, comprehensive_loaddata_csv, multi_channel_images):
        """Test metadata retrieval for an existing image path."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        # Get a known image path
        (plate, well, site, channel), img_path = list(multi_channel_images.items())[0]
        
        meta = index.metadata_for(img_path)
        
        assert isinstance(meta, dict)
        assert meta["Metadata_Plate"] == plate
        assert meta["Metadata_Well"] == well
        assert meta["Metadata_Site"] == site
    
    def test_metadata_for_nonexistent_path(self, comprehensive_loaddata_csv, temp_images_root):
        """Test metadata retrieval for a path not in the index."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        fake_path = temp_images_root / "nonexistent" / "fake.tiff"
        meta = index.metadata_for(fake_path)
        
        # Should return empty dict for missing paths
        assert meta == {}
    
    def test_metadata_for_returns_all_columns(self, comprehensive_loaddata_csv, multi_channel_images):
        """Test that metadata_for returns all original CSV columns."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        # Get a known image path
        img_path = list(multi_channel_images.values())[0]
        meta = index.metadata_for(img_path)
        
        # Should have all metadata columns
        assert "Metadata_Plate" in meta
        assert "Metadata_Well" in meta
        assert "Metadata_Site" in meta
        assert "Metadata_AbsPositionZ" in meta
        assert "Metadata_Row" in meta
        
        # Should also have file/path columns
        for channel in index.channel_stems:
            assert f"FileName_{channel}" in meta
            assert f"PathName_{channel}" in meta


class TestLoadDataIndexIterAllAbsPaths:
    """Test the iter_all_abs_paths() method."""
    
    def test_iter_all_abs_paths_returns_paths(self, comprehensive_loaddata_csv):
        """Test that iter_all_abs_paths yields Path objects."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        paths = list(index.iter_all_abs_paths())
        
        assert len(paths) > 0
        assert all(isinstance(p, Path) for p in paths)
    
    def test_iter_all_abs_paths_yields_unique_paths(self, comprehensive_loaddata_csv):
        """Test that iter_all_abs_paths yields unique paths."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        paths = list(index.iter_all_abs_paths())
        unique_paths = set(str(p) for p in paths)
        
        assert len(paths) == len(unique_paths)
    
    def test_iter_all_abs_paths_yields_absolute_paths(self, comprehensive_loaddata_csv):
        """Test that all yielded paths are absolute."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        for path in index.iter_all_abs_paths():
            assert path.is_absolute()
    
    def test_iter_all_abs_paths_yields_existing_files(self, comprehensive_loaddata_csv, multi_channel_images):
        """Test that all yielded paths point to existing files."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        for path in index.iter_all_abs_paths():
            assert path.exists(), f"Path does not exist: {path}"
            assert path.is_file(), f"Path is not a file: {path}"
    
    def test_iter_all_abs_paths_count(self, comprehensive_loaddata_csv, multi_channel_images, channel_names):
        """Test that iter_all_abs_paths yields expected number of paths."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        paths = list(index.iter_all_abs_paths())
        
        # Should have one path per unique image file
        # Number of unique (plate, well, site, channel) combinations
        expected_count = len(multi_channel_images)
        
        assert len(paths) == expected_count


class TestLoadDataIndexColumns:
    """Test the columns property."""
    
    def test_columns_property_returns_list(self, comprehensive_loaddata_csv):
        """Test that columns property returns a list."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        assert isinstance(index.columns, list)
    
    def test_columns_excludes_internal_columns(self, comprehensive_loaddata_csv):
        """Test that columns property excludes internal __loaddata_source."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        assert "__loaddata_source" not in index.columns
    
    def test_columns_includes_all_original_columns(self, comprehensive_loaddata_csv):
        """Test that columns includes all original CSV columns."""
        # Read CSV directly to get expected columns
        df_direct = pd.read_csv(comprehensive_loaddata_csv)
        expected_columns = set(df_direct.columns)
        
        index = LoadDataIndex([comprehensive_loaddata_csv])
        actual_columns = set(index.columns)
        
        assert actual_columns == expected_columns
    
    def test_columns_includes_metadata_columns(self, comprehensive_loaddata_csv):
        """Test that columns includes all Metadata_* columns."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        metadata_cols = [c for c in index.columns if c.startswith("Metadata_")]
        
        assert "Metadata_Plate" in metadata_cols
        assert "Metadata_Well" in metadata_cols
        assert "Metadata_Site" in metadata_cols
        assert "Metadata_AbsPositionZ" in metadata_cols
        assert "Metadata_Row" in metadata_cols
        assert "Metadata_Reimaged" in metadata_cols


class TestLoadDataIndexLen:
    """Test the __len__ method."""
    
    def test_len_returns_long_df_length(self, comprehensive_loaddata_csv):
        """Test that len(index) returns the length of long_df."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        assert len(index) == len(index.long_df)
    
    def test_len_equals_sites_times_channels(self, comprehensive_loaddata_csv, channel_names):
        """Test that len equals number of sites × number of channels."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        num_sites = len(index.df)
        num_channels = len(channel_names)
        
        assert len(index) == num_sites * num_channels


class TestLoadDataIndexEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_missing_channels_in_csv(self, missing_channel_loaddata_csv):
        """Test handling of CSV with only subset of channels."""
        index = LoadDataIndex([missing_channel_loaddata_csv])
        
        # Should only detect DNA and RNA
        assert set(index.channel_stems) == {"OrigDNA", "OrigRNA"}
        assert len(index) > 0
    
    def test_empty_csv_handling(self, temp_images_root):
        """Test handling of CSV with only headers (no data rows)."""
        csv_path = temp_images_root / "empty.csv"
        df = pd.DataFrame(columns=["FileName_DNA", "PathName_DNA", "Metadata_Plate"])
        df.to_csv(csv_path, index=False)
        
        index = LoadDataIndex([csv_path])
        
        # Should initialize but have no data
        assert len(index.df) == 0
        assert len(index.long_df) == 0
        assert len(index.path_to_meta) == 0
        assert len(list(index.iter_all_abs_paths())) == 0
    
    def test_multiple_csv_concatenation_order(self, multi_csv_loaddata):
        """Test that multiple CSVs are concatenated correctly."""
        index = LoadDataIndex(multi_csv_loaddata)
        
        # Should have __loaddata_source for tracking which CSV each row came from
        sources = index.df["__loaddata_source"].unique()
        
        # Should have entries from both CSV files
        assert len(sources) == len(multi_csv_loaddata)
        
        for csv_path in multi_csv_loaddata:
            assert str(csv_path) in sources
    
    def test_path_normalization_consistency(self, comprehensive_loaddata_csv, multi_channel_images):
        """Test that path normalization is consistent across methods."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        # Pick a known image
        img_path = list(multi_channel_images.values())[0]
        
        # metadata_for should work with both relative and absolute paths
        meta1 = index.metadata_for(img_path)
        meta2 = index.metadata_for(img_path.resolve())
        
        # Both should return the same metadata
        assert meta1 == meta2
    
    def test_special_characters_in_paths(self, temp_images_root, synthetic_uint16_image):
        """Test handling of paths with special characters."""
        import tifffile as tiff
        
        # Create directory and file with spaces and special chars
        special_dir = temp_images_root / "test folder (special)"
        special_dir.mkdir()
        
        img_path = special_dir / "test image [001].tiff"
        tiff.imwrite(str(img_path), synthetic_uint16_image)
        
        csv_path = temp_images_root / "special_chars.csv"
        df = pd.DataFrame({
            "FileName_DNA": [img_path.name],
            "PathName_DNA": [str(img_path.parent)],
            "Metadata_Plate": ["TestPlate"],
        })
        df.to_csv(csv_path, index=False)
        
        index = LoadDataIndex([csv_path])
        
        # Should handle special characters correctly
        assert len(list(index.iter_all_abs_paths())) == 1
        meta = index.metadata_for(img_path)
        assert meta["Metadata_Plate"] == "TestPlate"


class TestLoadDataIndexIntegration:
    """Integration tests combining multiple methods."""
    
    def test_full_workflow_single_csv(self, comprehensive_loaddata_csv, multi_channel_images):
        """Test complete workflow: init, iterate paths, get metadata."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        # Iterate through all paths
        paths_processed = 0
        for img_path in index.iter_all_abs_paths():
            # Each path should exist
            assert img_path.exists()
            
            # Should be able to get metadata
            meta = index.metadata_for(img_path)
            assert len(meta) > 0
            
            # Metadata should have expected fields
            assert "Metadata_Plate" in meta
            assert "Metadata_Well" in meta
            
            paths_processed += 1
        
        # Should have processed all images
        assert paths_processed == len(multi_channel_images)
    
    def test_full_workflow_multiple_csvs(self, multi_csv_loaddata, multi_channel_images):
        """Test complete workflow with multiple CSV files."""
        index = LoadDataIndex(multi_csv_loaddata)
        
        # Should combine data from all CSVs
        all_plates = set()
        for img_path in index.iter_all_abs_paths():
            meta = index.metadata_for(img_path)
            all_plates.add(meta["Metadata_Plate"])
        
        # Should have data from both plates
        assert "BR00143976" in all_plates
        assert "BR00143977" in all_plates
    
    def test_metadata_consistency_across_channels(self, comprehensive_loaddata_csv, multi_channel_images, channel_names):
        """Test that metadata is consistent for all channels of the same site."""
        index = LoadDataIndex([comprehensive_loaddata_csv])
        
        # Group images by (plate, well, site)
        by_site = {}
        for (plate, well, site, channel), img_path in multi_channel_images.items():
            key = (plate, well, site)
            if key not in by_site:
                by_site[key] = []
            by_site[key].append(img_path)
        
        # For each site, all channels should have same metadata (except file/path cols)
        for site_key, img_paths in by_site.items():
            plate, well, site = site_key
            
            metadata_list = [index.metadata_for(p) for p in img_paths]
            
            # All should have same plate, well, site
            for meta in metadata_list:
                assert meta["Metadata_Plate"] == plate
                assert meta["Metadata_Well"] == well
                assert meta["Metadata_Site"] == site
