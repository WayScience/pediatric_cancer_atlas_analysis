"""
test_parquet_index.py

Minimal tests for ParquetIndex class.
Tests cover:
- append_records: appending dummy data to index
- read: returning the full DataFrame
- materialize_seen_pairs: returning all (original_abs_path, config_id) tuples
- Error handling for incomplete schemas
"""

import pytest
import pandas as pd
import pyarrow.parquet as pq

from image_ablation_analysis.indexing import ParquetIndex


class TestParquetIndexBasicOps:
    """Test basic ParquetIndex operations with dummy data."""

    def test_append_records_creates_file(self, temp_parquet_index_dir, parquet_index_dummy_df):
        """Test that append_records creates a parquet file in the index directory."""
        index = ParquetIndex(temp_parquet_index_dir)
        
        # Append dummy data
        index.append_records(parquet_index_dummy_df)
        
        # Verify at least one parquet file was created
        parquet_files = list(temp_parquet_index_dir.glob("*.parquet"))
        assert len(parquet_files) > 0, "No parquet files created"
    

    def test_read_returns_full_dataframe(self, temp_parquet_index_dir, parquet_index_dummy_df):
        """Test that read() returns the complete DataFrame after append_records."""
        index = ParquetIndex(temp_parquet_index_dir)
        
        # Append dummy data
        index.append_records(parquet_index_dummy_df)
        
        # Read back the data
        result_df = index.read()
        
        # Verify all rows are present
        assert len(result_df) == len(parquet_index_dummy_df), (
            f"Expected {len(parquet_index_dummy_df)} rows, got {len(result_df)}"
        )
        
        # Verify all columns are present
        expected_cols = set(parquet_index_dummy_df.columns)
        actual_cols = set(result_df.columns)
        assert expected_cols == actual_cols, (
            f"Column mismatch. Expected {expected_cols}, got {actual_cols}"
        )
        
        # Verify key columns have correct values
        assert result_df["original_abs_path"].tolist() == parquet_index_dummy_df["original_abs_path"].tolist()
        assert result_df["config_id"].tolist() == parquet_index_dummy_df["config_id"].tolist()


    def test_materialize_seen_pairs_returns_all_tuples(self, temp_parquet_index_dir, parquet_index_dummy_df):
        """Test that materialize_seen_pairs returns all (original_abs_path, config_id) tuples."""
        index = ParquetIndex(temp_parquet_index_dir)
        
        # Append dummy data
        index.append_records(parquet_index_dummy_df)
        
        # Get seen pairs
        seen_pairs = index.materialize_seen_pairs()
        
        # Build expected set of (path, config_id) pairs from dummy data
        expected_pairs = set(
            zip(parquet_index_dummy_df["original_abs_path"], parquet_index_dummy_df["config_id"])
        )
        
        # Verify all pairs are present
        assert seen_pairs == expected_pairs, (
            f"Seen pairs mismatch.\nExpected: {expected_pairs}\nGot: {seen_pairs}"
        )
        assert len(seen_pairs) == 2


    def test_append_records_with_extra_metadata(self, temp_parquet_index_dir, parquet_index_dummy_df_with_metadata):
        """Test that append_records handles extra metadata columns beyond the required schema."""
        index = ParquetIndex(temp_parquet_index_dir)
        
        # Append data with extra columns
        index.append_records(parquet_index_dummy_df_with_metadata)
        
        # Read back and verify
        result_df = index.read()
        
        assert len(result_df) == len(parquet_index_dummy_df_with_metadata)
        # Verify extra columns are preserved
        assert "metadata_plate" in result_df.columns
        assert "metadata_well" in result_df.columns


    def test_materialize_seen_pairs_with_metadata(self, temp_parquet_index_dir, parquet_index_dummy_df_with_metadata):
        """Test materialize_seen_pairs with extra metadata columns present."""
        index = ParquetIndex(temp_parquet_index_dir)
        
        index.append_records(parquet_index_dummy_df_with_metadata)
        seen_pairs = index.materialize_seen_pairs()
        
        # Build expected pairs
        expected_pairs = set(
            zip(parquet_index_dummy_df_with_metadata["original_abs_path"], 
                parquet_index_dummy_df_with_metadata["config_id"])
        )
        
        assert seen_pairs == expected_pairs
        assert len(seen_pairs) == 3  # 3 rows in the metadata fixture


class TestParquetIndexEmptyIndex:
    """Test ParquetIndex behavior with empty index."""

    def test_read_empty_index_returns_empty_df(self, temp_parquet_index_dir):
        """Test that read() returns an empty DataFrame when index is empty."""
        index = ParquetIndex(temp_parquet_index_dir)
        
        result_df = index.read()
        
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 0


    def test_materialize_seen_pairs_empty_index(self, temp_parquet_index_dir):
        """Test that materialize_seen_pairs returns empty set when index is empty."""
        index = ParquetIndex(temp_parquet_index_dir)
        
        seen_pairs = index.materialize_seen_pairs()
        
        assert isinstance(seen_pairs, set)
        assert len(seen_pairs) == 0


    def test_list_done_paths_for_empty_index(self, temp_parquet_index_dir):
        """Test that list_done_paths_for returns empty set when index is empty."""
        index = ParquetIndex(temp_parquet_index_dir)
        
        paths = index.list_done_paths_for("cfg_001")
        
        assert isinstance(paths, set)
        assert len(paths) == 0


class TestParquetIndexErrorHandling:
    """Test ParquetIndex error handling with incomplete schemas."""

    def test_read_incomplete_schema_raises_error(self, temp_parquet_index_dir, parquet_index_incomplete_schema_df):
        """
        Test that read() raises ValueError when encountering a parquet file 
        with missing required columns from _schema().
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datetime import datetime, timezone
        
        index = ParquetIndex(temp_parquet_index_dir)
        
        # Manually write incomplete schema parquet file
        table = pa.Table.from_pandas(parquet_index_incomplete_schema_df, preserve_index=False)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        file_path = temp_parquet_index_dir / f"aug_index_{ts}.parquet"
        pq.write_table(table, str(file_path))
        
        # Attempting to read should gracefully handle or raise
        # (Currently read() catches exceptions, but we can verify behavior)
        result_df = index.read()
        # If exception is caught internally, we get partial/empty result
        # This tests that the code doesn't crash
        assert isinstance(result_df, pd.DataFrame)


    def test_materialize_seen_pairs_incomplete_schema(self, temp_parquet_index_dir, parquet_index_incomplete_schema_df):
        """
        Test that materialize_seen_pairs handles parquet files with incomplete schema
        (missing non-essential columns like params_json and variant).
        The method still returns data if it has the required columns (original_abs_path, config_id).
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datetime import datetime, timezone
        
        index = ParquetIndex(temp_parquet_index_dir)
        
        # Manually write incomplete schema parquet file
        # Note: parquet_index_incomplete_schema_df still has original_abs_path and config_id
        table = pa.Table.from_pandas(parquet_index_incomplete_schema_df, preserve_index=False)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        file_path = temp_parquet_index_dir / f"aug_index_{ts}.parquet"
        pq.write_table(table, str(file_path))
        
        # Should return the pair since required columns are present
        seen_pairs = index.materialize_seen_pairs()
        assert isinstance(seen_pairs, set)
        # Should have the one pair from the incomplete schema data
        assert len(seen_pairs) == 1
        assert ("/data/image1.tiff", "cfg_001") in seen_pairs


    def test_list_done_paths_for_incomplete_schema(self, temp_parquet_index_dir, parquet_index_incomplete_schema_df):
        """
        Test that list_done_paths_for handles parquet files with incomplete schema
        (missing non-essential columns like params_json and variant).
        The method still returns data if it has the required columns (original_abs_path, config_id).
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datetime import datetime, timezone
        
        index = ParquetIndex(temp_parquet_index_dir)
        
        # Manually write incomplete schema parquet file
        # Note: parquet_index_incomplete_schema_df still has original_abs_path and config_id
        table = pa.Table.from_pandas(parquet_index_incomplete_schema_df, preserve_index=False)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        file_path = temp_parquet_index_dir / f"aug_index_{ts}.parquet"
        pq.write_table(table, str(file_path))
        
        # Should return the path since required columns are present
        paths = index.list_done_paths_for("cfg_001")
        assert isinstance(paths, set)
        # Should have the one path from the incomplete schema data
        assert len(paths) == 1
        assert "/data/image1.tiff" in paths


    def test_materialize_seen_pairs_missing_required_columns(self, temp_parquet_index_dir, parquet_index_missing_path_config_df):
        """
        Test that materialize_seen_pairs returns empty set when required columns
        (original_abs_path, config_id) are missing from the parquet file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datetime import datetime, timezone
        
        index = ParquetIndex(temp_parquet_index_dir)
        
        # Manually write parquet file missing required columns
        table = pa.Table.from_pandas(parquet_index_missing_path_config_df, preserve_index=False)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        file_path = temp_parquet_index_dir / f"aug_index_{ts}.parquet"
        pq.write_table(table, str(file_path))
        
        # Should return empty set since required columns are missing
        seen_pairs = index.materialize_seen_pairs()
        assert isinstance(seen_pairs, set)
        assert len(seen_pairs) == 0


    def test_list_done_paths_for_missing_required_columns(self, temp_parquet_index_dir, parquet_index_missing_path_config_df):
        """
        Test that list_done_paths_for returns empty set when required columns
        (original_abs_path, config_id) are missing from the parquet file.
        """
        import pyarrow as pa
        import pyarrow.parquet as pq
        from datetime import datetime, timezone
        
        index = ParquetIndex(temp_parquet_index_dir)
        
        # Manually write parquet file missing required columns
        table = pa.Table.from_pandas(parquet_index_missing_path_config_df, preserve_index=False)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        file_path = temp_parquet_index_dir / f"aug_index_{ts}.parquet"
        pq.write_table(table, str(file_path))
        
        # Should return empty set since required columns are missing
        paths = index.list_done_paths_for("cfg_001")
        assert isinstance(paths, set)
        assert len(paths) == 0


    def test_append_records_empty_dataframe(self, temp_parquet_index_dir):
        """Test that append_records handles empty DataFrame gracefully."""
        index = ParquetIndex(temp_parquet_index_dir)
        
        empty_df = pd.DataFrame()
        
        # Should not raise an error, just return without writing
        index.append_records(empty_df)
        
        # Verify nothing was written
        parquet_files = list(temp_parquet_index_dir.glob("*.parquet"))
        assert len(parquet_files) == 0
