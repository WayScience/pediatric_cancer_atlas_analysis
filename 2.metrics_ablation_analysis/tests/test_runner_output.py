"""
End-to-end tests for AblationRunner to ensure properly typed image outputs.
Since the focus is on output, uses a no-op augmentation backend.
"""

import json
from typing import Any, Dict, Optional, Tuple

import pytest
import numpy as np
import tifffile as tiff

from image_ablation_analysis.ablation_runner import AblationRunner
from image_ablation_analysis.hooks.generic_hook import GenericTransformHook


# No-op Transform Backend for testing runner end to end
class NoOpBackend:
    """
    Minimal TransformBackend implementation that passes images through unchanged.
    Used for isolated testing of the runner infrastructure.
    """
    
    name = "noop"
    
    def apply(self, img: np.ndarray, *, seed: Optional[int] = None) -> np.ndarray:
        """Return image unchanged."""
        return img.copy()
    
    def describe(self) -> Tuple[str, Dict[str, Any]]:
        """Describe the transform."""
        return "NoOp", {"description": "identity transform"}


@pytest.fixture
def noop_backend():
    """No-op backend fixture."""
    return NoOpBackend()


class TestAblationRunnerE2E:
    """End-to-end tests for AblationRunner with synthetic data."""

    def test_runner_with_return_original_dtype_false(
        self,
        temp_images_root,
        temp_ablation_root,
        temp_loaddata_csv,
        synthetic_uint16_image,
        noop_backend,
    ):
        """
        Test full pipeline with return_original_dtype=False.
        Should output float32 normalized images.
        """
        # Create hook with return_original_dtype=False
        hook = GenericTransformHook(
            backend=noop_backend,
            variant_prefix="test",
            return_original_dtype=False,
        )
        
        # Create runner
        runner = AblationRunner(
            images_root=temp_images_root,
            ablation_root=temp_ablation_root,
            loaddata_csvs=[temp_loaddata_csv],
            skip_if_indexed=False,
            dry_run=False,
        )
        
        # Run ablation
        runner.run(augment_hook=hook)
        
        # Verify output file exists
        ablated_files = list(temp_ablation_root.rglob("*.tiff"))
        assert len(ablated_files) == 1, f"Expected 1 ablated file, found {len(ablated_files)}"
        
        output_path = ablated_files[0]
        
        # Read output and verify dtype
        output_img = tiff.imread(str(output_path))
        assert output_img.dtype == np.float32, f"Expected float32, got {output_img.dtype}"
        
        # Verify shape is preserved
        assert output_img.shape == synthetic_uint16_image.shape
        
        # Verify values are still float and [0, 1] normalized,
        # as this is the default behavior of `return_original_dtype=False`
        assert output_img.min() >= 0.0
        assert output_img.max() <= 1.0
        
        # Verify normalization correctness (uint16 -> [0,1])
        expected = synthetic_uint16_image.astype(np.float32) / 65535.0
        np.testing.assert_allclose(output_img, expected, rtol=1e-6)
        
        # Verify index was created and populated
        index_path = temp_ablation_root / "ablated_index"
        assert index_path.exists(), "Index file should exist"
        
        # Verify sidecar JSON exists
        sidecar_path = output_path.with_suffix(".json")
        assert sidecar_path.exists(), "Sidecar JSON should exist"
        
        with open(sidecar_path) as f:
            sidecar = json.load(f)
            assert "variant" in sidecar
            assert "params" in sidecar
            assert sidecar["params"]["backend"] == "noop"

    def test_runner_with_return_original_dtype_true(
        self,
        temp_images_root,
        temp_ablation_root,
        temp_loaddata_csv,
        synthetic_uint16_image,
        noop_backend,
    ):
        """
        Test full pipeline with return_original_dtype=True.
        Should output uint16 images in original dtype.
        """
        # Create hook with return_original_dtype=True
        hook = GenericTransformHook(
            backend=noop_backend,
            variant_prefix="test",
            return_original_dtype=True,
        )
        
        # Create runner
        runner = AblationRunner(
            images_root=temp_images_root,
            ablation_root=temp_ablation_root,
            loaddata_csvs=[temp_loaddata_csv],
            skip_if_indexed=False,
            dry_run=False,
        )
        
        # Run ablation
        runner.run(augment_hook=hook)
        
        # Verify output file exists
        ablated_files = list(temp_ablation_root.rglob("*.tiff"))
        assert len(ablated_files) == 1, f"Expected 1 ablated file, found {len(ablated_files)}"
        
        output_path = ablated_files[0]
        
        # Read output and verify dtype
        output_img = tiff.imread(str(output_path))
        assert output_img.dtype == np.uint16, f"Expected uint16, got {output_img.dtype}"
        
        # Verify shape is preserved
        assert output_img.shape == synthetic_uint16_image.shape
        
        # Verify values match original (no-op transform)
        np.testing.assert_array_equal(output_img, synthetic_uint16_image)
        
        # Verify index was created
        index_path = temp_ablation_root / "ablated_index"
        assert index_path.exists(), "Index file should exist"

    def test_runner_index_contents(
        self,
        temp_images_root,
        temp_ablation_root,
        temp_loaddata_csv,
        noop_backend,
    ):
        """Test that the parquet index contains expected fields."""
        hook = GenericTransformHook(
            backend=noop_backend,
            variant_prefix="test",
            return_original_dtype=False,
        )
        
        runner = AblationRunner(
            images_root=temp_images_root,
            ablation_root=temp_ablation_root,
            loaddata_csvs=[temp_loaddata_csv],
            skip_if_indexed=False,
            dry_run=False,
        )
        
        runner.run(augment_hook=hook)
        
        # Read index
        index = runner.index
        df = index.read()
        
        assert len(df) == 1, f"Expected 1 record in index, found {len(df)}"
        
        # Verify required columns exist
        required_cols = [
            "created_at",
            "run_id",
            "original_abs_path",
            "original_rel_path",
            "aug_abs_path",
            "aug_rel_path",
            "variant",
            "config_id",
            "params_json",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing required column: {col}"
        
        # Verify metadata columns from loaddata
        assert "Metadata_Plate" in df.columns
        assert "Metadata_Well" in df.columns
        assert "Metadata_Site" in df.columns
        
        # Verify values
        row = df.iloc[0]
        assert row["Metadata_Plate"] == "TestPlate001"
        assert row["Metadata_Well"] == "A01"
        assert row["Metadata_Site"] == 1
        
        # Verify variant name format
        assert row["variant"].startswith("test_noop-NoOp_")
        
        # Verify config_id
        assert "noop:NoOp:" in row["config_id"]
        
        # Verify params_json is valid JSON
        params = json.loads(row["params_json"])
        assert params["backend"] == "noop"
        assert params["transform_name"] == "NoOp"
