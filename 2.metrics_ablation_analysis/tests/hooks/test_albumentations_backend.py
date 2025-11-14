"""
test_albumentations_backend.py

Pytest suite for AlbumentationsBackend class.
Tests the backend's ability to apply Albumentations transforms correctly
with reproducibility and variance control.
"""

import pytest
import numpy as np


class TestAlbumentationsBackend:
    """Test suite for AlbumentationsBackend."""

    def test_backend_initialization(self, backend_no_noise):
        """Test that backend initializes correctly."""
        assert backend_no_noise.name == "albumentations"
        assert backend_no_noise.transform is not None

    def test_apply_returns_same_shape(self, backend_no_noise, synthetic_image):
        """Test that apply returns image with same shape as input."""
        result = backend_no_noise.apply(synthetic_image, seed=42)
        assert result.shape == synthetic_image.shape

    def test_apply_returns_float32(self, backend_no_noise, synthetic_image):
        """Test that apply returns float32 dtype."""
        result = backend_no_noise.apply(synthetic_image, seed=42)
        assert result.dtype == np.float32

    def test_near_identical_with_zero_std(self, backend_no_noise, synthetic_image):
        """Test that zero std_range produces near-identical image."""
        result = backend_no_noise.apply(synthetic_image, seed=42)
        
        # With zero variance, output should be extremely close to input
        # Using tight tolerance for near-identity
        np.testing.assert_allclose(result, synthetic_image, rtol=1e-6, atol=1e-7)

    def test_identical_with_same_seed(self, backend_with_noise, synthetic_image):
        """Test that same seed produces identical results across multiple runs."""
        seed = 123
        
        result1 = backend_with_noise.apply(synthetic_image, seed=seed)
        result2 = backend_with_noise.apply(synthetic_image, seed=seed)
        result3 = backend_with_noise.apply(synthetic_image, seed=seed)
        
        # All results should be exactly identical
        np.testing.assert_array_equal(result1, result2)
        np.testing.assert_array_equal(result2, result3)

    def test_different_with_different_seeds(self, backend_with_noise, synthetic_image):
        """Test that different seeds produce different results."""
        result1 = backend_with_noise.apply(synthetic_image, seed=111)
        result2 = backend_with_noise.apply(synthetic_image, seed=222)
        result3 = backend_with_noise.apply(synthetic_image, seed=333)
        
        # Results should be different from each other
        # Check that not all values are equal
        assert not np.array_equal(result1, result2)
        assert not np.array_equal(result2, result3)
        assert not np.array_equal(result1, result3)
        
        # Also verify they're meaningfully different (not just floating point errors)
        diff_12 = np.abs(result1 - result2).mean()
        diff_23 = np.abs(result2 - result3).mean()
        diff_13 = np.abs(result1 - result3).mean()
        
        # Mean absolute difference should be non-trivial
        # With var=0.01, we expect differences on the order of std=0.1
        assert diff_12 > 1e-4
        assert diff_23 > 1e-4
        assert diff_13 > 1e-4
    
    def test_apply_without_seed(self, backend_with_noise, synthetic_image):
        """Test that apply works without explicit seed (uses random state)."""
        result = backend_with_noise.apply(synthetic_image)
        
        assert result.shape == synthetic_image.shape
        assert result.dtype == np.float32

    def test_noise_actually_applied(self, backend_with_noise, synthetic_image):
        """Test that noise transform actually modifies the image."""
        result = backend_with_noise.apply(synthetic_image, seed=42)
        
        # Output should be different from input when noise is applied
        assert not np.array_equal(result, synthetic_image)
        
        # Difference should be on the order of the noise std (~0.1)
        diff = np.abs(result - synthetic_image).mean()
        assert diff > 1e-4  # Should have non-trivial differences

    def test_output_in_valid_range(self, backend_with_noise, synthetic_image):
        """Test that output values remain in valid float range after transform."""
        result = backend_with_noise.apply(synthetic_image, seed=42)
        
        # GaussNoise can push values outside [0, 1], but they should be reasonable
        # Check that values aren't wildly out of range
        assert result.min() >= -0.5  # With std=0.1, unlikely to go much below 0
        assert result.max() <= 1.5   # With std=0.1, unlikely to go much above 1

    def test_chw_format_preserved(self, backend_no_noise, synthetic_image):
        """Test that CxHxW format is preserved through transformation."""
        # Input is 1x64x64
        assert synthetic_image.shape == (1, 64, 64)
        
        result = backend_no_noise.apply(synthetic_image, seed=42)
        
        # Output should also be 1x64x64 (CxHxW format)
        assert result.shape == (1, 64, 64)
        # First dimension should be channels (smallest for this test)
        assert result.shape[0] == 1
