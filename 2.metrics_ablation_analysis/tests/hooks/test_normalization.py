from pathlib import Path

import pytest
import numpy as np


# Tests for unsigned integer dtypes
class TestUnsignedIntegerNormalization:
    """Test normalization for uint8 and uint16 images."""

    def test_uint8_auto_infer(self, normalizer, uint8_image, dummy_path):
        """Test uint8 normalization with auto-inferred bit depth."""
        norm, info = normalizer.normalize(uint8_image, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert info["strategy"] == "bitdepth"
        assert info["bit_depth"] == 8
        assert info["scale"] == 255.0
        assert info["orig_dtype"] == "uint8"
        
        # Check normalization correctness
        assert np.allclose(norm, uint8_image / 255.0)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_uint8_explicit_bitdepth(self, normalizer, uint8_image, dummy_path):
        """Test uint8 normalization with explicit 8-bit depth."""
        norm, info = normalizer.normalize(uint8_image, bit_depth=8, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert info["bit_depth"] == 8
        assert info["scale"] == 255.0
        assert np.allclose(norm, uint8_image / 255.0)

    def test_uint16_auto_infer(self, normalizer, uint16_image_16bit, dummy_path):
        """Test uint16 normalization with auto-inferred 16-bit depth."""
        norm, info = normalizer.normalize(uint16_image_16bit, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert info["strategy"] == "bitdepth"
        assert info["bit_depth"] == 16
        assert info["scale"] == 65535.0
        assert info["orig_dtype"] == "uint16"
        
        # Check normalization correctness
        assert np.allclose(norm, uint16_image_16bit / 65535.0)
        assert norm.min() >= 0.0
        assert norm.max() <= 1.0

    def test_uint16_12bit_explicit(self, normalizer, uint16_image_12bit, dummy_path):
        """Test uint16 array with 12-bit data (explicit bit_depth)."""
        norm, info = normalizer.normalize(uint16_image_12bit, bit_depth=12, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert info["bit_depth"] == 12
        assert info["scale"] == 4095.0
        
        # Check normalization correctness
        assert np.allclose(norm, uint16_image_12bit / 4095.0)
        assert norm.max() <= 1.0

    def test_uint16_8bit_explicit(self, normalizer, uint16_image_8bit, dummy_path):
        """Test uint16 array with 8-bit data (explicit bit_depth)."""
        norm, info = normalizer.normalize(uint16_image_8bit, bit_depth=8, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert info["bit_depth"] == 8
        assert info["scale"] == 255.0
        
        # Check normalization correctness
        assert np.allclose(norm, uint16_image_8bit / 255.0)


# Tests for floating point dtypes
class TestFloatNormalization:
    """Test normalization for float16 and float32 images."""

    def test_float32_already_normalized(self, normalizer, float32_normalized, dummy_path):
        """Test float32 image already in [0, 1] range (passthrough)."""
        norm, info = normalizer.normalize(float32_normalized, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert info["strategy"] == "bitdepth_passthrough_float01"
        assert info["bit_depth"] is None
        assert info["scale"] == 1.0
        assert info["orig_dtype"] == "float32"
        
        # Should be unchanged
        assert np.allclose(norm, float32_normalized)

    def test_float16_already_normalized(self, normalizer, float16_normalized, dummy_path):
        """Test float16 image already in [0, 1] range (passthrough)."""
        norm, info = normalizer.normalize(float16_normalized, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert info["strategy"] == "bitdepth_passthrough_float01"
        assert info["orig_dtype"] == "float16"
        
        # Should be converted to float32
        assert np.allclose(norm, float16_normalized.astype(np.float32))

    def test_float32_unnormalized_with_bitdepth(self, normalizer, float32_unnormalized, dummy_path):
        """Test float32 outside [0,1] with explicit bit_depth."""
        norm, info = normalizer.normalize(float32_unnormalized, bit_depth=8, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert info["strategy"] == "bitdepth_float_as_int"
        assert info["bit_depth"] == 8
        assert info["scale"] == 255.0
        
        # Should normalize as if it were 8-bit
        assert np.allclose(norm, float32_unnormalized / 255.0)

    def test_float32_unnormalized_no_bitdepth_raises(self, normalizer, float32_unnormalized, dummy_path):
        """Test float32 outside [0,1] without bit_depth raises error."""
        with pytest.raises(ValueError, match="Float input without clear normalization semantics"):
            normalizer.normalize(float32_unnormalized, path=dummy_path)


# Tests for edge cases and special dtypes
class TestEdgeCases:
    """Test edge cases and special dtypes."""

    def test_bool_dtype(self, normalizer, bool_image, dummy_path):
        """Test boolean image normalization."""
        norm, info = normalizer.normalize(bool_image, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert info["strategy"] == "bitdepth"
        assert info["bit_depth"] == 1
        assert info["scale"] == 1.0
        assert info["orig_dtype"] == "bool"
        
        # True -> 1.0, False -> 0.0
        expected = bool_image.astype(np.float32)
        assert np.array_equal(norm, expected)

    def test_uint8_all_zeros(self, normalizer, dummy_path):
        """Test normalization of all-zero image."""
        img = np.zeros((3, 3), dtype=np.uint8)
        norm, info = normalizer.normalize(img, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert np.all(norm == 0.0)

    def test_uint8_all_max(self, normalizer, dummy_path):
        """Test normalization of all-max-value image."""
        img = np.full((3, 3), 255, dtype=np.uint8)
        norm, info = normalizer.normalize(img, path=dummy_path)
        
        assert norm.dtype == np.float32
        assert np.allclose(norm, 1.0)

    def test_invalid_bitdepth_zero(self, normalizer, uint8_image, dummy_path):
        """Test that bit_depth=0 raises ValueError."""
        with pytest.raises(ValueError, match="Invalid bit_depth"):
            normalizer.normalize(uint8_image, bit_depth=0, path=dummy_path)

    def test_invalid_bitdepth_negative(self, normalizer, uint8_image, dummy_path):
        """Test that negative bit_depth raises ValueError."""
        with pytest.raises(ValueError, match="Invalid bit_depth"):
            normalizer.normalize(uint8_image, bit_depth=-1, path=dummy_path)

    def test_unsupported_dtype_signed_int(self, normalizer, dummy_path):
        """Test that signed integer dtype raises TypeError."""
        img = np.array([[0, 127, -128], [64, -64, 0]], dtype=np.int8)
        with pytest.raises(TypeError, match="Unsupported image dtype"):
            normalizer.normalize(img, path=dummy_path)


# Tests for normalization info dictionary
class TestNormalizationInfo:
    """Test that normalization info dictionary contains expected keys."""

    def test_info_dict_keys_uint(self, normalizer, uint8_image, dummy_path):
        """Test info dict has all required keys for uint images."""
        _, info = normalizer.normalize(uint8_image, path=dummy_path)
        
        assert "strategy" in info
        assert "orig_dtype" in info
        assert "bit_depth" in info
        assert "scale" in info

    def test_info_dict_keys_float(self, normalizer, float32_normalized, dummy_path):
        """Test info dict has all required keys for float images."""
        _, info = normalizer.normalize(float32_normalized, path=dummy_path)
        
        assert "strategy" in info
        assert "orig_dtype" in info
        assert "bit_depth" in info
        assert "scale" in info


# Tests for specific bit depth scenarios
class TestBitDepthScenarios:
    """Test various bit depth configuration scenarios."""

    def test_bitdepth_greater_than_dtype(self, normalizer, uint8_image, dummy_path):
        """Test warning when bit_depth > dtype bits."""
        # Should work but produce a warning
        with pytest.warns(UserWarning, match="bit_depth=16 > dtype bits=8"):
            norm, info = normalizer.normalize(uint8_image, bit_depth=16, path=dummy_path)
        
        # Should still normalize using the provided bit_depth
        assert info["bit_depth"] == 16
        assert info["scale"] == 65535.0

    def test_bitdepth_less_than_dtype(self, normalizer, uint16_image_16bit, dummy_path):
        """Test warning when bit_depth < dtype bits (12-bit in 16-bit)."""
        with pytest.warns(UserWarning, match="bit_depth=12 < dtype bits=16"):
            norm, info = normalizer.normalize(uint16_image_16bit, bit_depth=12, path=dummy_path)
        
        # Should normalize using 12-bit scale
        assert info["bit_depth"] == 12
        assert info["scale"] == 4095.0

    def test_auto_infer_produces_warning(self, normalizer, uint16_image_16bit, dummy_path):
        """Test that auto-inference produces a warning."""
        with pytest.warns(UserWarning, match="Bit depth not specified, auto-inferred"):
            norm, info = normalizer.normalize(uint16_image_16bit, path=dummy_path)
        
        assert info["bit_depth"] == 16
