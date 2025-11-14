"""
Minimal end-to-end tests for GenericTransformHook with Albumentations backend
and BitDepthNormalizer.
"""

import numpy as np
import tifffile as tiff
import albumentations as A

from image_ablation_analysis.hooks.generic_hook import GenericTransformHook
from image_ablation_analysis.hooks.albumentations import AlbumentationsBackend
from image_ablation_analysis.hooks.normalization import BitDepthNormalizer


def test_generic_hook_end_to_end_with_noise(temp_uint16_image):
    """
    Test GenericTransformHook end-to-end with Albumentations GaussNoise.
    Ensures transformed image is close to but not identical to the original.
    """
    # Create Albumentations backend with small Gaussian noise
    transform = A.GaussNoise(
        std_range=(0.01, 0.01),  # small standard deviation
        mean_range=(0.0, 0.0),   # zero mean
        noise_scale_factor=1.0,
        p=1.0  # always apply
    )
    backend = AlbumentationsBackend(transform=transform)
    
    # Create normalizer for 16-bit images
    normalizer = BitDepthNormalizer(bit_depth=16)
    
    # Create the hook
    hook = GenericTransformHook(
        backend=backend,
        variant_prefix="test",
        normalizer=normalizer,
        return_original_dtype=False,  # Keep as float32
    )
    
    raw = tiff.imread(str(temp_uint16_image))

    # Apply the hook
    variants = list(hook(temp_uint16_image))
    
    # Should yield exactly one variant
    assert len(variants) == 1
    variant = variants[0]
    
    # Check variant properties
    assert variant.variant.startswith("test_albumentations-GaussNoise_")
    assert variant.image.dtype == np.float32
    assert variant.image.shape == raw.shape
    
    # normalize the original image for comparison    
    normalized_original, _ = normalizer.normalize(raw, path=temp_uint16_image)
    
    # Transformed image should be close to but not identical to normalized original
    # Use allclose with reasonable tolerance given small noise
    assert np.allclose(variant.image, normalized_original, atol=0.05)
    
    # But should NOT be identical (noise was applied)
    assert not np.array_equal(variant.image, normalized_original)
    
    # Check that values are still in reasonable range [0, 1]
    assert variant.image.min() >= -0.1  # allow small noise outside bounds
    assert variant.image.max() <= 1.1


def test_generic_hook_end_to_end_no_noise(temp_uint16_image):
    """
    Test GenericTransformHook with zero noise (should produce nearly identical output).
    """
    # Create Albumentations backend with zero Gaussian noise
    transform = A.GaussNoise(
        std_range=(0.0, 0.0),    # zero standard deviation
        mean_range=(0.0, 0.0),   # zero mean
        noise_scale_factor=1.0,
        p=1.0
    )
    backend = AlbumentationsBackend(transform=transform)
    
    # Create normalizer for 16-bit images
    normalizer = BitDepthNormalizer(bit_depth=16)
    
    # Create the hook
    hook = GenericTransformHook(
        backend=backend,
        variant_prefix="test_no_noise",
        normalizer=normalizer,
        return_original_dtype=False,
    )
    
    # Apply the hook
    variants = list(hook(temp_uint16_image))
    assert len(variants) == 1
    variant = variants[0]
    
    # Load and normalize the original
    raw = tiff.imread(str(temp_uint16_image))
    normalized_original, _ = normalizer.normalize(raw, path=temp_uint16_image)
    
    # With zero noise, output should be very close to original (within floating point precision)
    assert np.allclose(variant.image, normalized_original, atol=1e-6)


def test_generic_hook_with_dtype_conversion(temp_uint16_image):
    """
    Test GenericTransformHook with return_original_dtype=True.
    Should denormalize back to uint16.
    """
    transform = A.GaussNoise(
        std_range=(0.01, 0.01),
        mean_range=(0.0, 0.0),
        p=1.0
    )
    backend = AlbumentationsBackend(transform=transform)
    normalizer = BitDepthNormalizer(bit_depth=16)
    
    hook = GenericTransformHook(
        backend=backend,
        variant_prefix="test_dtype",
        normalizer=normalizer,
        return_original_dtype=True,  # Convert back to original dtype
    )
    
    variants = list(hook(temp_uint16_image))
    variant = variants[0]
    
    # Should be back to uint16
    assert variant.image.dtype == np.uint16
    assert variant.orig_dtype == "uint16"
    assert variant.norm_info is not None
    
    # Values should be in valid uint16 range
    assert variant.image.min() >= 0
    assert variant.image.max() <= 65535


def test_generic_hook_per_channel_transform(temp_uint16_image):
    """
    Test GenericTransformHook with per_chan=True.
    Should apply transform to each channel independently with different seeds.
    """
    transform = A.GaussNoise(
        std_range=(0.02, 0.02),
        mean_range=(0.0, 0.0),
        p=1.0
    )
    backend = AlbumentationsBackend(transform=transform)
    normalizer = BitDepthNormalizer(bit_depth=16)
    
    hook = GenericTransformHook(
        backend=backend,
        variant_prefix="test_perchan",
        normalizer=normalizer,
        per_chan=True,
        return_original_dtype=False,
    )
    
    variants = list(hook(temp_uint16_image))
    variant = variants[0]
    
    # Load original
    raw = tiff.imread(str(temp_uint16_image))
    normalized_original, _ = normalizer.normalize(raw, path=temp_uint16_image)
        
    # But each channel should still be close to its original
    for i in range(variant.image.shape[0]):
        assert np.allclose(variant.image[i], normalized_original[i], atol=0.1)


def test_generic_hook_reproducibility(temp_uint16_image):
    """
    Test that applying the same hook twice produces identical results
    (seed is deterministic based on path hash).
    """
    transform = A.GaussNoise(
        std_range=(0.01, 0.01),
        mean_range=(0.0, 0.0),
        p=1.0
    )
    backend = AlbumentationsBackend(transform=transform)
    normalizer = BitDepthNormalizer(bit_depth=16)
    
    hook = GenericTransformHook(
        backend=backend,
        variant_prefix="test_repro",
        normalizer=normalizer,
    )
    
    # Apply hook twice
    variants1 = list(hook(temp_uint16_image))
    variants2 = list(hook(temp_uint16_image))
    
    # Results should be identical
    assert np.array_equal(variants1[0].image, variants2[0].image)


def test_generic_hook_fixed_seed(temp_uint16_image):
    """
    Test that using fixed_seed produces identical results when applied to the same image twice.
    """
    transform = A.GaussNoise(
        std_range=(0.01, 0.01),
        mean_range=(0.0, 0.0),
        p=1.0
    )
    backend = AlbumentationsBackend(transform=transform)
    normalizer = BitDepthNormalizer(bit_depth=16)
    
    hook = GenericTransformHook(
        backend=backend,
        variant_prefix="test_fixed",
        normalizer=normalizer,
        fixed_seed=12345,  # Fixed seed
    )
    
    # Apply to the same image twice
    variants1 = list(hook(temp_uint16_image))
    variants2 = list(hook(temp_uint16_image))
    
    # With a fixed seed, the results should be identical
    assert np.array_equal(variants1[0].image, variants2[0].image)
