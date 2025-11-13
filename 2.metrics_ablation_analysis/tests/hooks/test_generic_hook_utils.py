"""
test_generic_hook_utils.py
"""

from pathlib import Path
import pytest
import albumentations as A

from image_ablation_analysis.hooks.generic_hook import (
    _normalize_for_hash,
    _stable_hash_from_params,
    _seed_from_path,
)


class TestNormalizeForHash:
    """Test _normalize_for_hash utility function."""

    def test_normalize_simple_dict(self):
        """Test normalization of a simple dictionary."""
        obj = {"b": 2, "a": 1, "c": 3}
        result = _normalize_for_hash(obj)
        # Should sort keys
        assert list(result.keys()) == ["a", "b", "c"]
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_normalize_nested_dict(self):
        """Test normalization of nested dictionaries."""
        obj = {"outer": {"z": 3, "y": 2, "x": 1}}
        result = _normalize_for_hash(obj)
        assert list(result["outer"].keys()) == ["x", "y", "z"]

    def test_normalize_tuple_to_list(self):
        """Test that tuples are converted to lists."""
        obj = {"range": (0.0, 1.0), "value": 5}
        result = _normalize_for_hash(obj)
        assert isinstance(result["range"], list)
        assert result["range"] == [0.0, 1.0]

    def test_normalize_rounds_floats(self):
        """Test that floats are rounded to 8 decimal places."""
        obj = {"value": 0.123456789012345}
        result = _normalize_for_hash(obj)
        assert result["value"] == 0.12345679

    def test_normalize_nested_list(self):
        """Test normalization of nested lists."""
        obj = {"items": [{"b": 2, "a": 1}, {"d": 4, "c": 3}]}
        result = _normalize_for_hash(obj)
        assert isinstance(result["items"], list)
        assert list(result["items"][0].keys()) == ["a", "b"]
        assert list(result["items"][1].keys()) == ["c", "d"]


class TestStableHashFromParams:
    """Test _stable_hash_from_params utility function."""

    def test_identical_params_same_hash(self):
        """Test that identical parameters produce the same hash."""
        params1 = {"mean": 0.0, "std": 0.1, "p": 1.0}
        params2 = {"mean": 0.0, "std": 0.1, "p": 1.0}
        hash1 = _stable_hash_from_params(params1)
        hash2 = _stable_hash_from_params(params2)
        assert hash1 == hash2

    def test_different_order_same_hash(self):
        """Test that different key order produces the same hash."""
        params1 = {"mean": 0.0, "std": 0.1, "p": 1.0}
        params2 = {"p": 1.0, "std": 0.1, "mean": 0.0}
        hash1 = _stable_hash_from_params(params1)
        hash2 = _stable_hash_from_params(params2)
        assert hash1 == hash2

    def test_different_params_different_hash(self):
        """Test that different parameters produce different hashes."""
        params1 = {"mean": 0.0, "std": 0.1, "p": 1.0}
        params2 = {"mean": 0.0, "std": 0.2, "p": 1.0}
        hash1 = _stable_hash_from_params(params1)
        hash2 = _stable_hash_from_params(params2)
        assert hash1 != hash2

    def test_volatile_fields_ignored(self):
        """Test that volatile fields don't affect the hash."""
        params1 = {
            "mean": 0.0,
            "std": 0.1,
            "p": 1.0,
        }
        params2 = {
            "mean": 0.0,
            "std": 0.1,
            "p": 1.0,
            "__class_fullname__": "SomeClass",
            "id": "12345",
            "random_state": {"foo": "bar"},
        }
        hash1 = _stable_hash_from_params(params1)
        hash2 = _stable_hash_from_params(params2)
        assert hash1 == hash2

    def test_hash_length(self):
        """Test that hash is truncated to 16 characters."""
        params = {"mean": 0.0, "std": 0.1}
        hash_val = _stable_hash_from_params(params)
        assert len(hash_val) == 16

    def test_albumentation_transform_hashing(self):
        """Test hashing with real Albumentations transform .to_dict()."""
        # Create two identical transforms
        transform1 = A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1.0)
        transform2 = A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1.0)
        
        dict1 = transform1.to_dict()
        dict2 = transform2.to_dict()
        
        hash1 = _stable_hash_from_params(dict1)
        hash2 = _stable_hash_from_params(dict2)
        
        # Should produce same hash despite different random_state, id, etc.
        assert hash1 == hash2

    def test_albumentation_different_params(self):
        """Test that different Albumentations params produce different hashes."""
        transform1 = A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1.0)
        transform2 = A.GaussianBlur(blur_limit=(5, 9), sigma_limit=0, p=1.0)
        
        dict1 = transform1.to_dict()
        dict2 = transform2.to_dict()
        
        hash1 = _stable_hash_from_params(dict1)
        hash2 = _stable_hash_from_params(dict2)
        
        assert hash1 != hash2

    def test_albumentation_compose_hashing(self):
        """Test hashing with Albumentations Compose transform."""
        # Create two identical composed transforms
        compose1 = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1.0),
            A.HorizontalFlip(p=0.5),
        ])
        compose2 = A.Compose([
            A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1.0),
            A.HorizontalFlip(p=0.5),
        ])
        
        dict1 = compose1.to_dict()
        dict2 = compose2.to_dict()
        
        hash1 = _stable_hash_from_params(dict1)
        hash2 = _stable_hash_from_params(dict2)
        
        # Should produce same hash
        assert hash1 == hash2

    def test_albumentation_volatile_fields_dont_change_hash(self):
        """Test that modifying volatile fields in dict doesn't change hash."""
        transform = A.GaussianBlur(blur_limit=(3, 7), sigma_limit=0, p=1.0)
        dict1 = transform.to_dict()
        
        # Create a copy and modify volatile fields
        dict2 = transform.to_dict()
        dict2["__class_fullname__"] = "DifferentClass"
        dict2["id"] = "different_id"
        dict2["random_state"] = {"completely": "different"}
        
        hash1 = _stable_hash_from_params(dict1)
        hash2 = _stable_hash_from_params(dict2)
        
        # Hashes should still be identical
        assert hash1 == hash2


class TestSeedFromPath:
    """Test _seed_from_path utility function."""

    def test_same_path_same_seed(self):
        """Test that the same path produces the same seed."""
        path = Path("/fake/path/to/image.tiff")
        seed1 = _seed_from_path(path)
        seed2 = _seed_from_path(path)
        assert seed1 == seed2

    def test_different_paths_different_seeds(self):
        """Test that different paths produce different seeds."""
        path1 = Path("/fake/path/to/image1.tiff")
        path2 = Path("/fake/path/to/image2.tiff")
        seed1 = _seed_from_path(path1)
        seed2 = _seed_from_path(path2)
        assert seed1 != seed2

    def test_same_stem_different_dir_different_seeds(self):
        """Test that same filename in different dirs produces different seeds."""
        path1 = Path("/fake/dir1/image.tiff")
        path2 = Path("/fake/dir2/image.tiff")
        seed1 = _seed_from_path(path1)
        seed2 = _seed_from_path(path2)
        # These should be different because full path differs
        assert seed1 != seed2

    def test_salt_changes_seed(self):
        """Test that different salt values produce different seeds."""
        path = Path("/fake/path/to/image.tiff")
        seed1 = _seed_from_path(path, salt="salt1")
        seed2 = _seed_from_path(path, salt="salt2")
        assert seed1 != seed2

    def test_same_salt_same_seed(self):
        """Test that same path and salt produce the same seed."""
        path = Path("/fake/path/to/image.tiff")
        salt = "my_salt"
        seed1 = _seed_from_path(path, salt=salt)
        seed2 = _seed_from_path(path, salt=salt)
        assert seed1 == seed2
