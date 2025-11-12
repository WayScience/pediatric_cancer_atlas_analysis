"""
ablation_runner.py

Runner for image ablation analysis on a nested folder structure of tiff images.
Saving all ablated images to a specified output folder with same structure.

Classes:
- AugVariant: Container for a single augmented variant of an image.
- AblationRunner: Scaffolding to run ablation analysis on images in a folder structure
"""


from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Any, Union
)

import tifffile as tiff
import pandas as pd
from tqdm import tqdm

from .utils import check_path
from .indexing import ParquetIndex, LoadDataIndex


TIFF_SUFFIXES = {".tiff", ".tif"}


@dataclass
class AugVariant:
    """
    Container for a single augmented variant of an image.
    """
    variant: str
    image: Any
    params: Dict[str, Any]
    orig_dtype: Optional[str] = None
    norm_info: Optional[Dict[str, Any]] = None


class AblationRunner:
    """
    Scaffolding to run ablation analysis on images in a folder structure.

    Core functionalities:
    - Recursively find all tiff images in input folder included in loaddata CSVs.
    - Keep track of raw image metadata from loaddata.
    - Keep track of ablated images and parameters in a Parquet index.
    - Write ablated images to output folder with mirrored structure.
    - Evaluates pre-configured augment_hook for each image path with
        `run()` method to allow for applying multiple ablations per each image.
    - Writes sidecar JSON files with augmentation metadata with the same file
        name as the ablated image to facilitate later analysis
        (in case index breaks).
    """

    def __init__(
        self,
        images_root: Union[str, Path],
        ablation_root: Union[str, Path],
        loaddata_csvs: List[Path | str],
        *,
        index_filename: str = "ablated_index",
        keep_meta_columns: Optional[List[str]] = None,
        suffixes: Optional[Iterable[str]] = TIFF_SUFFIXES,
        skip_if_indexed: bool = True,
        dry_run: bool = False,
    ):
        """
        Initialize AblationRunner.

        :param images_root: Root folder containing source images.
        :param ablation_root: Root folder to write ablated images.
        :param loaddata_csvs: List of CSV files defining source images and metadata.
        :param index_filename: Filename for the Parquet index under ablation_root.
        :param keep_meta_columns: List of metadata columns to keep from loaddata.
            If None, keeps all columns present in loaddata.
        :param suffixes: Iterable of file suffixes (including dot) to consider as images.
        :param skip_if_indexed: If True, skip images already present in index.
        :param dry_run: If True, do not write any files; only simulate.
        """
        self.images_root = check_path(images_root, ensure_dir=True)
        self.ablation_root = check_path(ablation_root, ensure_dir=True)
        
        self.suffixes = suffixes

        self.loaddata = LoadDataIndex(
            [Path(p) for p in loaddata_csvs]
        )

        self.index_filename = index_filename
        self.index_path = self.ablation_root / self.index_filename
        self.index = ParquetIndex(self.index_path)
        self._default_meta_cols = self.loaddata.columns
        self.keep_meta_columns = keep_meta_columns or self.loaddata.columns

        self.skip_if_indexed = skip_if_indexed
        self.dry_run = dry_run        

    def run(
        self,
        *,
        augment_hook: Callable[[Path], Iterator[AugVariant]],
        writer: Optional[callable[[Path, Any], None]] = None,
        run_id: Optional[str] = None,
    ) -> None:
        """
        Run ablation analysis using the provided augment_hook.

        :param augment_hook: Callable that takes a source image path and
            yields AugVariant instances.
        :param writer: Optional callable to write images. 
            If None, uses default tifffile
        :param run_id: Optional run identifier. If None, generates a UUID. 
        """

        writer = writer or self._default_tiff_writer
        run_id = run_id or str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")

        # stable configuration id to check for existing records and skip
        config_id = getattr(augment_hook, "config_id", None)

        # iterate images from loaddata (authoritative), 
        # TODO add option for rglob
        batch_rows: List[Dict[str, Any]] = []
        src_files = [Path(p).resolve() for p in self._iter_source_images()]
        
        if self.skip_if_indexed and config_id:
            already = self.index.list_done_paths_for(config_id)
            already_posix = {Path(p).as_posix() for p in already}
            print(f"Skipping {len(already_posix)} paths due to ablation [{config_id}] already exists.")
            src_files = [p for p in src_files if p.as_posix() not in already_posix]
        
        pbar = tqdm(src_files, desc="Applying ablations ...", unit="image")
        # TODO parallelize this maybe
        for src_abs in pbar:
            rel = self._safe_relative_to_root(src_abs)
            meta = self.loaddata.metadata_for(src_abs)

            # Generate variants
            try:
                variants = list(augment_hook(src_abs))
            except Exception as e:
                print(f"[WARN] augment_hook failed on {src_abs}: {e}")
                continue
            if not variants:
                continue

            pbar.set_description("Applying ablations ...")

            # Write each variant + stage index rows
            for av in variants:
                
                cid = av.params.get("config_id", config_id)

                dst = self._destination_path(rel, av.variant)
                if not self.dry_run:
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    writer(dst, av.image)
                    # Optional: sidecar JSON
                    self._write_sidecar(dst, av)

                row = {
                    "created_at": created_at,
                    "run_id": run_id,
                    "original_abs_path": str(src_abs),
                    "original_rel_path": rel.as_posix(),
                    "aug_abs_path": str(dst.resolve()),
                    "aug_rel_path": dst.resolve().relative_to(self.ablation_root).as_posix(),
                    "variant": av.variant,
                    "config_id": cid,
                    "params_json": json.dumps(av.params, separators=(",", ":")),
                }

                # attach selected metadata fields
                for k, v in self._select_meta(meta).items():
                    # keep strings scalars; convert lists/tuples/dicts to JSON
                    if isinstance(v, (list, dict, tuple)):
                        row[k] = json.dumps(v, separators=(",", ":"))
                    else:
                        row[k] = v
                
                batch_rows.append(row)

        # Append once to Parquet for efficiency
        if not self.dry_run and batch_rows:
            df = pd.DataFrame(batch_rows)
            self.index.append_records(df)
            print("Updated ablation index.")

    # Internal helpers

    def _iter_source_images(self) -> Iterator[Path]:
        """
        From loaddata, iterate all absolute image paths that are TIFF-like, 
        and are under images_root. Used by the `run()` method.
        """
        for p in self.loaddata.iter_all_abs_paths():
            if p.suffix.lower() in self.suffixes:
                # Optionally filter to those under images_root;
                yield p

    def _safe_relative_to_root(self, abs_path: Path) -> Path:
        """
        Compute relative path to images_root if possible; 
            else fall back to filename-only. 
        Useful for mirroring raw image folder structure under ablation_root.
        """
        try:
            return abs_path.relative_to(self.images_root)
        except ValueError:
            # Not under the declared root; 
            # fall back to channel subfolders using metadata hints
            return Path(abs_path.name)

    def _destination_path(self, rel_src: Path, variant: str) -> Path:
        """
        Mirror the directory structure under ablation_root; 
        append `__{variant}` to the stem and normalize to .tiff.
        Useful for mirroring raw image folder structure under ablation_root.
        """
        out_name = f"{rel_src.stem}__{variant}.tiff"
        return (self.ablation_root / rel_src.parent / out_name)

    def _write_sidecar(self, dst_path: Path, av: AugVariant) -> None:
        """
        Write sidecar JSON file with augmentation metadata. 
        Called by `run()` alongside writing the ablated image.
        """
        sidecar = dst_path.with_suffix(".json")
        payload = {"variant": av.variant, "params": av.params}
        with sidecar.open("w") as f:
            json.dump(payload, f, indent=2)

    def _default_tiff_writer(self, out_path: Path, image: Any) -> None:
        # Default tifffile writer
        tiff.imwrite(str(out_path), image, bigtiff=True)

    def _select_meta(self, meta_row: Mapping[str, Any]) -> Dict[str, Any]:
        # Select metadata columns to keep from loaddata for index.
        if not meta_row:
            return {}
        if self.keep_meta_columns is None:
            keep = [c for c in self._default_meta_cols if c in meta_row]
        else:
            keep = [c for c in self.keep_meta_columns if c in meta_row]
        return {k: meta_row[k] for k in keep}
