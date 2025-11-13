"""
indexing.py

Indexing utilities for image ablation analysis. Used by the runner class
to keep track of raw images, metadata, where the ablated variants should
be stored, and what has already been processed.

Classes:
- ParquetIndex: append-only Parquet dataset writer/reader, used to index
    augmented images and their original metadata.
- LoadDataIndex: parses CellProfiler loaddata CSV(s) for channel path discovery
    and metadata mapping to the ablated variants. Also can be used to define
    the set of images to process. 
"""

import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterator, List, Any, Mapping, Optional

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq


class ParquetIndex:
    """
    Append-only Parquet dataset writer/reader.

    We write each run to a new file in `index_dir` (a dataset directory).
    You can query later with pyarrow.dataset or pandas.read_parquet.
    """

    def __init__(self, index_dir: Path):
        """
        Initialize the ParquetIndex with the given directory.
        Creates the directory if it does not exist.
        """
        self.index_dir = index_dir
        self.index_dir.mkdir(parents=True, exist_ok=True)

    # Internal helpers

    def _schema(self) -> pa.schema:
        """
        Minimal Parquet schema for the index.
        Real schema is inferred on first write as arbitrary metadata columns
        may be included.
        """
        fields = [
            pa.field("created_at", pa.timestamp("us")),
            pa.field("run_id", pa.string()),
            pa.field("original_abs_path", pa.string()),
            pa.field("original_rel_path", pa.string()),
            pa.field("aug_abs_path", pa.string()),
            pa.field("aug_rel_path", pa.string()),
            pa.field("variant", pa.string()),
            pa.field("config_id", pa.string()),
            pa.field("params_json", pa.string()),
        ]
        return pa.schema(fields)
    
    def _validate_required_columns(self, schema: pa.Schema) -> None:
        """
        Validate that a schema contains all required columns from _schema().
        
        :param schema: PyArrow schema to validate
        :raises ValueError: If required columns are missing
        """
        required_cols = {f.name for f in self._schema()}
        actual_cols = set(schema.names)
        missing = required_cols - actual_cols
        
        if missing:
            raise ValueError(
                f"Schema missing required columns: {sorted(missing)}. "
                f"Required: {sorted(required_cols)}, "
                f"Found: {sorted(actual_cols)}"
            )

    def _open_dataset_safe(self) -> Optional[ds.Dataset]:
        """
        Open the Parquet dataset. 
        Internal helper used by functions that need to look-up existing records.

        :return: ds.Dataset or None if dataset is empty/nonexistent.
        """
        try:
            dset = ds.dataset(
                self.index_dir,
                format="parquet",
                partitioning="hive",
                ignore_prefixes=[".", "_tmp"]  # ignore temp/hidden
            )
        except (FileNotFoundError, pa.ArrowInvalid):
            return None

        # If there are no fragments/files, treat as empty
        try:
            # Fast check: if schema only has virtual columns (no fields), or no fragments
            # Some versions don't expose fragments easily; guard with try/except.
            if dset.schema is None or len(dset.schema) == 0:
                return None
        except Exception:
            pass
        return dset
    
    # Public methods for runner use

    def append_records(self, df: pd.DataFrame) -> None:
        """
        Append a batch of rows by creating a new parquet file under index_dir.
        Intended to be used by the runner after processing a batch of images
            to update the index to reflect newly created ablations.

        :param df: DataFrame containing the rows to append.
        """
        if df.empty:
            return
        table = pa.Table.from_pandas(df, preserve_index=False)
        # use a time-based unique file name
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        file_path = self.index_dir / f"aug_index_{ts}.parquet"
        pq.write_table(table, file_path)

        return None

    def read(self) -> pd.DataFrame:
        """
        Read the entire index as a pandas DataFrame.
        Intended for testing and analysis purposes.

        :return: DataFrame containing all index records, or empty DataFrame if no data.
        """
        dset = self._open_dataset_safe()
        if dset is None:
            return pd.DataFrame()
        
        try:
            return dset.to_table().to_pandas()
        except Exception:
            return pd.DataFrame()

    def materialize_seen_pairs(self) -> set[tuple[str, str]]:
        """
        Load existing (original_abs_path, config_id) pairs, returning an empty set
        if the dataset is empty or columns aren't present yet. 
        Intended to be used by the runner to obtain a record of what has already
        been processed and skip on a single-image basis.

        :return: set of (original_abs_path, config_id) pairs already in the index.
        """
        dset = self._open_dataset_safe()
        if dset is None:
            return set()

        required = {"original_abs_path", "config_id"}
        have = {f.name for f in dset.schema}
        if not required.issubset(have):
            # No data files written yet (or old schema) → nothing to skip
            return set()

        pairs: set[tuple[str, str]] = set()
        try:
            scanner = dset.scanner(columns=list(required))
            for batch in scanner.to_batches():
                
                tab = pa.Table.from_batches([batch]).to_pandas()[
                    ["original_abs_path", "config_id"]
                ]
                pairs.update(map(
                    tuple, 
                    tab[["original_abs_path", "config_id"]].itertuples(
                        index=False, name=None)))
        except (FileNotFoundError, pa.ArrowInvalid):
            # Treat as empty if anything schema/files related pops
            return set()
        return pairs

    def list_done_paths_for(self, config_id: str) -> set[str]:
        """
        Return a set of original_abs_path strings that already have rows 
        with this config_id. For batch retrieval of what has been processed
        by the runner. 

        Uses partition pruning if you partitioned by config_id.
        """
        dset = self._open_dataset_safe()
        if dset is None:
            return set()

        needed = {"original_abs_path", "config_id"}
        if not needed.issubset({f.name for f in dset.schema}):
            return set()

        import pyarrow.dataset as ds
        filt = ds.field("config_id") == config_id

        paths = set()
        try:
            scanner = dset.scanner(filter=filt, columns=["original_abs_path"])
            for batch in scanner.to_batches():
                col = batch.to_pandas()["original_abs_path"].astype(str).str.replace("\\", "/", regex=False)
                paths.update(col.tolist())
        except Exception:
            return set()
        return paths
    

class LoadDataIndex:
    """
    Parses CellProfiler LoadData CSV(s) and exposes:
      - channel path discovery (FileName_* + PathName_*)
      - mapping: absolute file path -> metadata row (dict-like)
    """

    FILE_RE = re.compile(r"^FileName_(.+)$")
    PATH_RE = re.compile(r"^PathName_(.+)$")

    def __init__(self, csv_paths: List[Path]):
        """
        Initialize the LoadDataIndex by loading and parsing the given CSV(s).
        Raises FileNotFoundError if any CSV is missing.

        :param csv_paths: List of paths to CellProfiler LoadData CSV files.
        """

        if any(not p.is_file() for p in csv_paths):
            missing = [str(p) for p in csv_paths if not p.is_file()]
            raise FileNotFoundError(f"LoadData CSV(s) not found: {missing}")

        # Concatenate multiple loaddata CSVs if needed
        dfs = []
        for p in csv_paths:
            df = pd.read_csv(p)
            df["__loaddata_source"] = str(p)
            dfs.append(df)
        self.df = pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

        # freeze original columns
        self._columns = list(self.df.columns) 
        self._columns.remove("__loaddata_source")

        # detect channel name stems
        file_cols = [c for c in self.df.columns if self.FILE_RE.match(c)]
        path_cols = [c for c in self.df.columns if self.PATH_RE.match(c)]
        file_stems = {self.FILE_RE.match(c).group(1) for c in file_cols}
        path_stems = {self.PATH_RE.match(c).group(1) for c in path_cols}
        self.channel_stems = sorted(file_stems & path_stems)  # only pairs present in both

        # Build absolute path columns for each channel stem
        for stem in self.channel_stems:
            fcol = f"FileName_{stem}"
            pcol = f"PathName_{stem}"
            acol = f"AbsPath_{stem}"
            self.df[acol] = (
                self.df[pcol].astype(str).str.rstrip("/\\") + 
                "/" + self.df[fcol].astype(str)
            ).str.replace("\\", "/", regex=False)

        # Flatten to a long table: one row per (image, channel)
        abs_path_cols = [f"AbsPath_{stem}" for stem in self.channel_stems]
        if not abs_path_cols:
            # Empty case: no channels detected
            self.long_df = pd.DataFrame(columns=[
                "__channel", "__abs_path", "__meta_row"])
        else:
            # Melt absolute path columns while keeping all other columns as id_vars
            id_vars = [c for c in self.df.columns if c not in abs_path_cols]
            self.long_df = self.df.melt(
                id_vars=id_vars,
                value_vars=abs_path_cols,
                var_name="__channel_col",
                value_name="__abs_path"
            )
            
            # Extract channel stem from the column name (AbsPath_DNA -> DNA)
            self.long_df["__channel"] = self.long_df["__channel_col"].str.replace(
                "AbsPath_", "", regex=False
            )
            self.long_df = self.long_df.drop(columns=["__channel_col"])
            
            # Reconstruct __meta_row as dict for each row
            # Drop AbsPath columns since they're in __abs_path; keep other metadata
            meta_cols = [c for c in id_vars if not c.startswith("AbsPath_")]
            self.long_df["__meta_row"] = self.long_df[meta_cols].apply(
                lambda row: row.to_dict(), axis=1
            )
            
            # Keep only the columns we need
            self.long_df = self.long_df[["__channel", "__abs_path", "__meta_row"]]
            
            # Remove rows with null paths
            self.long_df = self.long_df.dropna(subset=["__abs_path"])

        # Build the main lookup: absolute path (normalized) -> metadata row (dict)
        self.path_to_meta: Dict[str, Mapping[str, Any]] = {}
        for _, row in self.long_df.iterrows():
            abs_p = str(Path(row["__abs_path"]).resolve())
            self.path_to_meta[abs_p] = row["__meta_row"]

    def metadata_for(self, abs_path: Path) -> Mapping[str, Any]:
        """
        Allows to retrieve the metadata columns for a specific absolute image
            path that are in the same row in the original LoadData CSV.
        Intended to be used by the runner to populate metadata in the ablation
            index through ParquetIndex.

        :param abs_path: Absolute path to the image file. By default only
            the resolved absolute path is used for lookup to avoid ambiguity.

        :return: Dictionary of metadata columns for the given image path.
        """
        key = str(abs_path.resolve())
        return self.path_to_meta.get(key, {})

    def iter_all_abs_paths(self) -> Iterator[Path]:
        """
        Iterate over all absolute image paths in the LoadData CSV(s).
        Intended to be used by the runner to define the set of images to process.

        :return: Iterator over absolute image paths as Path objects.
        """
        for p in self.long_df["__abs_path"].unique():
            yield Path(p).resolve()

    @property
    def columns(self) -> List[str]:
        """
        Property exposing the original LoadData CSV columns.
        Useful for the runner to know what metadata fields are available and
            select all to copy over to the ablation index as default behavior.
        
        :return: List of original LoadData CSV column names.
        """
        return self._columns
    
    def __len__(self) -> int:
        """
        Used by the runner to get the total number of image-channel pairs 
            for tqdm progress.
        """
        return len(self.long_df)
