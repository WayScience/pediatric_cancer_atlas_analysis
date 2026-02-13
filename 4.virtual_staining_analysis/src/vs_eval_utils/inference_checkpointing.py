"""
inference_checkpointing.py

Helper module for checkpointing and resuming inference runs.
- Initializes a location for writing the index files
- Keeps tracking of the metadata columns that uniquely identifies models 
    and inference tasks
- Checks new inference tasks against existing checkpoint and determine what
    is truely new and needs to be ran and what should be skipped.
- Provides helper utility for the inference running helper to update
    checkpoint files. 
"""

from dataclasses import dataclass, field
from datetime import datetime
import pathlib
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass
class _CheckpointColumnSchema:
    """
    Column configuration for checkpoint index files.
    Defines which columns are used to identify inference tasks and model runs,
    """

    # Columns that uniquely identify an inference task.
    # Modify as needed
    metadata: list[str] = field(default_factory=lambda: [
        "platemap_file",
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
    ])
    # Extra bookkeeping columns carried alongside tasks
    # (not part of the key).
    # Add/remove as needed
    metadata_bookkeeping: list[str] = field(default_factory=lambda: [        
        "time_point",
        "seeding_density",
    ])
    
    # Model-logging columns that uniquely identify a model run.
    # Modify as needed, here the run_id is from the mlflow logging records
    model: list[str] = field(default_factory=lambda: [
        "run_id"
    ])
    
    # Extra bookkeeping columns from model records.
    # (not part of the key). 
    # Add/remove as needed
    model_bookkeeping: list[str] = field(default_factory=lambda: [
        "architecture", # model architecture
        "density", # what density is the model training data
        "path", # where the logging artifacts are located
    ])

    # -- derived column lists ------------------------------------------------

    @property
    def model_prefixed(self) -> list[str]:
        return [f"Metadata_Model_{c}" for c in self.model]

    @property
    def model_bookkeeping_prefixed(self) -> list[str]:
        return [f"Metadata_Model_{c}" for c in self.model_bookkeeping]

    @property
    def index_columns(self) -> list[str]:
        """Columns forming the unique (task x model) multi-index key."""
        return self.metadata + self.model_prefixed

    @property
    def all_columns(self) -> list[str]:
        """Every column stored in a checkpoint parquet file."""
        return (
            self.metadata
            + self.model_prefixed
            + self.metadata_bookkeeping
            + self.model_bookkeeping_prefixed
            + ["output_file"]
        )

    def build_entry(
        self,
        task: pd.Series | dict,
        model_metadata: pd.Series | dict,
        output_file: pathlib.Path,
    ) -> dict:
        """
        Assemble one checkpoint-index row from task + model metadata.
        """
        return {
            **{c: task[c] for c in self.metadata},
            **{f"Metadata_Model_{c}": model_metadata[c] for c in self.model},
            **{c: (task[c] if c in task else None) for c in self.metadata_bookkeeping},
            **{
                f"Metadata_Model_{c}": (
                    model_metadata[c] if c in model_metadata else None
                )
                for c in self.model_bookkeeping
            },
            "output_file": str(output_file),
        }
    

@dataclass
class _CheckpointSession:
    """
    All mutable state for an active checkpointing session.
    """

    root: pathlib.Path
    index_root: pathlib.Path
    index_df: pd.DataFrame
    multi_index: pd.MultiIndex | None = None
    new_subdir: pathlib.Path | None = None
    new_part: int = 0


_columns = _CheckpointColumnSchema()
_session: _CheckpointSession | None = None

# Backward-compatible names imported by sibling modules (model_inference.py).
# At import time these reference the *same* list objects owned by _columns.
METADATA_COLUMNS: list[str] = _columns.metadata
METADATA_COLUMNS_BOOKKEEPING: list[str] = _columns.metadata_bookkeeping
METADATA_MODEL_COLUMNS: list[str] = _columns.model
METADATA_MODEL_COLUMNS_BOOKKEEPING: list[str] = _columns.model_bookkeeping


# ── Session lifecycle ─────────────────────────────────────────────────────


def set_checkpoint_index(
    checkpoint_root: pathlib.Path,
    checkpoint_index_path: pathlib.Path | None = None,
) -> None:
    """
    Initialise (or re-initialise) the global checkpoint session.

    Loads existing checkpoint parquet shards from *checkpoint_index_path*
    and creates a new timestamped subdirectory for this session's writes.

    :param checkpoint_root: Root directory for inference outputs.
    :param checkpoint_index_path: Directory holding checkpoint parquet files.
        Defaults to ``checkpoint_root / "checkpoint_index"``.
    """
    global _session

    if not checkpoint_root.exists():
        raise RuntimeError(
            f"Checkpoint root directory {checkpoint_root} does not exist."
        )

    if checkpoint_index_path is None:
        checkpoint_index_path = checkpoint_root / "checkpoint_index"
        checkpoint_index_path.mkdir(parents=True, exist_ok=True)
    elif not checkpoint_index_path.exists():
        raise RuntimeError(
            f"Checkpoint index path {checkpoint_index_path} does not exist."
        )

    # Load existing parquet shards
    collected = list(checkpoint_index_path.rglob("*.parquet"))
    if collected:
        index_df = pd.concat(
            (pd.read_parquet(p) for p in collected),
            ignore_index=True,
        )
    else:
        index_df = pd.DataFrame(columns=_columns.all_columns)

    if (
        not all(c in index_df.columns for c in _columns.metadata)
        and not all(c in index_df.columns for c in _columns.model_prefixed)
    ):
        raise ValueError(
            "Checkpoint index file is missing required metadata columns."
        )

    multi_index = None
    if not index_df.empty:
        multi_index = pd.MultiIndex.from_frame(
            index_df[_columns.index_columns]
        )

    new_subdir = checkpoint_index_path / datetime.now().strftime(
        "%Y-%m-%dT%H-%M-%S-%f"
    )
    new_subdir.mkdir(parents=True, exist_ok=True)

    _session = _CheckpointSession(
        root=checkpoint_root,
        index_root=checkpoint_index_path,
        index_df=index_df,
        multi_index=multi_index,
        new_subdir=new_subdir,
    )


def teardown_checkpoint_index() -> None:
    """
    Tear down the current checkpoint session.

    Iterates over all immediate subdirectories under the checkpoint index
    root and removes any that are empty (e.g. timestamped run directories
    that received no checkpoint writes).
    """
    global _session
    if _session is None:
        return

    for subdir in sorted(_session.index_root.iterdir()):
        if subdir.is_dir() and not any(subdir.iterdir()):
            subdir.rmdir()

    _session = None


def get_checkpoint_index() -> pd.DataFrame | None:
    """
    Get the global checkpoint index DataFrame.

    :return: The checkpoint index DataFrame, or None if it has not been set.
    """
    return _session.index_df if _session is not None else None


# ── Column configuration ─────────────────────────────────────────────────


def set_metadata_columns(columns: list[str]) -> None:
    """
    Set the metadata columns to use for checkpointing and resuming inference runs.

    :param columns: List of column names to use as metadata for identifying inference tasks.
    """
    global METADATA_COLUMNS
    _columns.metadata = columns
    METADATA_COLUMNS = columns


def set_metadata_columns_bookkeeping(columns: list[str]) -> None:
    """
    Set the bookkeeping metadata columns to include in checkpoint index files.

    :param columns: List of column names to include as bookkeeping metadata.
    """
    global METADATA_COLUMNS_BOOKKEEPING
    _columns.metadata_bookkeeping = columns
    METADATA_COLUMNS_BOOKKEEPING = columns


def set_metadata_model_columns(columns: list[str]) -> None:
    """
    Set the metadata model columns to use for checkpointing and resuming inference runs.

    :param columns: List of column names from model logging records to use as metadata for identifying model runs.
    """
    global METADATA_MODEL_COLUMNS
    _columns.model = columns
    METADATA_MODEL_COLUMNS = columns


def set_metadata_model_columns_bookkeeping(columns: list[str]) -> None:
    """
    Set the bookkeeping metadata model columns to include in checkpoint index files.

    :param columns: List of column names from model logging records to include as bookkeeping metadata.
    """
    global METADATA_MODEL_COLUMNS_BOOKKEEPING
    _columns.model_bookkeeping = columns
    METADATA_MODEL_COLUMNS_BOOKKEEPING = columns


# ── Task checking & checkpoint updates ────────────────────────────────────


def tasks_completed(
    tasks: pd.DataFrame | Iterable[pd.Series | dict],
    model: pd.Series | dict,
) -> pd.DataFrame:
    """
    Checks if a batch of inference tasks have already been completed against
        the global checkpoint index.

    :param tasks: Iterable of Series or dict containing metadata of the
        inference tasks to check.
    :param model: Series or dict containing metadata of the model used for the
        inference tasks.
    :return: DataFrame containing the subset of tasks that have not been
        completed according to the checkpoint index.
    """
    if _session is None:
        raise RuntimeError(
            "Checkpoint index is not set. "
            "Please call set_checkpoint_index() before checking tasks."
        )

    if not isinstance(tasks, pd.DataFrame):
        tasks = pd.DataFrame(tasks)

    if _session.multi_index is None:
        return tasks

    if not all(c in tasks.columns for c in _columns.metadata):
        raise ValueError(
            f"Tasks are missing required metadata columns. "
            f"Expected at least: {_columns.metadata}"
        )

    tasks = tasks.dropna(subset=_columns.metadata)
    tasks[_columns.model_prefixed] = [
        model[c] for c in _columns.model
    ]

    tasks_mi = pd.MultiIndex.from_frame(tasks[_columns.index_columns])
    return tasks.iloc[
        np.where(~tasks_mi.isin(_session.multi_index))[0]
    ].copy()


def assemble_update_batch(
    tasks: pd.DataFrame | Iterable[pd.Series | dict],
    output_files: list[pathlib.Path],
    model_metadata: pd.Series | dict,
) -> pd.DataFrame:
    """
    Assemble an update DataFrame for a batch of completed inference tasks.

    :param tasks: DataFrame or iterable of Series/dict with task metadata.
    :param output_files: Output file paths for the completed tasks.
    :param model_metadata: Metadata of the model used for inference.
    :return: DataFrame with one row per checkpoint entry.
    """
    if isinstance(tasks, pd.DataFrame):
        task_list = [row for _, row in tasks.iterrows()]
    else:
        task_list = list(tasks)

    if len(task_list) != len(output_files):
        raise ValueError(
            f"Number of tasks and output files must match. "
            f"Got {len(task_list)} tasks and {len(output_files)} output files."
        )

    return pd.DataFrame([
        _columns.build_entry(task, model_metadata, output_files[i])
        for i, task in enumerate(task_list)
    ])


def update_checkpoint_batch(
    tasks: pd.DataFrame | Iterable[pd.Series | dict],
    output_files: list[pathlib.Path],
    model_metadata: pd.Series | dict,
) -> None:
    """
    Update the checkpoint index with a batch of completed inference tasks.

    :param tasks: Iterable of Series or dict containing metadata of the completed inference tasks.
    :param output_files: List of output file paths corresponding to the completed inference tasks.
    :param model_metadata: Series or dict containing metadata of the model used for the inference tasks.
    """
    if _session is None:
        raise RuntimeError(
            "Checkpoint index is not set. "
            "Please call set_checkpoint_index() before updating."
        )

    _session.index_df = assemble_update_batch(tasks, output_files, model_metadata)
    part_path = _session.new_subdir / f"part_{_session.new_part}.parquet"
    _session.index_df.to_parquet(part_path, index=False)
    _session.new_part += 1
