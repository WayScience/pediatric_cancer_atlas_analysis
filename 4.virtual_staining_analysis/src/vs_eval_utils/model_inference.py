"""
model_inference.py

This module contains utilities for running inference on a specified image dataset
given a loaded model and its metadata, and write checkpoint files for indexing
what has been inferenced and what still needs to be done.
"""

import pathlib
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import torch
import tifffile as tiff

from .inference_checkpointing import (
    METADATA_COLUMNS,
    METADATA_MODEL_COLUMNS,
    tasks_completed,
    update_checkpoint_batch,
)


def construct_output_path(
    metadata_values: list[str],
    model_metadata_values: list[str],
    patch_i: int,
    output_root: pathlib.Path,
    flat: bool = False
) -> pathlib.Path:    
    """
    Construct output path for inference results based on metadata values.

    :param metadata_values: List of metadata values for the inference task.
    :param model_metadata_values: List of model metadata values for the inference task.
    :param output_root: Root directory to save inference outputs.
    :param flat: Whether to use a flat directory structure (default: False).
    :return: Path to save inference results for the given task.
    """

    if flat:
        # For flat structure, concatenate all metadata values into a single filename
        filename = "_".join(f"{value}" for value in metadata_values + model_metadata_values)
        filename = filename.replace("/", "_")  # replace any slashes in metadata values to avoid path issues
        return output_root / f"{filename}_{patch_i}.tiff"
    
    # For hierarchical structure, create subdirectories based on metadata values
    subdir = output_root
    for val in metadata_values:
        subdir /= str(val)
    
    model_subdir = subdir
    for val in model_metadata_values:
        model_subdir /= str(val)

    model_subdir.mkdir(parents=True, exist_ok=True)
    
    return model_subdir / f"{patch_i}.tiff"


def inference_and_checkpoint(
    model: torch.nn.Module,
    model_metadata: pd.Series | dict,
    tasks: pd.DataFrame,
    output_root: pathlib.Path,
    dataset: Optional[torch.utils.data.Dataset] = None,
    dataset_fn: Optional[callable] = None,
    output_flat: bool = True,
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
) -> None:
    """
    Run inference on the given dataloader using the provided model, and manage checkpointing of completed tasks.

    :param model: The PyTorch model to use for inference.
    :param model_metadata: Metadata of the model being used for inference.
    :param tasks: DataFrame containing metadata of tasks to run inference on.
    :param dataset: Dataset to run inference on, which should be compatible with the tasks DataFrame.
    :param dataset_fn: Optional function to create a dataset if dataset is None.
    :param output_root: Root directory to save inference outputs and checkpoint files.
    """

    # Use checkpoining module to determine if any tasks have already been
    # completed for this model and set of tasks, to avoid redundant inference runs.
    tasks_todo: pd.DataFrame = tasks_completed(
        tasks=tasks,
        model=model_metadata
    )
    tasks_todo.reset_index(drop=True, inplace=True)

    if tasks_todo.empty:
        print("All tasks have already been completed for this model. No inference to run.")
        return 
    
    # lazy dataset construction
    if dataset is None:
        if dataset_fn is None:
            raise ValueError("Either a dataset or a dataset_fn must be provided.")
        try:
            dataset = dataset_fn(tasks_todo)
        except Exception:
            raise ValueError("Failed to create dataset using dataset_fn.")
    
    metadata = dataset.metadata.copy()
    metadata['patch_idx'] = metadata.groupby(
        ['Metadata_Plate', 'Metadata_Well', 'Metadata_Site']).cumcount()
    meta_row_dicts = metadata.to_dict(orient='records')

    try:
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=48,
            num_workers=4,
            pin_memory=False,
            shuffle=False,
        )
    except Exception as e:
        raise RuntimeError(f"Error creating dataloader for inference: {e}")
    
    model = model.to(device)

    inference_cache: list[torch.Tensor] = []
    pbar = tqdm(dataloader, desc="Running inference", total=len(dataloader))
    with torch.inference_mode():
        for batch in pbar:
            
            inputs, _ = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            bs = outputs.shape[0]

            # temporarily store all outputs in gpu memory
            inference_cache.append(outputs)

    # once the inference completes the progress bar disappears too.
    pbar.close() 

    inference: np.ndarray = torch.cat(inference_cache, dim=0).cpu().numpy().astype(np.float32)
    # refuse to write outputs and update checkpoint if number of outputs does
    # not match number of tasks, to avoid silent data corruption
    if inference.shape[0] != len(metadata):
        raise RuntimeError(
            f"Number of inference outputs ({inference.shape[0]}) "
            f"does not match number of tasks ({len(metadata)})."
        )    

    update_paths: list[pathlib.Path] = []
    for i, task in enumerate(meta_row_dicts):         

        metadata_values: list[str] = [
            task[col] for col in METADATA_COLUMNS
        ]
        
        model_metadata_values: list[str] = [
            model_metadata[col] for col in METADATA_MODEL_COLUMNS
        ]

        output_path = construct_output_path(
            metadata_values=metadata_values,
            model_metadata_values=model_metadata_values,
            patch_i=task['patch_idx'],
            output_root=output_root,
            flat=output_flat,
        )

        try:
            tiff.imwrite(output_path, inference[i, ...])
            update_paths.append(output_path)
        except Exception:
            # If writing the output fails, we do not want to mark the task as completed in the checkpoint index, to avoid data corruption.
            err_file = output_path.with_suffix(".err")
            err_file.touch(exist_ok=True)
            update_paths.append(err_file)

    update_checkpoint_batch(
        tasks=metadata,
        output_files=update_paths,
        model_metadata=model_metadata,
    )
