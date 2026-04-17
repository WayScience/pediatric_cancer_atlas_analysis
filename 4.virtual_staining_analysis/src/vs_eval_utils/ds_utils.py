"""
Utility functions for dataset preparation and processing for virtual staining evaluation.
"""

from typing import Optional

import pandas as pd

from virtual_stain_flow.datasets.cp_loaddata_dataset import CPLoadDataImageDataset
from virtual_stain_flow.datasets.crop_cell_dataset import CropCellImageDataset
from virtual_stain_flow.transforms.normalizations import MaxScaleNormalize


def prep_crop_dataset(
    loaddata_df: pd.DataFrame,
    sc_features: pd.DataFrame,
    input_channel_keys: list[str] | str,
    patch_size: int = 256,
    object_coord_x_field: str = "Metadata_Cells_Location_Center_X",
    object_coord_y_field: str = "Metadata_Cells_Location_Center_Y",
    fov: tuple[int, int] = (1080, 1080),
    normalization_factor: int = 2**16 - 1,
) -> CropCellImageDataset:
    """
    Helper invoking the virtual_stain_flow dataset initialization steps
        to prepare a CropCellImageDataset from a loaddata DataFrame.
        Largely a wrapper around the CPLoadDataImageDataset and CropCellImageDataset
        initialization steps, with some additional configuration for normalization and channel keys.

    :param loaddata_df: DataFrame containing the loaddata information for 
        the samples to be included in the dataset
    :param sc_features: DataFrame containing the single-cell features 
        for the samples, resulting from Cell Profiler analysis. 
        The only information being used in this function and downstream in
        crop dataset generation is some form of cell/object coordinates which
        is configured by object_coord_x_field and object_coord_y_field.  
    :param input_channel_keys: List of channel keys to be used as input for the dataset.
        The only relevant channel configuration for inference purposes, 
        as the specified channel(s) will be provided as input to the model.
        Should match the model's expected input channels.
    :param patch_size: Size of the image patches to be extracted around each cell. 
        Default is 256.
    :param object_coord_x_field: Name of the field in the loaddata DataFrame that contains 
        the x-coordinates of the cell centers. Default is "Metadata_Cells_Location_Center_X".
    :param object_coord_y_field: Name of the field in the loaddata DataFrame that contains 
        the y-coordinates of the cell centers. Default is "Metadata_Cells_Location_Center_Y".
    :param fov: Tuple specifying the field of view (FOV) dimensions of the original images. 
        Default is (1080, 1080).
    :param normalization_factor: Factor used for normalizing the pixel values. 
        Default is 65535 (2^16 - 1), which is common for 16-bit images.
    :return: Initialized CropCellImageDataset ready for inference
    """
    cp_ids = CPLoadDataImageDataset(
            loaddata=loaddata_df,
            sc_feature=sc_features,
            pil_image_mode="I;16",
        )
    crop_ds = CropCellImageDataset.from_dataset(
        cp_ids,
        patch_size=patch_size,
        object_coord_x_field=object_coord_x_field,
        object_coord_y_field=object_coord_y_field,
        fov=fov,
    )
    crop_ds.transform = MaxScaleNormalize(
        p=1,
        normalization_factor=normalization_factor,
    )
    input_channel_keys = input_channel_keys \
        if isinstance(input_channel_keys, list) else [input_channel_keys]
    crop_ds.input_channel_keys = [input_channel_keys]
    crop_ds.target_channel_keys = [input_channel_keys]

    return crop_ds
