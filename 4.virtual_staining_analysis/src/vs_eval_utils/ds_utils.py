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
    target_channel_keys: Optional[list[str]] = None,
) -> CropCellImageDataset:
    """
    Helper invoking the virtual_stain_flow dataset initialization steps
        to prepare a CropCellImageDataset from a loaddata DataFrame.

    :param loaddata_df: DataFrame containing the loaddata information for 
        the samples to be included in the dataset
    :param sc_features: DataFrame containing the single-cell features 
        for the samples. The notebook initializes a global sc_features DataFrame
        that is used as default argument here. 
    :return: Initialized CropCellImageDataset ready for inference
    """
    cp_ids = CPLoadDataImageDataset(
            loaddata=loaddata_df,
            sc_feature=sc_features,
            pil_image_mode="I;16",
        )
    crop_ds = CropCellImageDataset.from_dataset(
        cp_ids,
        patch_size=256,
        object_coord_x_field="Metadata_Cells_Location_Center_X",
        object_coord_y_field="Metadata_Cells_Location_Center_Y",
        fov=(1080, 1080),
    )
    crop_ds.transform = MaxScaleNormalize(
        p=1,
        normalization_factor=2**16 - 1,  # normalize to [0, 1]
    )
    crop_ds.input_channel_keys = ["OrigBrightfield"]
    # Any target channel is fine for eval as we only need input BF
    crop_ds.target_channel_keys = ["OrigBrightfield"] \
        if target_channel_keys is None else target_channel_keys

    return crop_ds
