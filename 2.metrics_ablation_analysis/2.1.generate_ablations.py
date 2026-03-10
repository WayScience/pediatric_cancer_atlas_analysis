"""
2.1. Generate ablated full images using various albumentation ablation sweeps.

Utilizes the AblationRunner class to manage image loading, ablation application,
and saving of ablated images.
Uses pre-defined ablation sweeps from sweeps.py (parameters that vary and
are fixed across sweep are determined there, but the variant parameter
value is still be adjusted here). Please refer to sweeps.py for details.
"""

import pathlib

import numpy as np

from image_ablation_analysis.nb_utils import find_git_root
from image_ablation_analysis.ablation_runner import AblationRunner
from image_ablation_analysis.hooks.normalization import BitDepthNormalizer
from image_ablation_analysis.sweeps import (
    grid_distort_sweep,
    gauss_noise_sweep,
    blur_sweep,
    erode_sweep,
    dilate_sweep,
    gamma_sweep
)

data_split_path = find_git_root() / '0.data_preprocessing' / 'data_split_loaddata'
if not data_split_path.exists() and not data_split_path.is_dir():
    raise FileNotFoundError(f"Data split path not found: {data_split_path}")

# Include all datasplits for model training for the ablation analysis as we 
# are not training models here but assessing the metric sensitivity, so use
# all available data we have
loaddata_csvs = [
    data_split_path / file for file in ["loaddata_train.csv", "loaddata_eval.csv"]
]

if not all([csv.exists() for csv in loaddata_csvs]):
    missing_files = [csv for csv in loaddata_csvs if not csv.exists()]
    raise FileNotFoundError(f"Loaddata CSV files not found: {missing_files}")
else:
    print(f"Loaddata CSV files found: {loaddata_csvs}")

# Initialize AblationRunner
# This object interacts with a local `images_root` of images and a corresponding
# index file for the images (currently the only supported index is loaddata csv),
# and will write ablated images to the specified `ablation_root` directory mirroring
# the structure of the `images_root` directory. 
runner = AblationRunner(
    images_root=pathlib.Path(
        "/mnt/data_nvme1/data/ALSF_pilot_data/"
    ),
    ablation_root=pathlib.Path(
        "/mnt/hdd20tb/alsf_ablation/"
    ),
    loaddata_csvs=loaddata_csvs,
    keep_meta_columns=None,
    skip_if_indexed=True
)

# Define a collection of ablation sweeps to perform
hook_kwargs = {
    "normalizer": BitDepthNormalizer(bit_depth=16),
    "return_original_dtype": True,
}
sweep_hooks = [
    # Non-uniform morpholoical changes to the image
    grid_distort_sweep(distort_limit_values=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0], **hook_kwargs),
    
    # gaussian noise and blur sweeps with roughly equidistant increments
    # the albumentation gaussian noise with std=x y times is approximately equivalent
    # to adding a one time gaussian noise with std=sqrt(y)*x
    gauss_noise_sweep(std_range_values=[np.sqrt(i) * 0.1 for i in [1, 2, 3, 4, 5, 6]], **hook_kwargs),
    # blur sweep handles sigma scaling internally, so we just provide number of iterations
    blur_sweep(its = [10, 20, 30, 40, 50, 60], sigma_base=0.8, **hook_kwargs),
    
    # uniform erosion and dilation sweeps
    erode_sweep(its = [1, 2, 3, 4, 5, 6], k=3, **hook_kwargs),
    dilate_sweep(its = [1, 2, 3, 4, 5, 6], k=3, **hook_kwargs),

    # brightness adjustments with gamma and not global scaling
    # roughly equidistant increments in log space
    # brightening
    gamma_sweep(gamma_limit_values=[y * 100 for y in list(np.geomspace(1.0, 3.0, 6))], **hook_kwargs),
    # darkening
    gamma_sweep(gamma_limit_values=[y * 100 for y in list(np.geomspace(0.3, 1.0, 6))], **hook_kwargs),
]

# Set-off the sweeps
for sweep_hook in sweep_hooks:
    try:
        runner.run(
            augment_hook=sweep_hook,
        )
    except Exception as e:
        print(f"Error occurred while running {sweep_hook}: {e}")
        continue
