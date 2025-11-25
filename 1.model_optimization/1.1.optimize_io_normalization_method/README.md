# This folder contains code for determining the optimal combination of input and target normalization method to use with FNet. 
This is achieved with a running optuna trials sampling randomly from one of the three possible image normalization methods (independently for both the input and target normalization): 
1. min max normalization with the max pixel intensity of 16 bit image as the normalization factor
2. z score normalization on a per image/patch basis
3. raw 8 bit transformation from 16 bit image

Our optimizations are primarily dependent on the data processed in `0.data_preprocessing`.

For the analysis in this folder to work, the `config.yml` file at the repo home will need to be configured for the following fields: 
1. `pediatric_cancer_atlas_profiling_path`: The path to the pediatric_cancer_atlas_profiling repo. 
2. `sc_features_path`: The path to the single cell level features parquet files.
3. `software_path`: The path to the cloned virtual_stain_flow_software (branch `dev-0.1`) as the software is currently in active development and is not installable. 

## The FNet input and output (target) normalization optimization can be performed by running the following notebook:
1.1.optimize_io_normalization_method/1.1.1.optimize_fnet_io_norm.ipynb

## ALternatively, run
```bash
# Ensure `config.yml` at the root of the repository is configured
source io_norm_optimization.sh
```