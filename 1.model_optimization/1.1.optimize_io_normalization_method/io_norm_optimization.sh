#!/bin/bash

# Initialize Conda for Bash
conda init bash
# Activate the Conda environment
conda activate alsf_analysis

# Convert Jupyter notebooks to Python scripts in the nbconverted folder
jupyter nbconvert --to script --output-dir=nbconverted/ *.ipynb

# Run the Python scripts in order
python nbconverted/1.1.1.optimize_fnet_io_norm.py