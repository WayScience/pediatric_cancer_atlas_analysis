# Preliminary exploratory analysis

In this module, we perform EDA (exploratory data analysis) using the pilot morphology data that we have extracted.
Current analysis includes:

1. [Single-cell UMAPs](./0.UMAP_embeddings.ipynb): We generate UMAP embeddings with ~10,000 cells per plate by performing equal-size stratified sampling across cell line and seeding density.

We are using a Linux-based machine running Pop_OS! LTS 22.04 with an AMD Ryzen 7 3700X 8-Core Processor with 16 CPUs and 125 GB of MEM.

## Perform EDA

You can perform the EDA of morphology features using the command below:

```bash
# Make sure you are in the 4.preliminary_results directory
source run_eda.sh
```
