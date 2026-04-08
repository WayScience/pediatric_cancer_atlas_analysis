#!/usr/bin/env python
# coding: utf-8

# # Explores ablation through generating visualizations comparing different ablations against the untransformed raw 

# In[1]:


import pathlib
import yaml

import polars as pl

from image_ablation_analysis.ablation_explorer import AblationExplorer


# ## Pathing

# In[2]:


# config for raw ablation image output
module_config_path = pathlib.Path("..") / '2.metrics_ablation_analysis' / 'config.yml'
if not module_config_path.exists():
    raise FileNotFoundError(f"Module config file not found: {module_config_path}")
config = yaml.safe_load(module_config_path.read_text())

# visualization output
plot_output_dir = pathlib.Path(".") / "plots" / "fig_panels"
plot_output_dir.mkdir(parents=True, exist_ok=True)

# ablation index files
abl_root = pathlib.Path(config['ablation_output_path']).resolve(strict=True)

abl_indices = list((abl_root / "ablated_index").glob("aug_index_*.parquet"))
if not abl_indices:
    raise FileNotFoundError(f"No ablated index files found in {abl_root / 'ablated_index'}.")
else:
    print(f"Found {len(abl_indices)} ablated index files in {abl_root / 'ablated_index'}.")


# In[3]:


# helper class to sample random images and show ablation vs raw
explorer = AblationExplorer(
    abl_index_path=abl_root
)

# show all found ablation type and param combinations
combo_table = explorer.enumerate_combinations()
ablation_param_combos = combo_table[["ablation_type", "param_swept"]].drop_duplicates().reset_index(drop=True)
ablation_param_combos


# ## Loop over a number of random seeds and produce visualizations

# In[4]:


for seed in [1, 42, 123]:

    shared_args = {
        "seed": seed,
        "sample_n_levels": 3,
        "grayscale_limits_from": "original",
        # some level of cropping helps with the visualization of the ablation
        # things like gaussian noise and blur can be very hard to see in full fov
        "crop_center": 256, 
    }

    for ablation_type, param_swept in ablation_param_combos.itertuples(index=False):
        
        # plot and save random plot of ablation/param combo
        plot_dict = explorer.plot_random_combo(
            ablation_type=ablation_type, param_swept=param_swept, **shared_args
        )
        # save figure
        plot_dict["fig"].savefig(
            plot_output_dir / f"{ablation_type}_{param_swept}_seed={shared_args['seed']}.pdf",
            dpi=300,
            bbox_inches="tight"
        )

        # iterate over all 5 levels of U2-OS density and plot and save

        for seeding_density in [1000, 2000, 4000, 8000, 12000]:

            plot_dict = explorer.plot_random_combo(
                ablation_type=ablation_type, 
                param_swept=param_swept, 
                additional_filter={
                    "cell_line": "U2-OS",
                    "seeding_density": seeding_density
                },
                **shared_args
            )
            plot_dict["fig"].savefig(
                plot_output_dir / f"{ablation_type}_{param_swept}_U2OS_density={seeding_density}_seed={shared_args['seed']}.pdf",
                dpi=300,
                bbox_inches="tight"
            )

