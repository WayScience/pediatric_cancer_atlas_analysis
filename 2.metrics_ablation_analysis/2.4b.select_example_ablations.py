#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pathlib
import yaml

import polars as pl


# In[2]:


module_config_path = pathlib.Path("..") / '2.metrics_ablation_analysis' / 'config.yml'
if not module_config_path.exists():
    raise FileNotFoundError(f"Module config file not found: {module_config_path}")
config = yaml.safe_load(module_config_path.read_text())

plot_output_dir = pathlib.Path(".") / "plots" / "fig_panels"
plot_output_dir.mkdir(parents=True, exist_ok=True)

abl_root = pathlib.Path(config['ablation_output_path']).resolve(strict=True)

abl_indices = list((abl_root / "ablated_index").glob("aug_index_*.parquet"))
if not abl_indices:
    raise FileNotFoundError(f"No ablated index files found in {abl_root / 'ablated_index'}.")
else:
    print(f"Found {len(abl_indices)} ablated index files in {abl_root / 'ablated_index'}.")


# In[3]:


df = pl.concat([pl.read_parquet(p) for p in abl_indices], how="vertical").to_pandas()
df.loc[:, "ablation_type"] = df.loc[:, "config_id"].apply(lambda x: x.split(":")[1])
print(f"Combined DataFrame shape: {df.shape}")
df.head()


# In[4]:


all_ablation_types = df['ablation_type'].unique()
print(all_ablation_types)


# In[5]:


import ast
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class AblationComboExplorer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.meta = self._build_meta(df)

    @staticmethod
    def _to_list(x):
        if isinstance(x, (list, tuple, np.ndarray)):
            return list(x)
        if pd.isna(x):
            return []
        if isinstance(x, str):
            s = x.strip()
            try:
                parsed = ast.literal_eval(s)
                if isinstance(parsed, (list, tuple, np.ndarray)):
                    return list(parsed)
                return [parsed]
            except Exception:
                return [s]
        return [x]

    @classmethod
    def _first_or_nan(cls, x):
        vals = cls._to_list(x)
        return vals[0] if len(vals) else np.nan

    def _build_meta(self, df: pd.DataFrame) -> pd.DataFrame:
        meta = df.loc[
            :, ["ablation_type", "param_swept", "param_values", "original_abs_path", "aug_abs_path"]
        ].copy()

        meta["param_swept_name"] = meta["param_swept"].map(self._first_or_nan)
        meta["param_value"] = meta["param_values"].map(self._first_or_nan)
        meta["param_value_num"] = pd.to_numeric(meta["param_value"], errors="coerce")

        meta = meta.dropna(subset=["ablation_type", "param_swept_name", "original_abs_path", "aug_abs_path"])
        return meta

    def enumerate_combinations(self) -> pd.DataFrame:
        # Exact combinations requested: ablation_type, param_swept, aug_abs_path
        return (
            self.meta.loc[:, ["ablation_type", "param_swept_name", "aug_abs_path"]]
            .drop_duplicates()
            .rename(columns={"param_swept_name": "param_swept"})
            .reset_index(drop=True)
        )

    def _candidate_groups(self, ablation_type: str, param_swept: str):
        m = self.meta[
            (self.meta["ablation_type"] == ablation_type)
            & (self.meta["param_swept_name"] == param_swept)
        ]
        grouped = (
            m.groupby(["ablation_type", "param_swept_name", "original_abs_path"], sort=False)
            .size()
            .reset_index(name="n")
        )
        return grouped

    def plot_random_combo(
        self,
        ablation_type: str,
        param_swept: str,
        seed: int | None = None,
        sample_n_levels: int | None = None,
        grayscale_limits_from: Literal["original", "all"] = "original",
        crop_center: int | None = None,
    ):
        if grayscale_limits_from not in {"original", "all"}:
            raise ValueError("grayscale_limits_from must be one of {'original', 'all'}.")

        groups = self._candidate_groups(ablation_type, param_swept)
        if groups.empty:
            raise ValueError(f"No groups found for ablation_type='{ablation_type}', param_swept='{param_swept}'.")

        rng = np.random.default_rng(seed)
        chosen = groups.iloc[rng.integers(0, len(groups))]
        original_path = chosen["original_abs_path"]

        subset = self.meta[
            (self.meta["ablation_type"] == ablation_type)
            & (self.meta["param_swept_name"] == param_swept)
            & (self.meta["original_abs_path"] == original_path)
        ].copy()

        subset = subset.sort_values(["param_value_num", "param_value"], na_position="last")
        subset = subset.drop_duplicates(subset=["param_value", "aug_abs_path"], keep="first").reset_index(drop=True)

        # Optionally sample levels as evenly spaced as possible
        if sample_n_levels is not None:
            if sample_n_levels <= 0:
                raise ValueError("sample_n_levels must be a positive integer or None.")
            total_levels = len(subset)
            if total_levels > sample_n_levels:
                idx = np.linspace(0, total_levels - 1, num=sample_n_levels)
                idx = np.unique(np.round(idx).astype(int))
                subset = subset.iloc[idx].reset_index(drop=True)

        param_values = subset["param_value"].tolist()
        aug_paths = subset["aug_abs_path"].tolist()

        panels = [("original", original_path)] + [(str(v), p) for v, p in zip(param_values, aug_paths)]

        # Preload images so grayscale limits can be computed once and reused
        loaded = []
        for label, p in panels:
            img_path = Path(p)
            if not img_path.exists():
                loaded.append((label, p, None))
                continue
            loaded.append((label, p, plt.imread(img_path)))

        # Compute grayscale limits
        def _minmax(img_list):
            mins, maxs = [], []
            for img in img_list:
                if img is None or img.ndim != 2:
                    continue
                mins.append(np.nanmin(img))
                maxs.append(np.nanmax(img))
            if not mins:
                return None, None
            return float(np.min(mins)), float(np.max(maxs))

        def _compute_center_crop_bb(im_h, im_w, crop_h, crop_w):
            center_y, center_x = im_h // 2, im_w // 2
            half_crop_h, half_crop_w = crop_h // 2, crop_w // 2
            y1 = max(center_y - half_crop_h, 0)
            y2 = min(center_y + half_crop_h, im_h)
            x1 = max(center_x - half_crop_w, 0)
            x2 = min(center_x + half_crop_w, im_w)
            return y1, y2, x1, x2

        if grayscale_limits_from == "original":
            original_img = loaded[0][2] if loaded else None
            gray_vmin, gray_vmax = _minmax([original_img])
        else:  # "all"
            gray_vmin, gray_vmax = _minmax([img for _, _, img in loaded])

        fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
        if len(panels) == 1:
            axes = [axes]

        for ax, (label, p, img) in zip(axes, loaded):

            if img is None:
                ax.axis("off")
                ax.set_title(f"{label}\n[missing]")
                continue
            
            if crop_center is not None and img.ndim == 2:
                im_h, im_w = img.shape
                y1, y2, x1, x2 = _compute_center_crop_bb(im_h, im_w, crop_center, crop_center)
                img = img[y1:y2, x1:x2]

            if img.ndim == 2:
                ax.imshow(img, cmap="gray", vmin=gray_vmin, vmax=gray_vmax)
            elif img.ndim == 3:
                ax.imshow(img[0], cmap="gray", vmin=gray_vmin, vmax=gray_vmax)
            else:
                try:
                    ax.imshow(img)
                except Exception as e:
                    print(f"Error displaying image {label}: {e}")
            
            if isinstance(label, str):
                if label == "original":
                    pass
                else:
                    label = f"{param_swept}={float(label):.3g}"
            elif isinstance(label, (int, np.integer)):
                label = f"{param_swept}={label}"
            else:
                label = f"{param_swept}={str(label)}"

            ax.set_title(label)
            ax.axis("off")

        fig.suptitle(ablation_type, fontsize=12)
        fig.subplots_adjust(wspace=0.1)
        plt.tight_layout()
        plt.show()

        return {
            "fig": fig,
            "ablation_type": ablation_type,
            "param_swept": param_swept,
            "original_abs_path": original_path,
            "param_values_sorted": param_values,
            "aug_abs_paths_sorted": aug_paths,
            "sample_n_levels": sample_n_levels,
            "grayscale_limits_from": grayscale_limits_from,
            "gray_vmin": gray_vmin,
            "gray_vmax": gray_vmax,
        }

    def endpoint(self, ablation_type: str, param_swept: str):
        # Returns a callable "endpoint" for repeated random sampling/plotting
        def _run(seed: int | None = None):
            return self.plot_random_combo(ablation_type=ablation_type, param_swept=param_swept, seed=seed)

        return _run


# In[6]:


explorer = AblationComboExplorer(df)
combo_table = explorer.enumerate_combinations()
ablation_param_combos = combo_table[["ablation_type", "param_swept"]].drop_duplicates().reset_index(drop=True)
ablation_param_combos


# In[7]:


for seed in [1, 42, 123]:

    shared_args = {
        "seed": seed,
        "sample_n_levels": 3,
        "grayscale_limits_from": "original",
        "crop_center": 256,
    }

    for ablation_type, param_swept in ablation_param_combos.itertuples(index=False):
        plot_dict = explorer.plot_random_combo(ablation_type=ablation_type, param_swept=param_swept, **shared_args)
        # save figure
        plot_dict["fig"].savefig(
            plot_output_dir / f"{ablation_type}_{param_swept}_seed={shared_args['seed']}.pdf",
            dpi=300,
            bbox_inches="tight"
        )

