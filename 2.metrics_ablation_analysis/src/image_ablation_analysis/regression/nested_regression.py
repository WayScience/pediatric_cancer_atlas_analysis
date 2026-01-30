"""
nested_regression.py

Nested regression analysis + bootstrap for image ablation analysis.
Only supports single restricted parameter and single additional parameter,
    this decision was made for simplicity of how results can be reported
    and visualized in the context of image ablation analysis.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any, Sequence

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass
class BootstrapConfig:
    """
    Basic configuration for bootstrap nested regression.
    """
    n_boot: int = 300

    # bootstrap size relative to group size
    sample_frac: float = 1.0
    replace: bool = True

    # z-score config
    standardize: bool = False

    random_state: Optional[int] = 42
    use_tqdm: bool = True
    drop_na: bool = True

    # per-group filters
    min_group_size: int = 25
    max_per_group: Optional[int] = None

    # e.g. "HC3", "HC0", etc., or None
    robust_cov: Optional[str] = None


@dataclass
class ColumnSpec:
    """
    Generic column specification for the nested regression.

    - group_cols: columns defining a group (e.g. metric_name, ablation, cell_line, channel, ...)
    - y: outcome column (e.g. "metric_value")
    - x1: predictor in restricted model (e.g. "config")
    - x2: additional predictor in full model (e.g. "confluence")
    - standardize_cols: which columns to z-score within each bootstrap.
      If None, defaults to (x1, x2).
    """
    group_cols: Tuple[str, ...]
    y: str
    x1: str
    x2: str
    standardize_cols: Optional[Tuple[str, ...]] = None

    @property
    def required_cols(self) -> Tuple[str, ...]:
        return (*self.group_cols, self.y, self.x1, self.x2)

    @property
    def numeric_cols(self) -> Tuple[str, ...]:
        return (self.y, self.x1, self.x2)

    @property
    def std_cols(self) -> Tuple[str, ...]:
        return self.standardize_cols or (self.x1, self.x2)


"""
Helper modules for nested regression analysis.
"""
def _coerce_and_filter(
    df: pd.DataFrame,
    colspec: ColumnSpec,
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Ensure required columns exist and numeric columns are numeric.
    """
    missing = [c for c in colspec.required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    # Coerce numeric columns
    for c in colspec.numeric_cols:
        if not pd.api.types.is_numeric_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if drop_na:
        df = df.dropna(subset=list(colspec.numeric_cols))

    # Remove infs
    for c in colspec.numeric_cols:
        df = df[np.isfinite(df[c])]

    return df


def _standardize_in_place(gdf: pd.DataFrame,
                          cols: Sequence[str]) -> pd.DataFrame:
    gdf = gdf.copy()
    for c in cols:
        s = gdf[c].to_numpy()
        mu, sd = np.nanmean(s), np.nanstd(s, ddof=0)
        if sd > 0:
            gdf[c] = (s - mu) / sd
        # else leave as-is (all-constant)
    return gdf


def _fit_ols_formula(df: pd.DataFrame,
                     formula: str,
                     robust_cov: Optional[str] = None):
    """
    Fit OLS via statsmodels formula; optionally attach robust covariance.
    """
    model = smf.ols(formula, data=df)
    res = model.fit()
    if robust_cov:
        res = res.get_robustcov_results(cov_type=robust_cov)
    return res


def _one_bootstrap(
    df_group: pd.DataFrame,
    cfg: BootstrapConfig,
    colspec: ColumnSpec,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Run a single bootstrap replicate for one group defined by colspec.group_cols.
    Returns dict of betas, R²s, ΔR², partial R², f², and bookkeeping.
    """
    n = len(df_group)
    bsize = max(2, int(round(cfg.sample_frac * n))) if cfg.sample_frac else n
    idx = rng.choice(df_group.index.to_numpy(), size=bsize, replace=cfg.replace)
    boot = df_group.loc[idx]

    if cfg.standardize:
        boot = _standardize_in_place(boot, cols=colspec.std_cols)

    y, x1, x2 = colspec.y, colspec.x1, colspec.x2

    # e.g. "metric_value ~ config" and "metric_value ~ config + confluence"
    formula_re = f"{y} ~ {x1}"
    formula_fu = f"{y} ~ {x1} + {x2}"

    res_re = _fit_ols_formula(boot, formula_re, cfg.robust_cov)
    res_fu = _fit_ols_formula(boot, formula_fu, cfg.robust_cov)

    beta_x1_re = res_re.params.get(x1, np.nan)
    beta_x1_fu = res_fu.params.get(x1, np.nan)
    beta_x2    = res_fu.params.get(x2, np.nan)

    r2_re  = float(getattr(res_re, "rsquared", np.nan))
    r2_fu  = float(getattr(res_fu, "rsquared", np.nan))
    ssr_re = float(getattr(res_re, "ssr", np.nan))
    ssr_fu = float(getattr(res_fu, "ssr", np.nan))

    # Incremental R² (delta R²)
    if np.isfinite(r2_re) and np.isfinite(r2_fu):
        delta_r2 = r2_fu - r2_re
    else:
        delta_r2 = np.nan

    # Partial R² for x2
    if np.isfinite(r2_re) and np.isfinite(r2_fu) and (1 - r2_re) > 0:
        partial_r2_x2 = (r2_fu - r2_re) / (1 - r2_re)
    elif np.isfinite(ssr_re) and np.isfinite(ssr_fu) and ssr_re > 0:
        partial_r2_x2 = (ssr_re - ssr_fu) / ssr_re
    else:
        partial_r2_x2 = np.nan

    # Cohen’s f² for x2
    if np.isfinite(r2_fu) and (1 - r2_fu) > 0:
        f2_x2 = (r2_fu - r2_re) / (1 - r2_fu)
    else:
        f2_x2 = np.nan

    return {
        # keep generic names so the code is column-agnostic
        "beta_x1_restricted": beta_x1_re,
        "beta_x1_full": beta_x1_fu,
        "beta_x2": beta_x2,
        "r2_restricted": r2_re,
        "r2_full": r2_fu,
        "delta_r2": delta_r2,
        "partial_r2_x2": partial_r2_x2,
        "cohen_f2_x2": f2_x2,
        "n_boot_rows": bsize,
        "n_group_rows": n,
    }


def bootstrap_nested_regression(
    df: pd.DataFrame,
    colspec: ColumnSpec,
    cfg: Optional[BootstrapConfig] = None,
) -> pd.DataFrame:
    """
    Top level function to run bootstrap nested regression for 
        arbitrary grouping and column names.

    Parameters
    ----------
    df : DataFrame
        Long-format data.
    colspec : ColumnSpec
        Column configuration (grouping, y, x1, x2, standardize_cols).
    cfg : BootstrapConfig, optional
        Bootstrap configuration.

    Returns
    -------
    DataFrame
        One row per bootstrap replicate per group, with:
        - all group_cols
        - 'boot_idx'
        - regression / effect size metrics from _one_bootstrap.
    """
    if cfg is None:
        cfg = BootstrapConfig()

    df = _coerce_and_filter(df, colspec, drop_na=cfg.drop_na)

    if cfg.random_state is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(cfg.random_state)

    group_cols = list(colspec.group_cols)
    grouped = df.groupby(group_cols, sort=False)

    # optional tqdm
    if cfg.use_tqdm:
        try:
            from tqdm.auto import tqdm
            grouped_iter = tqdm(grouped, desc="Bootstrap groups")
        except Exception:
            grouped_iter = grouped
    else:
        grouped_iter = grouped

    records: list[Dict[str, Any]] = []

    for group_key, gdf in grouped_iter:
        # group_key is scalar if single group col, tuple otherwise
        if not isinstance(group_key, tuple):
            group_key = (group_key,)

        # enforce min/max group sizes
        n_group = len(gdf)
        if n_group < cfg.min_group_size:
            continue

        if cfg.max_per_group is not None and n_group > cfg.max_per_group:
            # subsample without replacement to max_per_group
            keep_idx = rng.choice(
                gdf.index.to_numpy(),
                size=cfg.max_per_group,
                replace=False,
            )
            gdf = gdf.loc[keep_idx]

        for b in range(cfg.n_boot):
            res = _one_bootstrap(gdf, cfg, colspec, rng)
            row = {col: val for col, val in zip(group_cols, group_key)}
            row["boot_idx"] = b
            row.update(res)
            records.append(row)

    return pd.DataFrame.from_records(records)
