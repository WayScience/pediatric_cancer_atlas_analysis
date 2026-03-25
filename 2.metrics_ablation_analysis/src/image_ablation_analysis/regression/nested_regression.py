"""
nested_regression.py

Nested regression analysis + bootstrap for image ablation analysis.
Only supports single restricted parameter and single additional parameter,
    this decision was made for simplicity of how results can be reported
    and visualized in the context of image ablation analysis.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from sklearn.preprocessing import StandardScaler

from .validation import _coerce_and_filter


@dataclass
class BootstrapConfig:
    """
    Basic configuration for bootstrap nested regression.

    - n_boot: number of bootstrap replicates per group.
    - sample_frac: fraction of group size to sample in each bootstrap replicate.
    - replace: whether to sample with replacement.
    - standardize: whether to z-score specified columns within each bootstrap replicate.
    - random_state: optional random seed for reproducibility.
    - use_tqdm: whether to use tqdm progress bars.
    - drop_na: whether to drop rows with NA in numeric columns before analysis.
    - min_group_size: minimum number of rows in a group to perform bootstrap.
    - max_per_group: optional maximum number of rows per group (subsample if exceeded).
    - robust_cov: optional string specifying robust covariance type for statsmodels.
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
    Stores column names and lightweight modeling options.

    - group_cols: columns defining a group (e.g. metric_name, ablation, cell_line, channel, ...)
    - y: outcome column (e.g. "metric_value")
    - x1: predictor in restricted model (e.g. "config")
    - x2: additional predictor in full model (e.g. "confluence" or "cell_line")
    - x2_categorical: whether x2 should be treated as a categorical factor in the
      full model via patsy/statsmodels C(x2)
    - standardize_cols: which columns to z-score within each bootstrap.
      If None, defaults to (x1, x2) when x2 is numeric, or just (x1) when x2 is categorical.
    """
    group_cols: Tuple[str, ...]
    y: str
    x1: str
    x2: str
    x2_categorical: bool = False
    standardize_cols: Optional[Tuple[str, ...]] = None


"""
Helper modules for nested regression analysis.
"""


def _std_cols(colspec: ColumnSpec) -> Tuple[str, ...]:
    if colspec.standardize_cols is not None:
        cols = colspec.standardize_cols
        if isinstance(cols, str):
            return (cols,)
        return tuple(cols)
    if colspec.x2_categorical:
        return (colspec.x1,)
    return (colspec.x1, colspec.x2)


def _x2_term(colspec: ColumnSpec) -> str:
    return f"C({colspec.x2})" if colspec.x2_categorical else colspec.x2


def _fit_ols_formula(df: pd.DataFrame,
                     formula: str,
                     robust_cov: Optional[str] = None):
    """
    Fit OLS via statsmodels formula; optionally attach robust covariance.
    Function abstracting the smallest unit of regression that will be
        ran multiple times per bootstrap.

    :param df: DataFrame with data.
    :param formula: Patsy formula string for OLS.
    :param robust_cov: Optional robust covariance type for statsmodels.
    :return: Fitted regression results.
    """
    model = smf.ols(formula, data=df)
    res = model.fit()
    if robust_cov:
        res = res.get_robustcov_results(cov_type=robust_cov)
    return res


def _compute_effect_sizes(res_re, res_fu) -> Dict[str, float]:
    r2_re = float(getattr(res_re, "rsquared", np.nan))
    r2_fu = float(getattr(res_fu, "rsquared", np.nan))
    ssr_re = float(getattr(res_re, "ssr", np.nan))
    ssr_fu = float(getattr(res_fu, "ssr", np.nan))

    if np.isfinite(r2_re) and np.isfinite(r2_fu):
        delta_r2 = r2_fu - r2_re
    else:
        delta_r2 = np.nan

    if np.isfinite(r2_re) and np.isfinite(r2_fu) and (1 - r2_re) > 0:
        partial_r2_x2 = (r2_fu - r2_re) / (1 - r2_re)
    elif np.isfinite(ssr_re) and np.isfinite(ssr_fu) and ssr_re > 0:
        partial_r2_x2 = (ssr_re - ssr_fu) / ssr_re
    else:
        partial_r2_x2 = np.nan

    if np.isfinite(r2_fu) and (1 - r2_fu) > 0:
        f2_x2 = (r2_fu - r2_re) / (1 - r2_fu)
    else:
        f2_x2 = np.nan

    return {
        "r2_restricted": r2_re,
        "r2_full": r2_fu,
        "delta_r2": delta_r2,
        "partial_r2_x2": partial_r2_x2,
        "cohen_f2_x2": f2_x2,
    }


def _one_bootstrap(
    df_group: pd.DataFrame,
    cfg: BootstrapConfig,
    colspec: ColumnSpec,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    """
    Run a single bootstrap replicate for one group defined by colspec.group_cols.
    Returns dict of betas, R²s, ΔR², partial R², f², and bookkeeping.

    :param df_group: DataFrame for one group.
    :param cfg: Bootstrap configuration.
    :param colspec: Column specification.
    :param rng: Numpy random generator.
    :return: Dict with regression results and effect size metrics.
    """
    n_group = len(df_group)
    bsize = max(2, int(round(cfg.sample_frac * n_group))) if cfg.sample_frac else n_group
    idx = rng.choice(df_group.index.to_numpy(), size=bsize, replace=cfg.replace)
    boot = df_group.loc[idx].copy()

    if cfg.standardize:
        cols = list(_std_cols(colspec))
        scaler = StandardScaler()
        boot.loc[:, cols] = scaler.fit_transform(boot.loc[:, cols])

    y, x1, x2 = colspec.y, colspec.x1, colspec.x2
    x2_term = _x2_term(colspec)
    formula_re = f"{y} ~ {x1}"
    formula_fu = f"{y} ~ {x1} + {x2_term}"

    res_re = _fit_ols_formula(boot, formula_re, cfg.robust_cov)
    res_fu = _fit_ols_formula(boot, formula_fu, cfg.robust_cov)

    beta_x1_re = res_re.params.get(x1, np.nan)
    beta_x1_fu = res_fu.params.get(x1, np.nan)
    beta_x2 = np.nan if colspec.x2_categorical else res_fu.params.get(x2, np.nan)
    effects = _compute_effect_sizes(res_re, res_fu)

    return {
        "beta_x1_restricted": beta_x1_re,
        "beta_x1_full": beta_x1_fu,
        "beta_x2": beta_x2,
        **effects,
        "n_boot_rows": bsize,
        "n_group_rows": n_group,
    }


def bootstrap_nested_regression(
    df: pd.DataFrame,
    colspec: ColumnSpec,
    cfg: Optional[BootstrapConfig] = None,
) -> pd.DataFrame:
    """
    Top level function to run bootstrap nested regression for 
        arbitrary grouping and column names.

    :param df: Input DataFrame.
    :param colspec: ColumnSpec defining required and numeric columns.
    :param cfg: Bootstrap configuration. If None, default config is used.
    :return: DataFrame with bootstrap regression results.
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
