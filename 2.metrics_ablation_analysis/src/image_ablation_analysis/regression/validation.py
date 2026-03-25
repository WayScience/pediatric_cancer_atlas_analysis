"""
validation.py

Validation helpers for nested regression inputs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .nested_regression import ColumnSpec


def _coerce_and_filter(
    df: pd.DataFrame,
    colspec: "ColumnSpec",
    drop_na: bool = True,
) -> pd.DataFrame:
    """
    Validate and sanitize DataFrame inputs before regression fitting.
    """
    required = [*colspec.group_cols, colspec.y, colspec.x1, colspec.x2]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {missing}")

    numeric = [colspec.y, colspec.x1]
    if not colspec.x2_categorical:
        numeric.append(colspec.x2)

    out = df.copy()

    for c in numeric:
        if not pd.api.types.is_numeric_dtype(out[c]):
            out[c] = pd.to_numeric(out[c], errors="coerce")

    if drop_na:
        subset = list(numeric)
        if colspec.x2_categorical:
            subset.append(colspec.x2)
        out = out.dropna(subset=subset)

    for c in numeric:
        out = out[np.isfinite(out[c])]

    return out
