import numpy as np
import pandas as pd

from image_ablation_analysis.regression.nested_regression import (
    BootstrapConfig,
    ColumnSpec,
    bootstrap_nested_regression,
)


def test_bootstrap_nested_regression_numeric_x2_returns_expected_columns():
    n = 40
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.linspace(0.0, 2.0, n)
    noise = np.random.default_rng(0).normal(scale=0.01, size=n)
    y = 1.0 + 2.0 * x1 + 0.5 * x2 + noise

    df = pd.DataFrame(
        {
            "group": ["g1"] * n,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )

    colspec = ColumnSpec(group_cols=("group",), y="y", x1="x1", x2="x2")
    cfg = BootstrapConfig(
        n_boot=2,
        min_group_size=2,
        use_tqdm=False,
        random_state=1,
    )

    result = bootstrap_nested_regression(df, colspec, cfg)

    expected_cols = {
        "group",
        "boot_idx",
        "beta_x1_restricted",
        "beta_x1_full",
        "beta_x2",
        "r2_restricted",
        "r2_full",
        "delta_r2",
        "partial_r2_x2",
        "cohen_f2_x2",
        "n_boot_rows",
        "n_group_rows",
    }

    assert len(result) == 2
    assert expected_cols.issubset(set(result.columns))
    assert result["beta_x2"].notna().all()


def test_bootstrap_nested_regression_categorical_x2_keeps_beta_x2_nan():
    n = 40
    x1 = np.linspace(-1.0, 1.0, n)
    x2 = np.array(["A", "B"] * (n // 2))
    x2_effect = np.where(x2 == "B", 0.8, 0.0)
    noise = np.random.default_rng(2).normal(scale=0.01, size=n)
    y = 1.0 + 1.5 * x1 + x2_effect + noise

    df = pd.DataFrame(
        {
            "group": ["g1"] * n,
            "y": y,
            "x1": x1,
            "x2": x2,
        }
    )

    colspec = ColumnSpec(
        group_cols=("group",),
        y="y",
        x1="x1",
        x2="x2",
        x2_categorical=True,
    )
    cfg = BootstrapConfig(
        n_boot=2,
        min_group_size=2,
        use_tqdm=False,
        random_state=3,
        standardize=True,
    )

    result = bootstrap_nested_regression(df, colspec, cfg)

    assert len(result) == 2
    assert result["beta_x2"].isna().all()
    assert "partial_r2_x2" in result.columns
