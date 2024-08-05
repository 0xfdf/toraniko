"""Test functions in the utils module."""

import pytest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal

from toraniko.utils import fill_features

###
# `fill_features`
###


@pytest.fixture
def sample_df():
    return pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
            "group": ["A", "A", "A", "B", "B"],
            "feature1": [1.0, np.nan, 3.0, np.inf, 5.0],
            "feature2": [np.nan, 2.0, np.nan, 4.0, np.nan],
        }
    )


@pytest.mark.parametrize("lazy", [True, False])
def test_fill_features(sample_df, lazy):
    if lazy:
        inp = sample_df.lazy()
    else:
        inp = sample_df
    result = (
        fill_features(inp, features=("feature1", "feature2"), sort_col="date", over_col="group").sort("group").collect()
    )

    expected = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
            "group": ["A", "A", "A", "B", "B"],
            "feature1": [1.0, 1.0, 3.0, None, 5.0],
            "feature2": [None, 2.0, 2.0, 4.0, 4.0],
        }
    )

    pl.testing.assert_frame_equal(result, expected)


def test_fill_features_string_nan(sample_df):
    df_with_string_nan = sample_df.with_columns(pl.col("feature1").cast(str))
    df_with_string_nan = df_with_string_nan.with_columns(
        pl.when(pl.col("feature1") == "nan").then("NaN").otherwise(pl.col("feature1"))
    )

    result = fill_features(df_with_string_nan, features=("feature1", "feature2"), sort_col="date", over_col="group")
    result = result.collect()

    expected = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05"],
            "group": ["A", "A", "A", "B", "B"],
            "feature1": [1.0, 1.0, 3.0, None, 5.0],
            "feature2": [None, 2.0, 2.0, 4.0, 4.0],
        }
    )

    pl.testing.assert_frame_equal(result, expected)


def test_fill_features_invalid_input():
    invalid_df = {"not": "a dataframe"}
    with pytest.raises(TypeError):
        fill_features(invalid_df, features=("feature1",), sort_col="date", over_col="group")


def test_fill_features_missing_column(sample_df):
    with pytest.raises(ValueError):
        fill_features(sample_df, features=("non_existent_feature",), sort_col="date", over_col="group")


def test_fill_features_all_null_column(sample_df):
    df_with_null_column = sample_df.with_columns(pl.lit(None).alias("null_feature"))
    result = fill_features(df_with_null_column, features=("null_feature",), sort_col="date", over_col="group")
    result = result.collect()

    assert all(result["null_feature"].is_null())


def test_fill_features_multiple_groups(sample_df):
    multi_group_df = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
            "group": ["A", "A", "A", "B", "B", "C"],
            "feature1": [1.0, np.nan, 3.0, np.inf, 5.0, np.nan],
            "feature2": [np.nan, 2.0, np.nan, 4.0, np.nan, 6.0],
        }
    )

    result = fill_features(multi_group_df, features=("feature1", "feature2"), sort_col="date", over_col="group")
    result = result.collect()

    expected = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
            "group": ["A", "A", "A", "B", "B", "C"],
            "feature1": [1.0, 1.0, 3.0, None, 5.0, None],
            "feature2": [None, 2.0, 2.0, 4.0, 4.0, 6.0],
        }
    )

    pl.testing.assert_frame_equal(result, expected)


def test_fill_features_different_sort_order(sample_df):
    result = fill_features(sample_df, features=("feature1", "feature2"), sort_col="date", over_col="group")
    result = result.sort("date", descending=True).collect()

    expected = pl.DataFrame(
        {
            "date": ["2023-01-05", "2023-01-04", "2023-01-03", "2023-01-02", "2023-01-01"],
            "group": ["B", "B", "A", "A", "A"],
            "feature1": [5.0, None, 3.0, 1.0, 1.0],
            "feature2": [4.0, 4.0, 2.0, 2.0, None],
        }
    )

    pl.testing.assert_frame_equal(result, expected)
