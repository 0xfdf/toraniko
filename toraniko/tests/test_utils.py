"""Test functions in the utils module."""

import pytest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal

from toraniko.utils import fill_features, smooth_features, top_n_by_group

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


def test_fill_features_invalid_input(sample_df):
    with pytest.raises(ValueError):
        fill_features(sample_df, features=("non_existent_feature",), sort_col="date", over_col="group")
    with pytest.raises(TypeError):
        fill_features("not_a_dataframe", features=("feature1",), sort_col="date", over_col="group")


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


###
# `smooth_features`
###


@pytest.fixture
def sample_smooth_df():
    return pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
            "group": ["A", "A", "A", "B", "B", "B"],
            "feature1": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )


@pytest.mark.parametrize("lazy", [True, False])
def test_smooth_features(sample_smooth_df, lazy):
    if lazy:
        inp = sample_smooth_df.lazy()
    else:
        inp = sample_smooth_df
    result = (
        smooth_features(inp, features=("feature1", "feature2"), sort_col="date", over_col="group", window_size=2)
        .sort("group")
        .collect()
    )

    expected = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
            "group": ["A", "A", "A", "B", "B", "B"],
            "feature1": [None, 1.5, 2.5, None, 4.5, 5.5],
            "feature2": [None, 15.0, 25.0, None, 45.0, 55.0],
        }
    )

    pl.testing.assert_frame_equal(result, expected)


def test_smooth_features_larger_window(sample_smooth_df):
    result = smooth_features(
        sample_smooth_df, features=("feature1", "feature2"), sort_col="date", over_col="group", window_size=10
    )
    result = result.collect()

    expected = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
            "group": ["A", "A", "A", "B", "B", "B"],
            "feature1": [None, None, None, None, None, None],
            "feature2": [None, None, None, None, None, None],
        }
    ).with_columns(pl.col("feature1").cast(float).alias("feature1"), pl.col("feature2").cast(float).alias("feature2"))

    pl.testing.assert_frame_equal(result, expected)


def test_smooth_features_invalid_input(sample_smooth_df):
    with pytest.raises(TypeError):
        smooth_features("not_a_dataframe", features=("feature1",), sort_col="date", over_col="group", window_size=2)
    with pytest.raises(ValueError):
        smooth_features(
            sample_smooth_df, features=("non_existent_feature",), sort_col="date", over_col="group", window_size=2
        )


def test_smooth_features_with_nulls(sample_smooth_df):
    df_with_nulls = sample_smooth_df.with_columns(
        pl.when(pl.col("feature1") == 3.0).then(None).otherwise(pl.col("feature1")).alias("feature1")
    )
    result = smooth_features(
        df_with_nulls, features=("feature1",), sort_col="date", over_col="group", window_size=2
    ).sort("date", "group")
    result = result.collect()

    expected = pl.DataFrame(
        {
            "date": ["2023-01-01", "2023-01-02", "2023-01-03", "2023-01-04", "2023-01-05", "2023-01-06"],
            "group": ["A", "A", "A", "B", "B", "B"],
            "feature1": [None, 1.5, None, None, 4.5, 5.5],
            "feature2": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        }
    )

    pl.testing.assert_frame_equal(result, expected)


def test_smooth_features_window_size_one(sample_smooth_df):
    result = smooth_features(
        sample_smooth_df, features=("feature1", "feature2"), sort_col="date", over_col="group", window_size=1
    )
    result = result.collect()

    # With window_size=1, the result should be the same as the input
    pl.testing.assert_frame_equal(result, sample_smooth_df)


###
# `top_n_by_group`
###


@pytest.fixture
def top_n_df():
    return pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "subgroup": [1, 1, 2, 1, 2, 2, 1, 2, 3],
            "value": [10, 20, 30, 15, 25, 35, 5, 15, 25],
        }
    )


@pytest.mark.parametrize("filter", [True, False])
def test_top_n_by_group(top_n_df, filter):
    result = (
        top_n_by_group(top_n_df, n=2, rank_var="value", group_var=("group",), filter=filter)
        .collect()
        .sort("group", "subgroup")
    )
    if filter:
        assert result.shape == (6, 3)
        assert result["group"].to_list() == ["A", "A", "B", "B", "C", "C"]
        assert result["value"].to_list() == [20, 30, 25, 35, 15, 25]
    else:
        assert result.shape == (9, 4)
        assert "rank_mask" in result.columns
        assert result["rank_mask"].to_list() == [0, 1, 1, 0, 1, 1, 0, 1, 1]


def test_top_n_by_multiple_groups(top_n_df):
    result = (
        top_n_by_group(top_n_df, n=1, rank_var="value", group_var=("group", "subgroup"), filter=True)
        .sort("group", "subgroup")
        .collect()
    )
    assert result.shape == (7, 3)
    assert result["group"].to_list() == ["A", "A", "B", "B", "C", "C", "C"]
    assert result["subgroup"].to_list() == [1, 2, 1, 2, 1, 2, 3]
    assert result["value"].to_list() == [20, 30, 15, 35, 5, 15, 25]


@pytest.mark.parametrize("filter", [True, False])
def test_lazyframe_input(filter):
    lazy_df = pl.LazyFrame({"group": ["A", "B"], "value": [1, 2]})
    result = top_n_by_group(lazy_df, n=1, rank_var="value", group_var=("group",), filter=filter)
    assert isinstance(result, pl.LazyFrame)


def test_invalid_input():
    df = pl.DataFrame({"group": ["A", "B"], "wrong_column": [1, 2]})
    with pytest.raises(ValueError, match="missing one or more required columns"):
        top_n_by_group(df, n=1, rank_var="value", group_var=("group",), filter=True)
    with pytest.raises(TypeError, match="must be a Polars DataFrame or LazyFrame"):
        top_n_by_group("not_a_dataframe", n=1, rank_var="value", group_var=("group",), filter=True)


def test_empty_dataframe():
    empty_df = pl.DataFrame({"group": [], "value": []})
    result = top_n_by_group(empty_df, n=1, rank_var="value", group_var=("group",), filter=True).collect()
    assert result.shape == (0, 2)


def test_n_greater_than_group_size(top_n_df):
    result = top_n_by_group(top_n_df, n=5, rank_var="value", group_var=("group",), filter=True).collect()
    assert result.shape == (9, 3)


def test_tie_handling(top_n_df):
    df_with_ties = top_n_df.with_columns(
        pl.when(pl.col("value") == 25).then(26).otherwise(pl.col("value")).alias("value")
    )
    result = (
        top_n_by_group(df_with_ties, n=1, rank_var="value", group_var=("group",), filter=True)
        .sort("group", "subgroup")
        .collect()
    )
    assert result.shape == (3, 3)
    assert result["value"].to_list() == [30, 35, 26]
