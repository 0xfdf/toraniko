"""Test functions in the math module."""

import pytest
import polars as pl
import numpy as np
from toraniko.math import center_xsection


@pytest.fixture
def sample_data():
    return pl.DataFrame({"group": ["A", "A", "A", "B", "B", "B"], "value": [1, 2, 3, 4, 5, 6]})


###
# `center_xsection`
###


def test_center_xsection_centering(sample_data):
    # Test centering without standardization
    centered_df = sample_data.with_columns(center_xsection("value", "group"))
    expected_centered_values = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    assert np.allclose(centered_df["value"].to_numpy(), expected_centered_values)


def test_center_xsection_standardizing(sample_data):
    # Test centering and standardizing
    standardized_df = sample_data.with_columns(center_xsection("value", "group", standardize=True))
    expected_standardized_values = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]  # These values are 0-mean, unit-variance
    assert np.allclose(standardized_df["value"].to_numpy(), expected_standardized_values)


def test_center_xsection_handle_nan(sample_data):
    # Test handling of NaN values
    sample_data_with_nan = sample_data.with_column(pl.lit(np.nan).alias("nan_col"))
    centered_df = sample_data_with_nan.with_columns(center_xsection("nan_col", "group"))
    expected_values = [np.nan] * len(centered_df)
    assert np.allclose(centered_df["nan_col"].to_numpy(), expected_values, equal_nan=True)


def test_center_xsection_different_groups():
    # Test with a more complex group and value structure
    data = pl.DataFrame({"group": ["A", "A", "B", "B", "C", "C"], "value": [1.5, 2.5, 5.5, 6.5, 9.0, 9.0]})
    centered_df = data.with_columns(center_xsection("value", "group"))
    expected_values = [-0.5, 0.5, -0.5, 0.5, 0.0, 0.0]
    assert np.allclose(centered_df["value"].to_numpy(), expected_values)


def test_center_xsection_empty_group():
    # Test with an empty group
    data = pl.DataFrame({"group": ["A", "A", "B", "B"], "value": [1.0, 2.0, 3.0, 4.0]})
    empty_group = data.filter(pl.col("group") == "C")
    assert empty_group.is_empty()
