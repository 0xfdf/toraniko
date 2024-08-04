"""Test functions in the math module."""

import pytest
import polars as pl
import numpy as np
from toraniko.math import center_xsection, norm_xsection


@pytest.fixture
def sample_data():
    """
    Fixture to provide sample data for testing.

    Returns
    -------
    pl.DataFrame
        A DataFrame with a 'group' column and a 'value' column.
    """
    return pl.DataFrame({"group": ["A", "A", "A", "B", "B", "B"], "value": [1, 2, 3, 4, 5, 6]})


###
# `center_xsection`
###


def test_center_xsection_centering(sample_data):
    """
    Test centering without standardization.

    Parameters
    ----------
    sample_data : pl.DataFrame
        The sample data to test.

    Asserts
    -------
    The centered values match the expected centered values.
    """
    centered_df = sample_data.with_columns(center_xsection("value", "group"))
    expected_centered_values = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    assert np.allclose(centered_df["value"].to_numpy(), expected_centered_values)


def test_center_xsection_standardizing(sample_data):
    """
    Test centering and standardizing.

    Parameters
    ----------
    sample_data : pl.DataFrame
        The sample data to test.

    Asserts
    -------
    The standardized values match the expected standardized values.
    """
    standardized_df = sample_data.with_columns(center_xsection("value", "group", standardize=True))
    expected_standardized_values = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    assert np.allclose(standardized_df["value"].to_numpy(), expected_standardized_values)


def test_center_xsection_handle_nan(sample_data):
    """
    Test handling of NaN values.

    Parameters
    ----------
    sample_data : pl.DataFrame
        The sample data to test, with an additional NaN column.

    Asserts
    -------
    The values in the 'nan_col' column are all NaN.
    """
    sample_data_with_nan = sample_data.with_columns(pl.lit(np.nan).alias("nan_col"))
    centered_df = sample_data_with_nan.with_columns(center_xsection("nan_col", "group"))
    expected_values = [np.nan] * len(centered_df)
    assert np.allclose(centered_df["nan_col"].to_numpy(), expected_values, equal_nan=True)


def test_center_xsection_different_groups():
    """
    Test with a more complex group and value structure.

    Asserts
    -------
    The centered values match the expected values.
    """
    data = pl.DataFrame({"group": ["A", "A", "B", "B", "C", "C"], "value": [1.5, 2.5, 5.5, 6.5, 9.0, 9.0]})
    centered_df = data.with_columns(center_xsection("value", "group"))
    expected_values = [-0.5, 0.5, -0.5, 0.5, 0.0, 0.0]
    assert np.allclose(centered_df["value"].to_numpy(), expected_values)


def test_center_xsection_empty_group():
    """
    Test with an empty group.

    Asserts
    -------
    The group 'C' is empty in the filtered DataFrame.
    """
    data = pl.DataFrame({"group": ["A", "A", "B", "B"], "value": [1.0, 2.0, 3.0, 4.0]})
    empty_group = data.filter(pl.col("group") == "C")
    assert empty_group.is_empty()


def test_center_xsection_all_nan():
    """
    Test when the entire column consists of NaN values.

    Asserts
    -------
    The result column should contain only NaN values.
    """
    data = pl.DataFrame({"group": ["A", "A", "B", "B"], "value": [np.nan, np.nan, np.nan, np.nan]})
    centered_df = data.with_columns(center_xsection("value", "group"))
    expected_values = [np.nan, np.nan, np.nan, np.nan]
    assert np.allclose(centered_df["value"].to_numpy(), expected_values, equal_nan=True)


def test_center_xsection_single_row_group():
    """
    Test centering when a group has only one row.

    Asserts
    -------
    The result should be 0.0 for the single row group.
    """
    data = pl.DataFrame({"group": ["A", "A", "B"], "value": [1.0, 2.0, 3.0]})
    centered_df = data.with_columns(center_xsection("value", "group"))
    expected_values = [-0.5, 0.5, 0.0]  # Centering for group B with one value should be 0.0
    assert np.allclose(centered_df["value"].to_numpy(), expected_values)


def test_center_xsection_mixed_data_types():
    """
    Test handling of mixed data types, where non-numeric columns are ignored.

    Asserts
    -------
    The centering process should only affect numeric columns.
    """
    data = pl.DataFrame(
        {"group": ["A", "A", "B", "B"], "value": [1.0, 2.0, 3.0, 4.0], "category": ["x", "y", "z", "w"]}
    )
    centered_df = data.with_columns(center_xsection("value", "group"))
    expected_values = [-0.5, 0.5, -0.5, 0.5]
    assert np.allclose(centered_df["value"].to_numpy(), expected_values)
    assert all(centered_df["category"] == data["category"])


###
# `norm_xsection`
###


def test_norm_xsection_default_range(sample_data):
    """
    Test normalization with default range [0, 1].

    Parameters
    ----------
    sample_data : pl.DataFrame
        The sample data to test.

    Asserts
    -------
    The normalized values match the expected values in the range [0, 1].
    """
    normalized_df = sample_data.with_columns(norm_xsection("value", "group"))
    expected_normalized_values = [0.0, 0.5, 1.0, 0.0, 0.5, 1.0]
    assert np.allclose(normalized_df["value"].to_numpy(), expected_normalized_values)


def test_norm_xsection_custom_range(sample_data):
    """
    Test normalization with a custom range [-1, 1].

    Parameters
    ----------
    sample_data : pl.DataFrame
        The sample data to test.

    Asserts
    -------
    The normalized values match the expected values in the range [-1, 1].
    """
    normalized_df = sample_data.with_columns(norm_xsection("value", "group", lower=-1, upper=1))
    expected_normalized_values = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0]
    assert np.allclose(normalized_df["value"].to_numpy(), expected_normalized_values)


def test_norm_xsection_nan_values():
    """
    Test normalization when the target column contains NaN values.

    Asserts
    -------
    The NaN values are preserved in the output.
    """
    data = pl.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, np.nan, 4, np.nan]})
    normalized_df = data.with_columns(norm_xsection("value", "group"))
    expected_normalized_values = [0.0, np.nan, 0.0, np.nan]
    assert np.allclose(normalized_df["value"].to_numpy(), expected_normalized_values, equal_nan=True)


def test_norm_xsection_single_value_group():
    """
    Test normalization when a group has a single value.

    Asserts
    -------
    The result should be the lower bound of the range, as there's no range to normalize.
    """
    data = pl.DataFrame({"group": ["A", "A", "B"], "value": [1, 2, 3]})
    normalized_df = data.with_columns(norm_xsection("value", "group"))
    expected_normalized_values = [0.0, 1.0, 0.0]  # Single value group should map to the lower bound
    assert np.allclose(normalized_df["value"].to_numpy(), expected_normalized_values)


def test_norm_xsection_identical_values():
    """
    Test normalization when all values in a group are identical.

    Asserts
    -------
    The normalized values should all be the lower bound, as there's no range to normalize.
    """
    data = pl.DataFrame({"group": ["A", "A", "B", "B"], "value": [5, 5, 5, 5]})
    normalized_df = data.with_columns(norm_xsection("value", "group"))
    expected_normalized_values = [0.0, 0.0, 0.0, 0.0]  # Identical values map to the lower bound
    assert np.allclose(normalized_df["value"].to_numpy(), expected_normalized_values)


def test_norm_xsection_mixed_data_types():
    """
    Test handling of mixed data types, where non-numeric columns are ignored.

    Asserts
    -------
    The normalization process should only affect numeric columns.
    """
    data = pl.DataFrame(
        {"group": ["A", "A", "B", "B"], "value": [1.0, 2.0, 3.0, 4.0], "category": ["x", "y", "z", "w"]}
    )
    normalized_df = data.with_columns(norm_xsection("value", "group"))
    expected_normalized_values = [0.0, 1.0, 0.0, 1.0]
    actual_values = normalized_df["value"].to_numpy()
    print(actual_values)
    assert np.allclose(actual_values, expected_normalized_values)
    assert all(normalized_df["category"] == data["category"])


def test_norm_xsection_custom_range_large_values():
    """
    Test normalization with a custom range [100, 200] and large values.

    Asserts
    -------
    The normalized values match the expected values in the range [100, 200].
    """
    data = pl.DataFrame({"group": ["A", "A", "B", "B"], "value": [1000, 2000, 3000, 4000]})
    normalized_df = data.with_columns(norm_xsection("value", "group", lower=100, upper=200))
    expected_normalized_values = [100.0, 200.0, 100.0, 200.0]
    assert np.allclose(normalized_df["value"].to_numpy(), expected_normalized_values)
