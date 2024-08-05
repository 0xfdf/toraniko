"""Test functions in the math module."""

import pytest
import polars as pl
import numpy as np
from polars.testing import assert_frame_equal

from toraniko.math import (
    center_xsection,
    exp_weights,
    norm_xsection,
    percentiles_xsection,
    winsorize,
    winsorize_xsection,
)


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
    data = pl.DataFrame({"group": ["A", "A", "B", "B"], "value": [1, np.nan, 4, np.nan]}, strict=False)
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


###
# `winsorize`
###


@pytest.fixture
def sample_columns():
    """
    Fixture to provide sample numpy array columns for testing.

    Returns
    -------
    np.ndarray
    """
    return np.array(
        [
            [1, 100, 1000],
            [2, 200, 2000],
            [3, 300, 3000],
            [4, 400, 4000],
            [5, 500, 5000],
            [6, 600, 6000],
            [7, 700, 7000],
            [8, 800, 8000],
            [9, 900, 9000],
            [10, 1000, 10000],
        ]
    )


@pytest.fixture
def sample_columns_with_nans():
    """
    Fixture to provide sample numpy array columns for testing.

    Returns
    -------
    np.ndarray
    """
    return np.array(
        [
            [1, 100, np.nan],
            [2, 200, 2000],
            [3, 300, np.nan],
            [4, 400, np.nan],
            [5, 500, 5000],
            [6, 600, 6000],
            [7, 700, 7000],
            [8, 800, 8000],
            [9, 900, np.nan],
            [10, 1000, 10000],
        ]
    )


@pytest.fixture
def sample_rows():
    """
    Fixture to provide sample numpy rows for testing.

    Returns
    -------
    np.ndarray
    """
    return np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
        ]
    )


@pytest.fixture
def sample_rows_with_nans():
    """
    Fixture to provide sample numpy rows for testing.

    Returns
    -------
    np.ndarray
    """
    return np.array(
        [
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
            [np.nan, 2000, np.nan, np.nan, 5000, 6000, 7000, 8000, np.nan, 10000],
        ]
    )


def test_winsorize_axis_0(sample_columns):
    """
    Test column-wise winsorization (axis=0).

    This test checks if the function correctly winsorizes a 2D array column-wise.

    Parameters
    ----------
    sample_columns : np.ndarray
        Sample data fixture for column-wise testing.

    Raises
    ------
    AssertionError
        If the winsorized output doesn't match the expected result.
    """
    result = winsorize(sample_columns, percentile=0.2, axis=0)

    # Expected results (calculated manually for 20th and 80th percentiles)
    expected = np.array(
        [
            [2.8, 280, 2800],
            [2.8, 280, 2800],
            [3.0, 300, 3000],
            [4.0, 400, 4000],
            [5.0, 500, 5000],
            [6.0, 600, 6000],
            [7.0, 700, 7000],
            [8.0, 800, 8000],
            [8.2, 820, 8200],
            [8.2, 820, 8200],
        ]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_winsorize_axis_1(sample_rows):
    """
    Test row-wise winsorization (axis=1).

    This test checks if the function correctly winsorizes a 2D array row-wise.

    Parameters
    ----------
    sample_rows : np.ndarray
        Sample data fixture for row-wise testing.

    Raises
    ------
    AssertionError
        If the winsorized output doesn't match the expected result.
    """
    result = winsorize(sample_rows, percentile=0.2, axis=1)

    # Expected results (calculated manually for 20th and 80th percentiles)
    expected = np.array(
        [
            [2.8, 2.8, 3, 4, 5, 6, 7, 8, 8.2, 8.2],
            [280, 280, 300, 400, 500, 600, 700, 800, 820, 820],
            [2800, 2800, 3000, 4000, 5000, 6000, 7000, 8000, 8200, 8200],
        ]
    )

    np.testing.assert_array_almost_equal(result, expected)


def test_winsorize_axis_0_with_nans(sample_columns_with_nans):
    """
    Test winsorize function with NaN values.

    This test verifies that the function handles NaN values correctly.

    Raises
    ------
    AssertionError
        If the winsorized output doesn't handle NaN values as expected.
    """
    result = winsorize(sample_columns_with_nans, percentile=0.2, axis=0)
    expected = np.array(
        [
            [2.8e00, 2.8e02, np.nan],
            [2.8e00, 2.8e02, 5.0e03],
            [3.0e00, 3.0e02, np.nan],
            [4.0e00, 4.0e02, np.nan],
            [5.0e00, 5.0e02, 5.0e03],
            [6.0e00, 6.0e02, 6.0e03],
            [7.0e00, 7.0e02, 7.0e03],
            [8.0e00, 8.0e02, 8.0e03],
            [8.2e00, 8.2e02, np.nan],
            [8.2e00, 8.2e02, 8.0e03],
        ]
    )
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_winsorize_axis_1_with_nans(sample_rows_with_nans):
    """
    Test winsorize function with NaN values.

    This test verifies that the function handles NaN values correctly.

    Raises
    ------
    AssertionError
        If the winsorized output doesn't handle NaN values as expected.
    """
    result = winsorize(sample_rows_with_nans, percentile=0.2, axis=1)
    # take transpose of axis=0 test to test axis=1
    expected = np.array(
        [
            [2.8e00, 2.8e02, np.nan],
            [2.8e00, 2.8e02, 5.0e03],
            [3.0e00, 3.0e02, np.nan],
            [4.0e00, 4.0e02, np.nan],
            [5.0e00, 5.0e02, 5.0e03],
            [6.0e00, 6.0e02, 6.0e03],
            [7.0e00, 7.0e02, 7.0e03],
            [8.0e00, 8.0e02, 8.0e03],
            [8.2e00, 8.2e02, np.nan],
            [8.2e00, 8.2e02, 8.0e03],
        ]
    ).T
    np.testing.assert_allclose(result, expected, equal_nan=True)


def test_winsorize_invalid_percentile(sample_rows):
    """
    Test winsorize function with invalid percentile values.

    This test checks if the function raises a ValueError for percentile values outside [0, 1].

    Raises
    ------
    AssertionError
        If the function doesn't raise a ValueError for invalid percentile values.
    """
    with pytest.raises(TypeError):
        winsorize(sample_rows, percentile="string_percentile")
    with pytest.raises(ValueError):
        winsorize(sample_rows, percentile=-0.1)
    with pytest.raises(ValueError):
        winsorize(sample_rows, percentile=1.1)


def test_winsorize_1d_array():
    """
    Test winsorize function with a 1D array.

    This test verifies that the function works correctly with 1D input.

    Raises
    ------
    AssertionError
        If the winsorized output doesn't match the expected result for a 1D array.
    """
    data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    result = winsorize(data, percentile=0.2)
    expected = np.array([2.8, 2.8, 3, 4, 5, 6, 7, 8, 8.2, 8.2])
    np.testing.assert_array_equal(result, expected)


def test_winsorize_empty_array():
    """
    Test winsorize function with an empty array.

    This test checks if the function handles empty arrays correctly.

    Raises
    ------
    AssertionError
        If the function doesn't return an empty array for empty input.
    """
    data = np.array([])
    result = winsorize(data)
    np.testing.assert_array_equal(result, data)


def test_winsorize_all_nan():
    """
    Test winsorize function with an array containing only NaN values.

    This test verifies that the function handles arrays with only NaN values correctly.

    Raises
    ------
    AssertionError
        If the function doesn't return an array of NaN values for all-NaN input.
    """
    data = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    result = winsorize(data)
    np.testing.assert_array_equal(result, data)


###
# winsorize_xsection
###


@pytest.fixture
def sample_df():
    """
    Fixture to provide a sample DataFrame for testing.
    """
    return pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "value1": [1, 2, 10, 4, 5, 20, 7, 8, 30],
            "value2": [100, 200, 1000, 400, 500, 2000, 700, 800, 3000],
        }
    )


@pytest.mark.parametrize("lazy", [True, False])
def test_winsorize_xsection(sample_df, lazy):
    """
    Test basic functionality of winsorize_xsection.
    """
    if lazy:
        result = winsorize_xsection(
            sample_df.lazy(), data_cols=("value1", "value2"), group_col="group", percentile=0.1
        ).sort("group")
        assert isinstance(result, pl.LazyFrame)
        result = result.collect()
    else:
        result = winsorize_xsection(sample_df, data_cols=("value1", "value2"), group_col="group", percentile=0.1).sort(
            "group"
        )

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "value1": [1.2, 2.0, 8.4, 4.2, 5.0, 17.0, 7.2, 8.0, 25.6],
            "value2": [120.0, 200.0, 840.0, 420.0, 500.0, 1700.0, 720.0, 800.0, 2560.0],
        }
    )

    assert_frame_equal(result, expected, check_exact=False)


@pytest.fixture
def sample_df_with_nans():
    """
    Fixture to provide a sample DataFrame with NaN values for testing.
    """
    return pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "value1": [1, np.nan, 10, 4, 5, np.nan, 7, 8, 30],
            "value2": [100, 200, np.nan, np.nan, 500, 2000, 700, np.nan, 3000],
        },
        strict=False,
    )


def test_winsorize_xsection_with_nans(sample_df_with_nans):
    """
    Test winsorize_xsection with NaN values.
    """
    result = winsorize_xsection(
        sample_df_with_nans, data_cols=("value1", "value2"), group_col="group", percentile=0.1
    ).sort("group")

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "value1": [1.9, np.nan, 9.1, 4.1, 4.9, np.nan, 7.2, 8.0, 25.6],
            "value2": [110.0, 190.0, np.nan, np.nan, 650.0, 1850.0, 930.0, np.nan, 2770.0],
        },
        strict=False,
    )

    assert_frame_equal(result, expected, check_exact=False)


###
# `xsection_percentiles`
###


def test_xsection_percentiles(sample_df):
    """
    Test basic functionality of xsection_percentiles.
    """
    result = sample_df.with_columns(percentiles_xsection("value1", "group", 0.25, 0.75).alias("result")).sort("group")

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "value1": [1, 2, 10, 4, 5, 20, 7, 8, 30],
            "result": [1.0, 2.0, 10.0, 4.0, 5.0, 20.0, 7.0, 8.0, 30.0],
        }
    )

    pl.testing.assert_frame_equal(result.select("group", "value1", "result"), expected)


def test_xsection_percentiles_with_nans(sample_df_with_nans):
    """
    Test xsection_percentiles with NaN values.
    """
    result = sample_df_with_nans.with_columns(percentiles_xsection("value1", "group", 0.25, 0.75).alias("result"))

    expected = pl.DataFrame(
        {
            "group": ["A", "A", "A", "B", "B", "B", "C", "C", "C"],
            "value1": [1.0, np.nan, 10.0, 4.0, 5.0, np.nan, 7.0, 8.0, 30.0],
            "result": [1.0, np.nan, 10.0, 4.0, 5.0, np.nan, 7.0, 8.0, 30.0],
        }
    )

    pl.testing.assert_frame_equal(result.select("group", "value1", "result"), expected)


###
# `exp_weights`
###


def test_exp_weights_basic():
    """
    Test basic functionality of exp_weights.
    """
    result = exp_weights(window=5, half_life=2)
    expected = np.array([0.25, 0.35355339, 0.5, 0.70710678, 1.0])
    np.testing.assert_array_almost_equal(result, expected, decimal=6)


def test_exp_weights_window_1():
    """
    Test exp_weights with window of 1.
    """
    result = exp_weights(window=1, half_life=2)
    expected = np.array([1.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_exp_weights_half_life_1():
    """
    Test exp_weights with half_life of 1.
    """
    result = exp_weights(window=5, half_life=1)
    expected = np.array([0.0625, 0.125, 0.25, 0.5, 1.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_exp_weights_large_window():
    """
    Test exp_weights with a large window.
    """
    result = exp_weights(window=100, half_life=10)
    assert len(result) == 100
    assert result[-1] == 1.0
    assert result[0] < result[-1]


def test_exp_weights_decreasing():
    """
    Test that weights are decreasing from end to start.
    """
    result = exp_weights(window=10, half_life=3)
    assert np.all(np.diff(result) > 0)


def test_exp_weights_half_life():
    """
    Test that weights actually decay by half each half_life.
    """
    half_life = 5
    window = 20
    weights = exp_weights(window, half_life)
    for i in range(0, window - half_life, half_life):
        assert np.isclose(weights[i], 0.5 * weights[i + half_life], rtol=1e-5)


def test_exp_weights_invalid_window():
    """
    Test exp_weights with invalid window value.
    """
    with pytest.raises(ValueError):
        exp_weights(window=0, half_life=2)

    with pytest.raises(ValueError):
        exp_weights(window=-1, half_life=2)

    with pytest.raises(TypeError):
        exp_weights(window="window", half_life=2)

    with pytest.raises(TypeError):
        exp_weights(window=5.1, half_life=3)


def test_exp_weights_invalid_half_life():
    """
    Test exp_weights with invalid half_life value.
    """
    with pytest.raises(ValueError):
        exp_weights(window=5, half_life=0)

    with pytest.raises(ValueError):
        exp_weights(window=5, half_life=-1)

    with pytest.raises(TypeError):
        exp_weights(window=5, half_life="half_life")

    with pytest.raises(TypeError):
        exp_weights(window=5, half_life=3.2)


def test_output():
    """
    Test with a specific input and output.
    """
    result = exp_weights(10, 10)
    expected = np.array(
        [0.53588673, 0.57434918, 0.61557221, 0.65975396, 0.70710678, 0.75785828, 0.8122524, 0.87055056, 0.93303299, 1.0]
    )
    np.testing.assert_array_almost_equal(result, expected)
