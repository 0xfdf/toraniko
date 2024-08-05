"""Utility functions, primarily for data cleaning."""

import numpy as np
import polars as pl


def fill_features(
    df: pl.DataFrame | pl.LazyFrame, features: tuple[str, ...], sort_col: str, over_col: str
) -> pl.LazyFrame:
    """Cast feature columns to numeric (float), convert NaN and inf values to null, then forward fill nulls
    for each column of `features`, sorted on `sort_col` and partitioned by `over_col`.

    Parameters
    ----------
    df: Polars DataFrame or LazyFrame containing columns `sort_col`, `over_col` and each of `features`
    features: collection of strings indicating which columns of `df` are the feature values
    sort_col: str column of `df` indicating how to sort
    over_col: str column of `df` indicating how to partition

    Returns
    -------
    Polars LazyFrame containing the original columns with cleaned feature data
    """
    try:
        # eagerly check all `features`, `sort_col`, `over_col` present: can't catch ColumNotFoundError in lazy context
        assert all(c in df.columns for c in features + (sort_col, over_col))
        return (
            df.lazy()
            .with_columns([pl.col(f).cast(float).alias(f) for f in features])
            .with_columns(
                [
                    pl.when(
                        (pl.col(f).abs() == np.inf)
                        | (pl.col(f) == np.nan)
                        | (pl.col(f).is_null())
                        | (pl.col(f).cast(str) == "NaN")
                    )
                    .then(None)
                    .otherwise(pl.col(f))
                    .alias(f)
                    for f in features
                ]
            )
            .sort(by=sort_col)
            .with_columns([pl.col(f).forward_fill().over(over_col).alias(f) for f in features])
        )
    except AttributeError as e:
        raise TypeError("`df` must be a Polars DataFrame | LazyFrame, but it's missing required attributes") from e
    except AssertionError as e:
        raise ValueError(f"`df` must have all of {[over_col, sort_col] + list(features)} as columns") from e


def smooth_features(
    df: pl.DataFrame | pl.LazyFrame,
    features: tuple[str, ...],
    sort_col: str,
    over_col: str,
    window_size: int,
) -> pl.LazyFrame:
    """Smooth the `features` columns of `df` by taking the rolling mean of each, sorted over `sort_col` and
    partitioned by `over_col`, using `window_size` trailing periods for the moving average window.

    Parameters
    ----------
    df: Polars DataFrame | LazyFrame containing columns `sort_col`, `over_col` and each of `features`
    features: collection of strings indicating which columns of `df` are the feature values
    sort_col: str column of `df` indicating how to sort
    over_col: str column of `df` indicating how to partition
    window_size: int number of time periods for the moving average

    Returns
    -------
    Polars LazyFrame containing the original columns, with each of `features` replaced with moving average
    """
    try:
        # eagerly check `over_col`, `sort_col`, `features` present: can't catch pl.ColumnNotFoundError in lazy context
        assert all(c in df.columns for c in features + (over_col, sort_col))
        return (
            df.lazy()
            .sort(by=sort_col)
            .with_columns([pl.col(f).rolling_mean(window_size=window_size).over(over_col).alias(f) for f in features])
        )
    except AttributeError as e:
        raise TypeError("`df` must be a Polars DataFrame | LazyFrame, but it's missing required attributes") from e
    except AssertionError as e:
        raise ValueError(f"`df` must have all of {[over_col, sort_col] + list(features)} as columns") from e


def top_n_by_group(
    df: pl.DataFrame | pl.LazyFrame,
    n: int,
    rank_var: str,
    group_var: tuple[str, ...],
    filter: bool = True,
) -> pl.LazyFrame:
    """Mark the top `n` rows in each of `group_var` according to `rank_var` descending.

    If `filter` is True, the returned DataFrame contains only the filtered data. If `filter` is False,
    the returned DataFrame has all data, with an additional 'rank_mask' column indicating if that row
    is in the filter.

    Parameters
    ----------
    df: Polars DataFrame | LazyFrame
    n: integer indicating the top rows to take in each group
    rank_var: str column name to rank on
    group_var: tuple of str column names to group and sort on
    filter: boolean indicating how much data to return

    Returns
    -------
    Polars LazyFrame containing original columns and optional filter column
    """
    try:
        # eagerly check `rank_var`, `group_var` are present: we can't catch a ColumnNotFoundError in a lazy context
        assert all(c in df.columns for c in (rank_var,) + group_var)
        rdf = (
            df.lazy()
            .sort(by=list(group_var) + [rank_var])
            .with_columns(pl.col(rank_var).rank(descending=True).over(group_var).cast(int).alias("rank"))
        )
        match filter:
            case True:
                return rdf.filter(pl.col("rank") <= n).drop("rank").sort(by=list(group_var) + [rank_var])
            case False:
                return (
                    rdf.with_columns(
                        pl.when(pl.col("rank") <= n).then(pl.lit(1)).otherwise(pl.lit(0)).alias("rank_mask")
                    )
                    .drop("rank")
                    .sort(by=list(group_var) + [rank_var])
                )
    except AssertionError as e:
        raise ValueError(f"`df` is missing one or more required columns: '{rank_var}' and '{group_var}'") from e
    except AttributeError as e:
        raise TypeError("`df` must be a Polars DataFrame or LazyFrame but is missing a required attribute") from e
