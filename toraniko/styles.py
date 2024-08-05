"""Style factor implementations."""

import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from toraniko.math import (
    exp_weights,
    center_xsection,
    percentiles_xsection,
    winsorize_xsection,
)

###
# NB: These functions do not try to handle NaN or null resilience for you, nor do they make allowances
# for data having pathological distributions. Garbage in, garbage out. You need to inspect your data
# and use the functions in the math and utils modules to ensure your features are sane and
# well-behaved before you try to construct factors from them!
###


def factor_mom(
    returns_df: pl.DataFrame | pl.LazyFrame,
    trailing_days: int = 504,
    half_life: int = 126,
    lag: int = 20,
    winsor_factor: float = 0.01,
) -> pl.LazyFrame:
    """Estimate rolling symbol by symbol momentum factor scores using asset returns.

    Parameters
    ----------
    returns_df: Polars DataFrame containing columns: | date | symbol | asset_returns |
    trailing_days: int look back period over which to measure momentum
    half_life: int decay rate for exponential weighting, in days
    lag: int number of days to lag the current day's return observation (20 trading days is one month)

    Returns
    -------
    Polars DataFrame containing columns: | date | symbol | mom_score |
    """
    weights = exp_weights(trailing_days, half_life)

    def weighted_cumprod(values: np.ndarray) -> float:
        return (np.cumprod(1 + (values * weights[-len(values) :])) - 1)[-1]  # type: ignore

    try:
        df = (
            returns_df.lazy()
            .sort(by="date")
            .with_columns(pl.col("asset_returns").shift(lag).over("symbol").alias("asset_returns"))
            .with_columns(
                pl.col("asset_returns")
                .rolling_map(weighted_cumprod, window_size=trailing_days)
                .over(pl.col("symbol"))
                .alias("mom_score")
            )
        ).collect()
        df = winsorize_xsection(df, ("mom_score",), "date", percentile=winsor_factor)
        return df.lazy().select(
            pl.col("date"),
            pl.col("symbol"),
            center_xsection("mom_score", "date", True).alias("mom_score"),
        )
    except AttributeError as e:
        raise TypeError("`returns_df` must be a Polars DataFrame | LazyFrame, but it's missing attributes") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("`returns_df` must have 'date', 'symbol' and 'asset_returns' columns") from e


def factor_sze(
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    lower_decile: float = 0.2,
    upper_decile: float = 0.8,
) -> pl.LazyFrame:
    """Estimate rolling symbol by symbol size factor scores using asset market caps.

    Parameters
    ----------
    mkt_cap_df: Polars DataFrame containing columns: | date | symbol | market_cap |

    Returns
    -------
    Polars DataFrame containing columns: | date | symbol | sze_score |
    """
    try:
        return (
            mkt_cap_df.lazy()
            # our factor is the Fama-French SMB, i.e. small-minus-big, because the size risk premium
            # is on the smaller firms rather than the larger ones. consequently we multiply by -1
            .with_columns(pl.col("market_cap").log().alias("sze_score") * -1)
            .with_columns(
                "date",
                "symbol",
                (center_xsection("sze_score", "date", True)).alias("sze_score"),
            )
            .with_columns(percentiles_xsection("sze_score", "date", lower_decile, upper_decile, 0.0).alias("sze_score"))
            .select("date", "symbol", "sze_score")
        )
    except AttributeError as e:
        raise TypeError("`mkt_cap_df` must be a Polars DataFrame or LazyFrame, but it's missing attributes") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError("`mkt_cap_df` must have 'date', 'symbol' and 'market_cap' columns") from e


def factor_val(value_df: pl.DataFrame | pl.LazyFrame, winsorize_features: float | None = None) -> pl.LazyFrame:
    """Estimate rolling symbol by symbol value factor scores using price ratios.

    Parameters
    ----------
    value_df: Polars DataFrame containing columns: | date | symbol | book_price | sales_price | cf_price
    winsorize_features: optional float indicating if the features should be winsorized. none applied if None

    Returns
    -------
    Polars DataFrame containing: | date | symbol | val_score |
    """
    try:
        if winsorize_features is not None:
            value_df = winsorize_xsection(value_df, ("book_price", "sales_price", "cf_price"), "date")
        return (
            value_df.lazy()
            .with_columns(
                pl.col("book_price").log().alias("book_price"),
                pl.col("sales_price").log().alias("sales_price"),
            )
            .with_columns(
                center_xsection("book_price", "date", True).alias("book_price"),
                center_xsection("sales_price", "date", True).alias("sales_price"),
                center_xsection("cf_price", "date", True).alias("cf_price"),
            )
            .with_columns(
                # NB: it's imperative you've properly handled NaNs prior to this point
                pl.mean_horizontal(
                    pl.col("book_price"),
                    pl.col("sales_price"),
                    pl.col("cf_price"),
                ).alias("val_score")
            )
            .select(
                pl.col("date"),
                pl.col("symbol"),
                center_xsection("val_score", "date", True).alias("val_score"),
            )
        )
    except AttributeError as e:
        raise TypeError("`value_df` must be a Polars DataFrame or LazyFrame, but it's missing attributes") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(
            "`value_df` must have 'date', 'symbol', 'book_price', 'sales_price' and 'fcf_price' columns"
        ) from e
