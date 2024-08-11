"""Style factor implementations."""

import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from toraniko.math import (
    exp_weights,
    center_xsection,
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
    winsor_factor: float | None = 0.01,
    center: bool = True,
    standardize: bool = True,
    asset_returns_col: str = "asset_returns",
    symbol_col: str = "symbol",
    date_col: str = "date",
    score_col: str = "mom_score",
) -> pl.LazyFrame:
    """Estimate rolling symbol by symbol momentum factor scores using asset returns.

    This implements the momentum factor similar to the common Barra specification:

    1. First lag the asset returns by `lag=20` days to remove the most recent month (1 month = 20 trading days)
       from consideration.
    2. Next exponentially weight the asset returns with a `half_life=126` rate of decay, and take the cumulative
       return of each asset from t - `trailing_days=504` through to t for each time period t.

    There are also two optional post-processing steps:

    1. Optionally winsorize the momentum scores at the `winsor_factor` percentile, symmetrically. Default 1st and 99th.
    2. Optionally center (and standardize if `standardize=True`) the final momentum scores around 0. Defaults to true.

    In practice, you should center and standardize your factor scores unless you have a very good reason not to.

    Parameters
    ----------
    returns_df: Polars DataFrame containing columns: | date | symbol | asset_returns |
    trailing_days: int look back period over which to measure momentum
    half_life: int decay rate for exponential weighting, in days
    lag: int number of days to lag the current day's return observation (20 trading days is one month)
    winsor_factor: optional float symmetric percentile at which to winsorize, e.g. 0.01 is 1st and 99th percentiles
    center: boolean indicating whether to center the final momentum scores before returning
    standardize: boolean indicating whether to standardize the final momentum scores after centering
    asset_returns_col: str name of the column we expect to find the asset returns value in, defaults to "asset_returns"
    symbol_col: str name of the column we expect to find the symbol names in, defaults to "symbol"
    date_col: str name of the column we expect to find the dates (or datetimes) in, defaults to "date"
    score_col: str name of the column to place the score values under, defaults to "mom_score"

    Returns
    -------
    Polars DataFrame containing columns: | `date_col` | `symbol_col` | `score_col` |
    """
    weights = exp_weights(trailing_days, half_life)

    def weighted_cumprod(values: np.ndarray) -> float:
        """Wrapper function to calculate weighted cumulative product (geometric sum)."""
        return (np.cumprod(1 + (values * weights[-len(values) :])) - 1)[-1]  # type: ignore

    try:
        df = returns_df.lazy()
        df = (
            df.sort(by=date_col)
            .with_columns(pl.col(asset_returns_col).shift(lag).over(symbol_col).alias(asset_returns_col))
            .with_columns(
                pl.col(asset_returns_col)
                .rolling_map(weighted_cumprod, window_size=trailing_days)
                .over(pl.col(symbol_col))
                .alias(score_col)
            )
        )
        if winsor_factor is not None:
            df = winsorize_xsection(df.collect(), (score_col,), date_col, percentile=winsor_factor).lazy()
        if center:
            df = df.with_columns(
                center_xsection(score_col, date_col, standardize=standardize).alias(score_col),
            )
        return df.select(date_col, symbol_col, score_col)
    except AttributeError as e:
        raise TypeError("`returns_df` must be a Polars DataFrame | LazyFrame, but it's missing attributes") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(
            f"`returns_df` must have '{date_col}', '{symbol_col}' and '{asset_returns_col}' columns"
        ) from e


def factor_sze(
    mkt_cap_df: pl.DataFrame | pl.LazyFrame,
    center: bool = True,
    standardize: bool = True,
    mkt_cap_col: str = "market_cap",
    symbol_col: str = "symbol",
    date_col: str = "date",
    score_col: str = "sze_score",
) -> pl.LazyFrame:
    """Estimate rolling symbol by symbol size factor scores using asset market caps.

    The market factor is very simple, we just take the logarithm of asset market caps. This reduces the market caps
    to their rough order of magnitude, which is the salient feature we care most about. Given that we want to capture
    the risk premium of smaller factors over larger ones, we also multiply by -1.

    In practice, you should center and standardize your factor scores unless you have a very good reason not to.

    Parameters
    ----------
    mkt_cap_df: Polars DataFrame containing columns: | date | symbol | market_cap |
    center: boolean indicating whether to center the final size scores before returning
    standardize: boolean indicating whether to standardize the final size scores after centering
    mkt_cap_col: str name of the column we expect to find the market cap values in, defaults to "market_cap"
    symbol_col: str name of the column we expect to find the symbol names in, defaults to "symbol"
    date_col: str name of the column we expect to find the dates (or datetimes) in, defaults to "date"
    score_col: str name of the column to place the score values under, defaults to "sze_score"

    Returns
    -------
    Polars DataFrame containing columns: | `date` | `symbol` | `score_col` |
    """
    try:
        df = mkt_cap_df.lazy().with_columns(pl.col(mkt_cap_col).log().alias(score_col))
        if center:
            df = df.with_columns((center_xsection(score_col, date_col, standardize=standardize)).alias(score_col) * -1)
        return df.select(date_col, symbol_col, score_col)
    except AttributeError as e:
        raise TypeError("`mkt_cap_df` must be a Polars DataFrame or LazyFrame, but it's missing attributes") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(f"`mkt_cap_df` must have '{date_col}', '{symbol_col}' and '{mkt_cap_col}' columns") from e


def factor_val(
    value_df: pl.DataFrame | pl.LazyFrame,
    winsor_factor: float | None = 0.05,
    center: bool = True,
    standardize: bool = True,
    bp_col: str = "book_price",
    sp_col: str = "sales_price",
    cf_col: str = "cf_price",
    symbol_col: str = "symbol",
    date_col: str = "date",
    score_col: str = "val_score",
) -> pl.LazyFrame:
    """Estimate rolling symbol by symbol value factor scores using price ratios.

    This implements the value factor using the three major variables considered by most vendors such as Barra: the
    book to price ratio, the sales to price ratio and the cash flow to price ratio.

    First we individually center and standardize each of the three features cross-sectionally. Then we take their
    simple average in each time period's cross-section. This is the final score, which is optionally centered and
    standardized once more.

    In practice, you should center and standardize your factor scores unless you have a very good reason not to.

    Parameters
    ----------
    value_df: Polars DataFrame containing columns: | date | symbol | book_price | sales_price | cf_price
    winsor_factor: optional float indicating what percentile to symmetrically winsorize features at, if desired
    center: boolean indicating whether to center the final value scores before returning
    standardize: boolean indicating whether to standardize the final value scores after centering
    bp_col: str name of the column we expect to find the book-price ratio values in, defaults to "book_price"
    sp_col: str name of the column we expect to find the sales-price ratio values in, defaults to "sales_price"
    cf_col: str name of the column we expect to find the cash flow-price ratio values in, defaults to "cf_price"
    symbol_col: str name of the column we expect to find the symbol names in, defaults to "symbol"
    date_col: str name of the column we expect to find the dates (or datetimes) in, defaults to "date"
    score_col: str name of the column to place the score values under, defaults to "val_score"

    Returns
    -------
    Polars DataFrame containing: | `date` | `symbol` | `score_col` |
    """
    try:
        if winsor_factor is not None:
            value_df = winsorize_xsection(value_df, (bp_col, sp_col, cf_col), date_col, percentile=winsor_factor)
        df = (
            value_df.lazy()
            .with_columns(
                pl.col(bp_col).log().alias(bp_col),
                pl.col(sp_col).log().alias(sp_col),
            )
            .with_columns(
                center_xsection(bp_col, date_col, True).alias(bp_col),
                center_xsection(sp_col, date_col, True).alias(sp_col),
                center_xsection(cf_col, date_col, True).alias(cf_col),
            )
            .with_columns(
                # NB: it's imperative you've properly handled NaNs prior to this point
                pl.mean_horizontal(
                    pl.col(bp_col),
                    pl.col(sp_col),
                    pl.col(cf_col),
                ).alias(score_col)
            )
        )
        if center:
            df = df.with_columns(center_xsection(score_col, date_col, standardize=True).alias(score_col))
        return df.select(
            date_col,
            symbol_col,
            score_col,
        )
    except AttributeError as e:
        raise TypeError("`value_df` must be a Polars DataFrame or LazyFrame, but it's missing attributes") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(
            f"`value_df` must have '{date_col}', '{symbol_col}', '{bp_col}', '{sp_col}' and '{cf_col}' columns"
        ) from e
