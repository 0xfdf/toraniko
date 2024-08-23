"""Basic mathematical and statistical operations used in the model."""

from typing import Literal

import numpy as np
import polars as pl
import polars.exceptions as pl_exc


def center_xsection(target_col: str, over_col: str, standardize: bool = False) -> pl.Expr:
    """Cross-sectionally center (and optionally standardize) a Polars DataFrame `target_col` partitioned by `over_col`.

    This returns a Polars expression, so it can be chained in a `select` or `with_columns` invocation
    without needing to set a new intermediate DataFrame or materialize lazy evaluation.

    Parameters
    ----------
    target_col: the column to be standardized
    over_col: the column over which standardization should be applied, cross-sectionally
    standardize: boolean indicating if we should also standardize the target column

    Returns
    -------
    Polars Expr
    """
    expr = pl.col(target_col) - pl.col(target_col).drop_nulls().drop_nans().mean().over(over_col)
    if standardize:
        return expr / pl.col(target_col).drop_nulls().drop_nans().std().over(over_col)
    return expr


def norm_xsection(
    target_col: str,
    over_col: str,
    lower: int | float = 0,
    upper: int | float = 1,
) -> pl.Expr:
    """Cross-sectionally normalize a Polars DataFrame `target_col` partitioned by `over_col`, with rescaling
    to the interval [`lower`, `upper`].

    This returns a Polars expression, so it can be chained in a `select` or `with_columns` invocation
    without needing to set a new intermediate DataFrame or materialize lazy evaluation.

    NaN values are not propagated in the max and min calculation, but NaN values are preserved for normalization.

    Parameters
    ----------
    target_col: str name of the column to normalize
    over_col: str name of the column to partition the normalization by
    lower: lower bound of the rescaling interval, defaults to 0 to construct a percent
    upper: upper bound of the rescaling interval, defaults to 1 to construct a percent

    Returns
    -------
    Polars Expr
    """
    min_col = pl.col(target_col).drop_nans().min().over(over_col)
    max_col = pl.col(target_col).drop_nans().max().over(over_col)

    norm_col = (
        pl.when(pl.col(target_col).is_nan())
        .then(pl.col(target_col))  # Preserve NaN values
        .when(max_col != min_col)  # Avoid division by zero by making sure min != max
        .then((pl.col(target_col) - min_col) / (max_col - min_col) * (upper - lower) + lower)
        .otherwise(lower)
    )

    return norm_col


def winsorize(data: np.ndarray, percentile: float = 0.05, axis: int = 0) -> np.ndarray:
    """Windorize each vector of a 2D numpy array to symmetric percentiles given by `percentile`.

    This returns a Polars expression, not a DataFrame, so it can be chained (including lazily) in
    a `select` or `with_columns` invocation without needing to set a new intermediate DataFrame variable.

    Parameters
    ----------
    data: numpy array containing original data to be winsorized
    percentile: float indicating the percentiles to apply winsorization at
    axis: int indicating which axis to apply winsorization over (i.e. orientation if `dara` is 2D)

    Returns
    -------
    numpy array
    """
    try:
        if not 0 <= percentile <= 1:
            raise ValueError("`percentile` must be between 0 and 1")
    except AttributeError as e:
        raise TypeError("`percentile` must be a numeric type, such as an int or float") from e

    fin_data = np.where(np.isfinite(data), data, np.nan)

    # compute lower and upper percentiles for each column
    lower_bounds = np.nanpercentile(fin_data, percentile * 100, axis=axis, keepdims=True)
    upper_bounds = np.nanpercentile(fin_data, (1 - percentile) * 100, axis=axis, keepdims=True)

    # clip data to within the bounds
    return np.clip(data, lower_bounds, upper_bounds)


def winsorize_xsection(
    df: pl.DataFrame | pl.LazyFrame,
    data_cols: tuple[str, ...],
    group_col: str,
    percentile: float = 0.05,
) -> pl.LazyFrame:
    """Cross-sectionally winsorize the `data_cols` of `df`, grouped on `group_col`, to the symmetric percentile
    given by `percentile`.

    Parameters
    ----------
    df: Polars DataFrame or LazyFrame containing feature data to winsorize
    data_cols: collection of strings indicating the columns of `df` to be winsorized
    group_col: str column of `df` to use as the cross-sectional group
    percentile: float value indicating the symmetric winsorization threshold

    Returns
    -------
    Polars DataFrame or LazyFrame
    """

    def winsorize_group(group: pl.DataFrame) -> pl.DataFrame:
        for col in data_cols:
            winsorized_data = winsorize(group[col].to_numpy(), percentile=percentile)
            group = group.with_columns(pl.Series(col, winsorized_data).alias(col))
        return group

    try:
        result = df.lazy().group_by(group_col).map_groups(winsorize_group, schema=df.collect_schema())
        return result if isinstance(df, pl.LazyFrame) else result.collect()
    except AttributeError as e:
        raise TypeError(
            "`df` must be a Polars DataFrame or LazyFrame, but it's missing `group_by`, `map_groups` "
            "and `collect_schema` attributes"
        ) from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(f"`df` must have `data_cols` {data_cols} and `group_col` '{group_col}'") from e


def percentiles_xsection(
    target_col: str,
    over_col: str,
    lower_pct: float,
    upper_pct: float,
    fill_val: float | int = 0.0,
) -> pl.Expr:
    """Cross-sectionally mark all values of `target_col` that fall outside the `lower_pct` percentile or
    `upper_pct` percentile, within each `over_col` group. This is essentially an anti-winsorization, suitable for
    building high - low groups. The `fill_val` is inserted to each value between the percentile cutoffs.

    This returns a Polars expression, so it can be chained in a `select` or `with_columns` invocation
    without needing to set a new intermediate DataFrame or materialize lazy evaluation.

    Parameters
    ----------
    target_col: str column name to have non-percentile thresholded values masked
    over_col: str column name to apply masking over, cross-sectionally
    lower_pct: float lower percentile under which to keep values
    upper_pct: float upper percentile over which to keep values
    fill_val: numeric value for masking

    Returns
    -------
    Polars Expr
    """
    return (
        pl.when(
            (pl.col(target_col) <= pl.col(target_col).drop_nans().quantile(lower_pct).over(over_col))
            | (pl.col(target_col) >= pl.col(target_col).drop_nans().quantile(upper_pct).over(over_col))
        )
        .then(pl.col(target_col))
        .otherwise(fill_val)
    )


def exp_weights(window: int, half_life: int) -> np.ndarray:
    """Generate exponentially decaying weights over `window` trailing values, decaying by half each `half_life` index.

    Parameters
    ----------
    window: integer number of points in the trailing lookback period
    half_life: integer decay rate

    Returns
    -------
    1D numpy array with shape (window,)
    """
    try:
        assert isinstance(window, int)
        if not window > 0:
            raise ValueError("`window` must be a strictly positive integer")
    except (AttributeError, AssertionError) as e:
        raise TypeError("`window` must be an integer type") from e
    try:
        assert isinstance(half_life, int)
        if not half_life > 0:
            raise ValueError("`half_life` must be a strictly positive integer")
    except (AttributeError, AssertionError) as e:
        raise TypeError("`half_life` must be an integer type") from e
    decay = np.log(2) / half_life
    return np.exp(-decay * np.arange(window))[::-1]


# TODO: test
def ledoit_wolf_covariance(X: np.ndarray) -> tuple[float | int, np.ndarray]:
    """Estimate the covariance matrix of `X` via standard Ledoit-Wolf shrinkage.

    Parameters
    ----------
    X : array-like input data matrix for which to estimate covariance, having shape (n_samples, m_features)

    Returns
    -------
    shrinkage: float estimated shrinkage parameter.
    shrunk_cov: numpy ndarray estimated shrunk covariance matrix having shape (n_features, n_features)
    """
    n, m = X.shape

    # Center the data
    X = X - X.mean(axis=0)

    # Estimate sample covariance
    sample_cov = np.dot(X.T, X) / n

    # Calculate the squared Frobenius norm of sample covariance
    f_norm2 = np.sum(sample_cov**2)

    # Estimate of tr(sigma^2) / p
    mu = np.trace(sample_cov) / m

    # Estimate of tr((X^T X)^2) / (n_samples^2 p)
    alpha = (n / (m * (n - 1) ** 2)) * (np.sum((X**2).T.dot(X**2)) / n - f_norm2 / n)

    # Estimate of tr(sigma^2) / p
    beta = (1 / (m * n)) * (np.sum(sample_cov**2) - (np.sum(sample_cov.diagonal() ** 2) / n))

    # Estimate shrinkage parameter
    shrinkage = min(alpha / beta, 1)

    # Compute shrunk covariance matrix
    shrunk_cov = (1 - shrinkage) * sample_cov + shrinkage * mu * np.eye(m)

    return shrinkage, shrunk_cov


# TODO: docstring
# TODO: test
def rolling_covariance(
    df: pl.DataFrame | pl.LazyFrame,
    window: int = 252,
    over_col: str = "date",
    group_col: str = "symbol",
    values_col: str = "asset_returns",
    shrinkage: Literal["ledoit_wolf"] | None = "ledoit_wolf",
):
    df = df.sort(over_col)

    dates = df.lazy().collect()[over_col].unique().sort()

    pivot_df = df.pivot(index=over_col, columns=group_col, values=values_col).sort(over_col)

    returns_array = pivot_df.select(pl.exclude(over_col)).to_numpy()

    results = []

    match shrinkage:
        case "ledoit_wolf":
            for i in range(window - 1, len(dates)):
                window_data = returns_array[i - window + 1 : i + 1]
                shrink, cov_mat = ledoit_wolf_covariance(window_data)

                results.append({over_col: dates[i], "cov_mat": cov_mat, "shrinkage": shrink})
        case None:
            for i in range(window - 1, len(dates)):
                window_data = returns_array[i - window + 1 : i + 1]
                cov_mat = np.cov(window_data.T)

                results.append({over_col: dates[i], "cov_mat": cov_mat, "shrinkage": None})
        case _:
            raise ValueError(f"`shrinkage` value must be 'ledoit_wolf' or None, not '{shrinkage}'")

    return results
