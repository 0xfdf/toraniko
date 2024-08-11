"""Complete implementation of the factor model."""

import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from toraniko.math import winsorize


def _factor_returns(
    returns: np.ndarray,
    mkt_caps: np.ndarray,
    sector_scores: np.ndarray,
    style_scores: np.ndarray,
    residualize_styles: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate market, sector, style and residual asset returns for one time period, robust to rank deficiency.

    Parameters
    ----------
    returns: np.ndarray returns of the assets (shape n_assets x 1)
    mkt_caps: np.ndarray of asset market capitalizations (shape n_assets x 1)
    sector_scores: np.ndarray of asset scores used to estimate the sector return (shape n_assets x m_sectors)
    style_scores: np.ndarray of asset scores used to estimate style factor returns (shape n_assets x m_styles)
    residualize_styles: bool indicating if styles should be orthogonalized to market + sector

    Returns
    -------
    tuple of arrays: (market/sector/style factor returns, residual returns)
    """
    # Proxy for the inverse of asset idiosyncratic variances
    W = np.diag(np.sqrt(mkt_caps.ravel()))

    # Estimate sector factor returns with a constraint that the sector factors sum to 0
    # Economically, we assert that the market return is completely spanned by the sector returns
    beta_sector = np.hstack([np.ones(returns.shape[0]).reshape(-1, 1), sector_scores])
    m_sectors = sector_scores.shape[1]
    a = np.concatenate([np.array([0]), (-1 * np.ones(m_sectors - 1))])
    Imat = np.identity(m_sectors)
    R_sector = np.vstack([Imat, a])
    # Change of variables to add the constraint
    B_sector = beta_sector @ R_sector

    V_sector, _, _, _ = np.linalg.lstsq(B_sector.T @ W @ B_sector, B_sector.T @ W, rcond=None)
    # Change of variables to recover all sectors
    g = V_sector @ returns
    fac_ret_sector = R_sector @ g

    sector_resid_returns = returns - (B_sector @ g)

    # Estimate style factor returns without sector constraint
    V_style, _, _, _ = np.linalg.lstsq(style_scores.T @ W @ style_scores, style_scores.T @ W, rcond=None)
    if residualize_styles:
        fac_ret_style = V_style @ sector_resid_returns
    else:
        fac_ret_style = V_style @ returns

    # Combine factor returns
    fac_ret = np.concatenate([fac_ret_sector, fac_ret_style])

    # Calculate final residuals
    epsilon = sector_resid_returns - (style_scores @ fac_ret_style)

    return fac_ret, epsilon


def estimate_factor_returns(
    returns_df: pl.DataFrame,
    mkt_cap_df: pl.DataFrame,
    sector_df: pl.DataFrame,
    style_df: pl.DataFrame,
    winsor_factor: float | None = 0.05,
    residualize_styles: bool = True,
    asset_returns_col: str = "asset_returns",
    mkt_cap_col: str = "market_cap",
    symbol_col: str = "symbol",
    date_col: str = "date",
) -> tuple[pl.DataFrame, pl.DataFrame] | pl.DataFrame:
    """Estimate factor and residual returns across all time periods using input asset factor scores.

    Parameters
    ----------
    returns_df: Polars DataFrame containing | date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame containing | date | symbol | market_cap |
    sector_df: Polars DataFrame containing | date | symbol | followed by one column for each sector
    style_df: Polars DataFrame containing | date | symbol | followed by one column for each style
    winsor_factor: winsorization proportion
    residualize_styles: bool indicating if style returns should be orthogonalized to market + sector returns
    asset_returns_col: str name of the column we expect to find asset return values in, defaults to "asset_returns"
    mkt_cap_col: str name of the column we expect to find market cap values in, defaults to "market_cap"
    symbol_col: str name of the column we expect to find symbol names in, defaults to "symbol"
    date_col: str name of the column we expect to find time periods in, defaults to "date"

    Returns
    -------
    tuple of Polars DataFrames melted by date: (factor returns, residual returns)
    """
    returns, residuals = [], []
    try:
        sectors = sorted(sector_df.select(pl.exclude(date_col, symbol_col)).columns)
    except AttributeError as e:
        raise TypeError("`sector_df` must be a Polars DataFrame, but it's missing required attributes") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(
            f"`sector_df` must have columns for '{date_col}' and '{symbol_col}' in addition to each sector"
        ) from e
    try:
        styles = sorted(style_df.select(pl.exclude(date_col, symbol_col)).columns)
    except AttributeError as e:
        raise TypeError("`style_df` must be a Polars DataFrame, but it's missing required attributes") from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(
            f"`style_df` must have columns for '{date_col}' and '{symbol_col}' in addition to each style"
        ) from e
    try:
        returns_df = (
            returns_df.join(mkt_cap_df, on=[date_col, symbol_col])
            .join(sector_df, on=[date_col, symbol_col])
            .join(style_df, on=[date_col, symbol_col])
        )
        dates = returns_df[date_col].unique().to_list()
        # iterate through, one day at a time
        # this could probably be made more efficient with Polars' `.map_groups` method
        for dt in dates:
            ddf = returns_df.filter(pl.col(date_col) == dt).sort(symbol_col)
            r = ddf[asset_returns_col].to_numpy()
            if winsor_factor is not None:
                r = winsorize(r, winsor_factor)
            f, e = _factor_returns(
                r,
                ddf[mkt_cap_col].to_numpy(),
                ddf.select(sectors).to_numpy(),
                ddf.select(styles).to_numpy(),
                residualize_styles,
            )
            returns.append(f)
            residuals.append(dict(zip(ddf[symbol_col].to_list(), e)))
    except AttributeError as e:
        raise TypeError(
            "`returns_df` and `mkt_cap_df` must be Polars DataFrames, but there are missing attributes"
        ) from e
    except pl_exc.ColumnNotFoundError as e:
        raise ValueError(
            f"`returns_df` must have columns '{date_col}', '{symbol_col}' and '{asset_returns_col}'; "
            f"`mkt_cap_df` must have '{date_col}', '{symbol_col}' and '{mkt_cap_col}' columns"
        ) from e
    ret_df = pl.DataFrame(np.array(returns), schema=["market"] + sectors + styles).with_columns(
        pl.Series(dates).alias(date_col)
    )
    eps_df = pl.DataFrame(residuals).with_columns(pl.Series(dates).alias(date_col))
    return ret_df, eps_df
