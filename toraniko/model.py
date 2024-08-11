"""Complete implementation of the factor model."""

import numpy as np
import polars as pl
import polars.exceptions as pl_exc

from toraniko.math import winsorize


def factor_returns_cs(
    returns: np.ndarray,
    mkt_caps: np.ndarray,
    sector_scores: np.ndarray,
    style_scores: np.ndarray,
    residualize_styles: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate market, sector, style and residual asset returns for one time period, robust to rank deficiency.

    The risk model formulation is:

        r_t = alpha_t + Beta_t x f_t + epsilon_t

    where
        - r_t is an n-vector of all assets' respective excess returns at time period t,
        - alpha_t is an n-vector of the expected returns of each asset,
        - Beta_t is an n x m matrix of factor scores,
        - f_t is an m-vector consisting of the returns of each factor at time period t,
        - epsilon_t is an n-vector of the error, or idiosyncratic returns

    This function is evaluated once for each time period t. With r_t and Beta_t known (parameterized by `returns` and
    [1, `sector_scores`, `style_scores`] respectively), we seek to estimate f_t. Our problem to solve is:

        min(r_t - Beta_t x f_t) Sigma^-1 (r_t - Beta_t x f_t)

    where Sigma^-1 is the inverse of the population idiosyncratic covariance of the asset returns. Given we don't know
    the true population idiosyncratic covariance, we instead take the convention established by Barra/Axioma and use
    the square root of the market caps, W:

        min(r_t - Beta_t x f_t) Sigma^-1 (r_t - Beta_t x f_t)

    Beta_t is actually [Beta_sector, Beta_style], with Beta_sector being [1, `sector_scores`]: the market scores are
    a column of ones representing inclusion in the market for each asset under consideration, and the market betas
    are spanned by the sector betas. In other words, any asset's exposure to the market can be precisely replicated via
    linear combination of its sector exposures. To see this, consider that the row for any asset's market + sector
    factor scores is a 1 for the market inclusion, followed by 0, 0, ..., 1, ..., 0, 0, where the only other 1 is
    in the sector membership and all 0s are in extraneous sectors. So, every Beta_sector row will have exactly two 1s:
    the market and the sector, and so the market is trivially a linear combination with 1 = 0 + 0 + ... + 1 + ... 0 + 0.
    Algebraically this means Beta_sector is rank-deficient and thus the solution space underdetermined.

    To resolve this add a constraint forcing the sector returns to sum to 0, which economically has the interpretation
    that the market return is the sum of all sector returns, and sector returns are relative to the market.

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
    mkt_factor_col: str = "market",
    res_ret_col: str = "res_asset_returns",
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Estimate factor and residual returns across all time periods using input asset factor scores.

    See the documentation under `_factor_returns` for a complete mathematical specification of the model and
    implementation considerations. The `_factor_returns` function is the complete implementation of the model for
    a single time period's cross-section. This function is a wrapper that obtains the timeseries of rolling factor
    returns from calling that function once in each time period. It also cleans up the output and returns it from
    numpy array objects into Polars DataFrame objects for convenience.

    Parameters
    ----------
    returns_df: Polars DataFrame containing | date | symbol | asset_returns |
    mkt_cap_df: Polars DataFrame containing | date | symbol | market_cap |
    sector_df: Polars DataFrame containing | date | symbol | followed by one column for each sector
    style_df: Polars DataFrame containing | date | symbol | followed by one column for each style
    winsor_factor: optional float indicating the symmetric percentile at which winsorization should be applied
    residualize_styles: bool indicating if style returns should be orthogonalized to market + sector returns
    asset_returns_col: str name of the column we expect to find asset return values in, defaults to "asset_returns"
    mkt_cap_col: str name of the column we expect to find market cap values in, defaults to "market_cap"
    symbol_col: str name of the column we expect to find symbol names in, defaults to "symbol"
    date_col: str name of the column we expect to find time periods in, defaults to "date"
    mkt_factor_col: str name to use for the column containing returned market factor, defaults to "market"
    res_ret_col: str name to use for the column containing asset residual returns, defaults to "res_asset_returns"

    Returns
    -------
    tuple of Polars DataFrames melted by date: (factor returns, residual returns)
    """
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
    else:
        try:
            returns_df = (
                returns_df.join(mkt_cap_df, on=[date_col, symbol_col])
                .join(sector_df, on=[date_col, symbol_col])
                .join(style_df, on=[date_col, symbol_col])
            )
            # split the conditional winsorization branch into two functions, so we don't have a conditional
            # needlessly evaluated on each iteration of the `.map_groups`
            if winsor_factor is not None:

                def _estimate_factor_returns(data):
                    r = winsorize(data[asset_returns_col].to_numpy())
                    fac, eps = factor_returns_cs(
                        r,
                        data[mkt_cap_col].to_numpy(),
                        data.select(sectors).to_numpy(),
                        data.select(styles).to_numpy(),
                        residualize_styles,
                    )
                    return (
                        # reshape so we get a row vector instead of a column vector for the DataFrame
                        pl.DataFrame(fac.reshape(1, -1), schema=[mkt_factor_col] + sectors + styles)
                        # add back the time period group to disambiguate
                        .with_columns(pl.lit(data[date_col].head(1).item()).cast(pl.Date).alias(date_col)).with_columns(
                            pl.lit(eps.tolist()).alias(res_ret_col),
                            pl.lit(data[symbol_col].to_list()).alias(symbol_col),
                        )
                    )

            else:

                def _estimate_factor_returns(data):
                    fac, eps = factor_returns_cs(
                        data[asset_returns_col].to_numpy(),
                        data[mkt_cap_col].to_numpy(),
                        data.select(sectors).to_numpy(),
                        data.select(styles).to_numpy(),
                        residualize_styles,
                    )
                    return (
                        # reshape so we get a row vector instead of a column vector for the DataFrame
                        pl.DataFrame(fac.reshape(1, -1), schema=[mkt_factor_col] + sectors + styles)
                        # add back the time period group to disambiguate
                        .with_columns(pl.lit(data[date_col].head(1).item()).cast(pl.Date).alias(date_col)).with_columns(
                            pl.lit(eps.tolist()).alias(res_ret_col),
                            pl.lit(data[symbol_col].to_list()).alias(symbol_col),
                        )
                    )

            fac_df = returns_df.group_by(date_col).map_groups(_estimate_factor_returns)
            eps_df = fac_df[[date_col, symbol_col, res_ret_col]].explode([symbol_col, res_ret_col])
            return fac_df.drop([symbol_col, res_ret_col]), eps_df
        except AttributeError as e:
            raise TypeError(
                "`returns_df` and `mkt_cap_df` must be Polars DataFrames, but there are missing attributes"
            ) from e
        except pl_exc.ColumnNotFoundError as e:
            raise ValueError(
                f"`returns_df` must have columns '{date_col}', '{symbol_col}' and '{asset_returns_col}'; "
                f"`mkt_cap_df` must have '{date_col}', '{symbol_col}' and '{mkt_cap_col}' columns"
            ) from e
