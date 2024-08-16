"""End-to-end estimation of the style factor scores, factor returns and factor covariance."""

from functools import reduce
import logging
from typing import Callable, Iterable, Literal

import polars as pl
import polars.exceptions as pl_exc

from toraniko.config import load_config
from toraniko.math import winsorize_xsection
from toraniko.model import estimate_factor_returns
from toraniko import styles
from toraniko.utils import fill_features, smooth_features, top_n_by_group

# TODO: consider turning the config into kwargs so it can be run on its own, not just driven by config
# TODO: the ergonomics of this really need to be solid

# TODO: before factor score return estimation
# Step 6: Estimate rolling Ledoit-Wolf shrunk asset covariance, or create rolling market cap weight matrix

# Step 8: Optionally re-estimate loadings via timeseries regression of each asset on market, sector and custom + style factors

# Step 9: Estimate rolling Ledoit-Wolf + STFU factor covariance


class FactorModel:

    # TODO: add settings and diagnostics here
    def __repr__(self):
        return f"<toraniko.FactorModel>"

    def __init__(
        self,
        feature_data: pl.DataFrame | pl.LazyFrame,
        sector_encodings: pl.DataFrame | pl.LazyFrame,
        log_level: Literal["INFO", "ERROR", "DEBUG"] = "INFO",
        symbol_col: str = "symbol",
        date_col: str = "date",
        mkt_cap_col: str = "market_cap",
    ):
        """"""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.symbol_col = symbol_col
        self.date_col = date_col
        self.mkt_cap_col = mkt_cap_col
        self.feature_data = feature_data
        self.sector_encodings = sector_encodings
        self.filled_features = None
        self.smoothed_features = None
        self.top_n_mkt_cap = None
        self.style_factors = {}
        self.style_df = None
        self.weight_df = None
        self.factor_returns = None
        self.residual_returns = None

    # TODO: optimize this: we shouldn't have to loop over features in to_winsorize or to_smooth, it's wasteful
    def clean_features(
        self,
        to_winsorize: dict[str:float] | None,
        to_fill: list[str] | tuple[str, ...] | None,
        to_smooth: dict[str:int] | None,
    ) -> None:
        """"""
        if to_winsorize is not None:
            for feature in to_winsorize:
                self.feature_data = winsorize_xsection(
                    self.feature_data, (feature,), self.date_col, to_winsorize[feature]
                )
        if to_fill is not None:
            self.feature_data = fill_features(self.feature_data, to_fill, self.date_col, self.symbol_col)
            self.filled_features = to_fill
        if to_smooth is not None:
            for feature in to_smooth:
                self.feature_data = smooth_features(
                    self.feature_data, (feature,), self.date_col, self.symbol_col, to_smooth[feature]
                )
            self.smoothed_features = to_smooth

    def reduce_universe_by_market_cap(self, top_n: int | None = 2000, collect: bool = True) -> None:
        """"""
        if top_n is not None:
            self.feature_data = top_n_by_group(self.feature_data, top_n, self.mkt_cap_col, (self.date_col,), True)
            self.top_n_mkt_cap = top_n
        if collect:
            self.feature_data = self.feature_data.collect()

    # TODO: figure out multiprocessing or multithreading, and implement it for a meaningful speed up
    def estimate_style_scores(
        self,
        style_factor_funcs: Iterable[Callable],
        style_factor_kwargs: Iterable[dict],
        collect: bool = True,
    ) -> None:

        df = self.feature_data.lazy()

        frames = []

        for f, kwargs in zip(style_factor_funcs, style_factor_kwargs):

            frames.append(f(df, **kwargs))

        self.style_df = reduce(
            lambda left, right: left.join(
                right,
                on=[self.date_col, self.symbol_col],
                how="inner",
            ),
            frames,
        )

        if collect:
            self.style_df = self.style_df.collect()

    def proxy_idio_cov(self, method: Literal["market_cap", "ledoit_wolf"] = "market_cap") -> None:
        match method:
            case "ledoit_wolf":
                raise NotImplementedError("Not implemented yet!")
            case "market_cap":
                self.weight_df = self.feature_data.select(self.date_col, self.symbol_col, self.mkt_cap_col)
            case _:
                raise ValueError(f"'{method}' is not a valid option for `proxy_for_idio_cov`")

    def estimate_factor_returns(
        self,
        winsor_factor: float = 0.01,
        residualize_styles: bool = False,
        asset_returns_col: str = "asset_returns",
        mkt_factor_col: str = "Market",
        res_ret_col: str = "res_asset_returns",
    ) -> None:
        """"""
        self.factor_returns, self.residual_returns = estimate_factor_returns(
            self.feature_data.select(self.date_col, self.symbol_col, asset_returns_col),
            self.weight_df,
            self.sector_encodings,
            self.style_df,
            winsor_factor,
            residualize_styles,
            asset_returns_col,
            self.mkt_cap_col,
            self.symbol_col,
            self.date_col,
            mkt_factor_col,
            res_ret_col,
        )
