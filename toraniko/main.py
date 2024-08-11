"""End-to-end estimation of the style factor scores, factor returns and factor covariance."""

import configparser
from io import StringIO
import logging
from typing import Literal

import polars as pl

from toraniko.styles import factor_mom, factor_sze, factor_val
from toraniko.utils import smooth_features


class FactorModel:

    def __repr__(self):
        return f"<Toraniko.FactorModel: {self.str_config}>"

    def __init__(self, config_file: str, log_level: Literal["info", "error", "debug"] = "info"):
        """"""
        self.load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.style_factor_data = {}
        self.asset_returns = None
        self.market_caps = None
        self.sector_factor_scores = None
        self.style_factor_scores = None
        self.factor_returns = None
        self.residual_returns = None

    def load_config(self, config_file) -> None:
        """"""
        try:
            self.config = configparser.ConfigParser()
            self.config.read(config_file)
        except FileNotFoundError as e:
            raise ValueError(f"Failed to read `config_file` '{config_file}'; does it exist in that location?") from e
        try:
            self.asset_returns_col = self.config["global_column_names"]["asset_returns_col"]
            self.symbol_col = self.config["global_column_names"]["symbol_col"]
            self.date_col = self.config["global_column_names"]["date_col"]
            self.mkt_cap_col = self.config["global_column_names"]["market_cap"]
        except KeyError as e:
            raise ValueError(
                "Failed to set global column names; your config file must have a section "
                "'global_column_names' with string values for 'asset_returns_col', 'symbol_col', "
                "'date_col' and 'market_cap'"
            ) from e
        try:
            self.model_estimation_settings = {
                "winsor_factor": float(self.config["model_estimation"]["winsor_factor"]),
                "residualize_styles": self.config["model_estimation"]["residualize_styles"].lower() == "true",
                "mkt_factor_col": self.config["model_estimation"]["mkt_factor_col"],
                "res_ret_col": self.config["model_estimation"]["res_ret_col"],
                "top_n_by_mkt_cap": int(self.config["model_estimation"]["top_n_by_mkt_cap"]),
            }
        except (KeyError, ValueError) as e:
            raise ValueError(
                "Failed to set model estimation settings; your config gile must have a section "
                "'model_estimation' with values for 'winsor_factor' (float), 'residualize_styles' (string bool) "
                "'mkt_factor_col' (string), 'res_ret_col', (string), 'top_n_by_mkt_cap' (int), 'mkt_cap_smooth_window' "
                "(int)"
            ) from e
        self.str_config = StringIO()
        self.config.write(self.str_config)

    # TODO: improve performance, these style factors can be processed concurrently
    def _estimate_style_factor_scores(self) -> None:
        """"""

    def _estimate_factor_returns(self) -> None:
        """"""

    def _estimate_factor_cov(self) -> None:
        """"""

    # docstring NB: `sector_scores` must be in dummy variables, if this is not the case, use `utils.create_dummy_variables`
    # docstring NB: `asset_returns` should already be cleaned of nulls and nans, if this is not the case, use `utils.fill_features`
    # docstring NB: `mkt_caps` should already be cleaned of nulls and nans, if this is not the case, use `utils.fill_features`
    def estimate(
        self,
        sector_scores: pl.DataFrame,
        asset_returns: pl.DataFrame,
        mkt_caps: pl.DataFrame,
        valuations: pl.DataFrame | None = None,
    ) -> None:
        """"""
        # TODO: validate sector_scores
        self.sector_factor_scores = sector_scores
        if self.config["model_estimation"]["mkt_cap_smooth_window"] is not None:
            mkt_caps = smooth_features(
                mkt_caps,
                (self.config["global_column_names"],),
                self.config["date_col"],
                self.config["symbol_col"],
                self.config["mkt_cap_smooth_window"],
            )
        if self.config["style_factors.mom"]["enabled"]:
            self.style_factor_data["mom"] = asset_returns
        if self.config["style_factors.sze"]["enabled"]:
            self.style_factor_data["sze"] = mkt_caps
        if self.config["style_factors.val"]["enabled"]:
            if valuations is None:
                raise ValueError(
                    "`style_factors.val.enabled` is set to true in config, but no `valuations` were passed"
                )
            self.style_factor_data["val"] = valuations
        self._estimate_style_factor_scores()
        # TODO: implement custom factor logic
