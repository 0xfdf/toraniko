"""End-to-end estimation of the style factor scores, factor returns and factor covariance."""

import configparser
from io import StringIO
import logging
from typing import Literal

import polars as pl

from toraniko.styles import factor_mom, factor_sze, factor_val
from toraniko.utils import smooth_features
from toraniko.config import load_config, init_config


# TODO: on first load, check to see if .toraniko directory exists in home directory
#   - if it does not, create it
#   - then check to see if there is a config.ini file in there; if there isn't, copy it from the sample config.ini
class FactorModel:

    # TODO: integrate any custom factors to the repr, if active
    def __repr__(self):
        return (
            f"<toraniko.FactorModel: (Custom factors: {self.enabled_custom}, Style factors: {self.enabled_styles}, "
            f"Scores estimated: {self.scores_estimated}, Returns estimated: "
            f"{self.returns_estimated}, Covariance estimated: {self.covariance_estimated})>"
        )

    def __init__(self, config_file: str | None = None, log_level: Literal["INFO", "ERROR", "DEBUG"] = "INFO"):
        """"""
        self.settings = load_config(config_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        self.style_factor_data = {}
        self.asset_returns = None
        self.market_caps = None
        self.sector_factor_scores = None
        self.style_factor_scores = None
        self.factor_returns = None
        self.residual_returns = None
        self.scores_estimated = False
        self.returns_estimated = False
        self.covariance_estimated = False
        self.enabled_styles = [
            f for f in self.settings["style_factors"] if self.settings["style_factors"][f]["enabled"]
        ]
        self.enabled_custom = [f for f in self.settings.get("custom_factors", [])]

    # TODO: improve performance, these style factors can be processed concurrently
    def _estimate_style_factor_scores(self) -> None:
        """"""
        style_factor_scores = {}
        if self.settings["style_factors"]["mom"]["enabled"]:
            style_factor_scores["mom"] = factor_mom(
                self.style_factor_data["mom"],
                asset_returns_col=self.settings["global_column_names"]["asset_returns_col"],
                symbol_col=self.settings["global_column_names"]["symbol_col"],
                date_col=self.settings["global_column_names"]["date_col"],
                **self.settings["style_factors"]["mom"],
            )
        self.style_factor_scores = style_factor_scores
        self.scores_estimated = True

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
        if self.settings["model_estimation"]["mkt_cap_smooth_window"] is not None:
            mkt_caps = smooth_features(
                mkt_caps,
                (self.settings["global_column_names"]["mkt_cap_col"],),
                self.settings["global_column_names"]["date_col"],
                self.settings["global_column_names"]["symbol_col"],
                self.settings["model_estimation"]["mkt_cap_smooth_window"],
            )
        if self.settings["style_factors"]["mom"]["enabled"]:
            self.style_factor_data["mom"] = asset_returns
        if self.settings["style_factors"]["sze"]["enabled"]:
            self.style_factor_data["sze"] = mkt_caps
        # if self.settings["style_factors"]["val"]["enabled"]:
        #     if valuations is None:
        #         raise ValueError(
        #             "`style_factors.val.enabled` is set to true in config, but no `valuations` were passed"
        #         )
        #     self.style_factor_data["val"] = valuations
        self._estimate_style_factor_scores()
