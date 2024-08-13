"""End-to-end estimation of the style factor scores, factor returns and factor covariance."""

import logging
from functools import reduce
from timeit import default_timer as timer
from typing import Literal

import polars as pl
import polars.exceptions as pl_exc

from toraniko.config import load_config
from toraniko.model import estimate_factor_returns
from toraniko.styles import factor_mom, factor_sze, factor_val
from toraniko.utils import create_dummy_variables, fill_features, smooth_features, top_n_by_group


class FactorModel:

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
        self.enabled_custom = [f for f in self.enabled_custom if self.settings["custom_factors"][f]["enabled"]]

    # TODO: should be a standalone function accessible outside this class, with testing
    def _validate_features(self) -> None:
        """"""
        try:
            if not all(c in self.feature_data.columns for c in self.settings["global_column_names"].values()):
                raise ValueError(
                    "`feature_data` appears to be misisng one or more of the required columns "
                    "specified by your config's 'global_column_names' section"
                )
        except AttributeError as e:
            raise TypeError(
                "`feature_data` must be a Polars DataFrame, but it's missing the `columns` attribute"
            ) from e
        if self.settings["model_estimation"]["mkt_cap_col"] not in self.feature_data.columns:
            raise ValueError("`feature_data` is missing the 'mkt_cap_col' specified by your config file")
        if self.settings["style_factor.val"]["enabled"]:
            if not all(
                c in self.feature_data.columns
                for k in ["bp_col", "sp_col", "cf_col"]
                for c in self.settings["style_factor.val"][k]
            ):
                raise ValueError(
                    "`feature_data` is missing one or more required columns specified in "
                    "your config's 'style_factors.val' section, and the value factor is enabled"
                )

    def _validate_sector_scores(self) -> None:
        """"""
        try:
            if not all(
                c in self.sector_encodings.columns
                for c in [
                    self.settings["global_column_names"]["date_col"],
                    self.settings["global_column_names"]["symbol_col"],
                ]
            ):
                raise ValueError(
                    "`sector_encodings` is missing the required 'date_col' and 'symbol_col' as specified "
                    "by your config file"
                )
        except AttributeError as e:
            raise TypeError(
                "`sector_encodings` must be a Polars DataFrame, but it's missing the columns attribute"
            ) from e
        if self.settings["model_estimation"]["make_sector_dummies"]:
            self.sector_encodings = create_dummy_variables(
                self.sector_encodings,
                self.settings["global_column_names"]["symbol_col"],
                self.settings["global_column_names"]["sectors_col"],
            )

    def _fill_features(self) -> None:
        """"""
        # optional
        # NB: this does not obviate the requirement to know your data! do not use this carelessly!
        features = self.settings["model_estimation"]["clean_features"]
        if features is not None:
            try:
                self.feature_data = fill_features(
                    self.feature_data,
                    features,
                    self.settings["global_column_names"]["date_col"],
                    self.settings["global_column_name"]["symbol_col"],
                )
            except pl_exc.ColumnNotFoundError as e:
                raise ValueError(
                    "`feature_data` is due to have features filled per the config, but `feature_data` is missing "
                    "one or more of the column names specified in the config under 'model_estimation.clean_features'"
                ) from e

    def _smooth_market_caps(self) -> None:
        """"""
        window = self.settings["model_estimation"]["mkt_cap_smooth_window"]
        if window is not None:
            self.feature_data = smooth_features(
                self.feature_data,
                (self.settings["global_column_names"]["mkt_cap_col"],),
                self.settings["global_column_names"]["date_col"],
                self.settings["global_column_names"]["symbol_col"],
                window,
            )

    def _restrict_universe(self) -> None:
        """"""
        top_n = self.settings["model_estimation"]["top_n_by_mkt_cap"]
        if top_n is not None:
            self.feature_data = top_n_by_group(
                self.feature_data,
                top_n,
                self.settings["global_column_names"]["mkt_cap_col"],
                self.settings["global_column_names"]["date_col"],
                True,
            )

    # TODO: improve performance, these style factors can be processed concurrently
    # TODO: this should be a standalone function available outside the class, and tested
    def _estimate_style_factor_scores(self) -> None:
        """"""
        style_factor_scores = {}
        if self.settings["style_factors"]["mom"]["enabled"]:
            style_factor_scores["mom"] = factor_mom(
                self.style_factor_data["mom"],
                **self.settings["global_column_names"],
                **self.settings["style_factors"]["mom"],
            )
        if self.settings["style_factors"]["sze"]["enabled"]:
            style_factor_scores["sze"] = factor_sze(
                self.style_factor_data["sze"],
                **self.settings["global_column_names"],
                **self.settings["style_factors"]["sze"],
            )
        if self.settings["style_factors"]["val"]["enabled"]:
            style_factor_scores["val"] = factor_val(
                self.style_factor_data["val"],
                **self.settings["global_column_names"],
                **self.settings["style_factors"]["val"],
            )
        self.style_factor_scores = style_factor_scores
        self.scores_estimated = True

    def _estimate_factor_returns(self) -> None:
        """"""
        returns_df = self.feature_data.select(
            self.settings["global_column_names"]["date_col"],
            self.settings["global_column_names"]["symbol_col"],
            self.settings["global_column_names"]["asset_returns_col"],
        )
        mkt_cap_df = self.feature_data.select(
            self.settings["global_column_names"]["date_col"],
            self.settings["global_column_names"]["symbol_col"],
            self.settings["global_column_names"]["mkt_cap_col"],
        )
        sector_df = self.feature_data.select(
            self.settings["global_column_names"]["date_col"], self.settings["global_column_names"]["symbol_col"]
        ).join(self.sector_encodings, on=self.settings["global_column_names"]["date_col"])
        style_df = result_df = reduce(
            lambda left, right: left.join(
                right,
                on=[
                    self.settings["global_column_names"]["date_col"],
                    self.settings["global_column_names"]["symbol_col"],
                ],
                how="inner",
            ),
            self.style_factor_scores.values(),
        )
        self.factor_returns, self.residual_returns = estimate_factor_returns(
            returns_df,
            mkt_cap_df,
            sector_df,
            style_df,
            self.settings["model_estimation"]["winsor_factor"],
            self.settings["model_estimation"]["residualize_styles"],
            self.settings["global_column_names"]["asset_returns_col"],
            self.settings["global_column_names"]["mkt_cap_col"],
            self.settings["global_column_names"]["symbol_col"],
            self.settings["global_column_names"]["date_col"],
            self.settings["model_estimation"]["mkt_factor_col"],
            self.settings["model_estimation"]["res_ret_col"],
        )
        self.returns_estimated = True

    def _estimate_factor_cov(self) -> None:
        """"""

    # docstring NB: `sector_scores` must be in dummy variables, if this is not the case, use `utils.create_dummy_variables`
    # docstring NB: `asset_returns` should already be cleaned of nulls and nans, if this is not the case, use `utils.fill_features`
    # docstring NB: `mkt_caps` should already be cleaned of nulls and nans, if this is not the case, use `utils.fill_features`
    # TODO: make both inputs lazy
    def estimate(
        self,
        feature_data: pl.DataFrame,
        sector_encodings: pl.DataFrame,
    ) -> None:
        """"""
        # TODO: validate sector_scores
        # setup initial data and time tracking
        self.est_start_time = timer()
        self.sector_encodings = sector_encodings
        self.feature_data = feature_data
        # validate inputs
        self._validate_features()
        self._validate_sector_scores()
        # forward fill features, optionally
        self._fill_features()
        # smooth market caps, optionally
        self._smooth_market_caps()
        # restrict the universe
        self._restrict_universe()
        # estimate style factor scores
        self._estimate_style_factor_scores()
        # estimate factor returns
        self._estimate_factor_returns()
