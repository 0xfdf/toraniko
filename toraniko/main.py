"""End-to-end estimation of the style factor scores, factor returns and factor covariance."""

import configparser
from io import StringIO
import logging
from typing import Literal

import polars as pl

from toraniko.utils import create_dummy_variables


class FactorModel:

    def __repr__(self):
        return f"<Toraniko.FactorModel: {self.str_config}>"

    def __init__(self, config_file: str, log_level: Literal["info", "error", "debug"] = "info"):
        """"""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.str_config = StringIO()
        self.config.write(self.str_config)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def _estimate_style_factor_scores(self) -> None:
        """"""

    def _estimate_factor_returns(self) -> None:
        """"""

    def _estimate_factor_cov(self) -> None:
        """"""

    def estimate(self, sector_scores: pl.DataFrame) -> None:
        """"""
