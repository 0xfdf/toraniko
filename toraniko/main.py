"""End-to-end estimation of the style factor scores, factor returns and factor covariance."""

import configparser
import logging

from typing import Literal


class FactorModel:

    def __init__(self, config_file: str, log_level: Literal["info", "error", "debug"] = "info") -> None:
        """"""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    def _estimate_style_factor_scores(self) -> None:
        """"""

    def _estimate_factor_returns(self) -> None:
        """"""

    def _estimate_factor_cov(self) -> None:
        """"""

    def estimate(self) -> None:
        """"""
