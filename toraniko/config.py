"""Functions related to config handling."""

from configparser import ConfigParser
import json
import logging
import os
from pkg_resources import resource_filename
import shutil

logger = logging.getLogger(__name__)


# TODO: needs testing
def init_config() -> None:
    """Initialize the project's config directory and create a sample config.ini file if one doesn't already exist.

    If the directory already exists with a file called "config.ini" in it, this function does nothing.
    """
    toraniko_dir = os.path.expanduser("~/.toraniko")
    if not os.path.exists(toraniko_dir):
        logger.info(f"INFO: Creating project config directory: {toraniko_dir}")
        os.mkdir(toraniko_dir)

    config_file = f"{toraniko_dir}/config.ini"
    if not os.path.exists(config_file):
        logger.info(
            f"INFO: Config does not exist in {toraniko_dir}; initializing a sample config.ini file at {config_file}"
        )
        sample_config = resource_filename("toraniko", "config.ini")
        shutil.copy(sample_config, config_file)
    else:
        logger.warning(
            f"WARNING: Config already exists at {config_file}; nothing to do. Consider "
            f"backing up the existing file to backup_config.ini and rerunning if you want a fresh config."
        )


# TODO: needs testing
def load_config(config_file: str | None) -> dict:
    """Load a .ini config file according to an expected specification, and transform each of the settings into correct
    types. For example, "null" -> None, "3.14" -> 3.14, "true" -> True, etc.

    Then the `config_file` is none, the function loads the config file located in the project's config directory
    in the user's home directory.

    Parameters
    ----------
    config_file: optional str filename location to look for the config_file, including directory.

    Returns
    -------
    dictionary containing full settings mimicking the config file, with values transformed to their proper Python types
    """
    if config_file is None:
        config_file = os.path.expanduser(f"~/.toraniko/config.ini")
    try:
        config = ConfigParser()
        config.read(config_file)
    except FileNotFoundError as e:
        raise ValueError(
            f"Failed to read `config_file` '{config_file}'; does it exist in that location? Running toraniko-init from "
            "the command line should install a default config file at ~/.toraniko/config.ini. Alternatively, try "
            "running `toraniko.config.init_config()."
        ) from e
    settings = {}
    try:
        settings["global_column_names"] = {
            "asset_returns_col": config["global_column_names"]["asset_returns_col"],
            "symbol_col": config["global_column_names"]["symbol_col"],
            "date_col": config["global_column_names"]["date_col"],
            "mkt_cap_col": config["global_column_names"]["mkt_cap_col"],
            "sectors_col": config["global_column_names"]["sectors_col"],
        }
    except KeyError as e:
        raise ValueError(
            "Failed to set global column names; your config file must have a section "
            "'global_column_names' with string values for 'asset_returns_col', 'symbol_col', "
            "'date_col' and 'mkt_cap_col'. These control the global column names used by the library functions."
        ) from e
    try:
        settings["model_estimation"] = {
            "winsor_factor": float(config["model_estimation"]["winsor_factor"]),
            "residualize_styles": config["model_estimation"]["residualize_styles"].lower() == "true",
            "mkt_factor_col": config["model_estimation"]["mkt_factor_col"],
            "res_ret_col": config["model_estimation"]["res_ret_col"],
            "make_sector_dummies": config["model_estimation"]["make_sector_dummies"].lower() == "true",
        }
        if config["model_estimation"]["top_n_by_mkt_cap"].lower() != "null":
            settings["model_estimation"]["top_n_by_mkt_cap"] = int(config["model_estimation"]["top_n_by_mkt_cap"])
        else:
            settings["model_estimation"]["top_n_by_mkt_cap"] = None
        if config["model_estimation"]["clean_features"].lower() != "null":
            settings["model_estimation"]["clean_features"] = json.loads(config["model_estimation"]["clean_features"])
        else:
            settings["model_estimation"]["clean_features"] = None
        if config["model_estimation"]["mkt_cap_smooth_window"].lower() != "null":
            settings["model_estimation"]["mkt_cap_smooth_window"] = int(
                config["model_estimation"]["mkt_cap_smooth_window"]
            )
        else:
            settings["model_estimation"]["mkt_cap_smooth_window"] = None
    except (KeyError, ValueError) as e:
        raise ValueError(
            "Failed to set model estimation settings; your config gile must have a section "
            "'model_estimation' with values for 'winsor_factor' (float), 'residualize_styles' (string bool) "
            "'mkt_factor_col' (string), 'res_ret_col', (string), 'top_n_by_mkt_cap' (int), 'mkt_cap_smooth_window' "
            "(int), 'make_sector_dummies' (bool) 'clean_features' (list[str])"
            "See the sample config.ini for full documentation of the types and uses for each setting. "
            "If you don't have a config, try toraniko-init at the command line."
        ) from e
    settings["style_factors"] = {}
    try:
        if config["style_factors.mom"]["enabled"].lower() == "false":
            settings["style_factors"]["mom"] = {
                "enabled": False,
            }
        else:
            settings["style_factors"]["mom"] = {
                "enabled": True,
                "trailing_days": int(config["style_factors.mom"]["trailing_days"]),
                "half_life": int(config["style_factors.mom"]["half_life"]),
                "lag": int(config["style_factors.mom"]["lag"]),
                "center": config["style_factors.mom"]["center"].lower() == "true",
                "standardize": config["style_factors.mom"]["standardize"].lower() == "true",
                "score_col": config["style_factors.mom"]["score_col"],
            }
            if config["style_factors.mom"]["winsor_factor"].lower() != "null":
                settings["style_factors"]["mom"]["winsor_factor"] = float(config["style_factors.mom"]["winsor_factor"])
            else:
                settings["style_factors"]["mom"]["winsor_factor"] = None
    except KeyError as e:
        raise ValueError(
            "Failed to read `style_factors.mom` section in config; please see "
            "documentation in the sample config.ini file"
        ) from e
    except ValueError as e:
        raise TypeError(
            "'trailing_days', 'half_life' and 'lag' must all have integer "
            "values in config `style_factors.mom`. `winsor_factor` must be null or float. "
            "center and standardize must be boolean strings"
        ) from e
    try:
        if config["style_factors.sze"]["enabled"].lower() == "false":
            settings["style_factors"]["sze"] = {
                "enabled": False,
            }
        else:
            settings["style_factors"]["sze"] = {
                "enabled": True,
                "center": config["style_factors.sze"]["center"].lower() == "true",
                "standardize": config["style_factors.sze"]["standardize"].lower() == "true",
                "score_col": config["style_factors.sze"]["score_col"],
            }
        if config["style_factors.sze"]["lower_decile"].lower() == "null":
            settings["style_factors"]["sze"]["lower_decile"] = None
        else:
            settings["style_factors"]["sze"]["lower_decile"] = float(config["style_factors.sze"]["lower_decile"])
        if config["style_factors.sze"]["upper_decile"].lower() == "null":
            settings["style_factors"]["sze"]["upper_decile"] = None
        else:
            settings["style_factors"]["sze"]["upper_decile"] = float(config["style_factors.sze"]["upper_decile"])
    except KeyError as e:
        raise ValueError(
            "Failed to read `style_factors.sze` section in config; please see "
            "documentation in the sample config.ini file"
        ) from e
    except ValueError as e:
        raise TypeError("'lower_decile' and 'upper_decile' in `style_factors.sze` must be float or null") from e
    try:
        if config["style_factors.val"]["enabled"].lower() == "false":
            settings["style_factors"]["val"] = {"enabled": False}
        else:
            settings["style_factors"]["val"] = {
                "enabled": True,
                "center": config["style_factors.val"]["center"].lower() == "true",
                "standardize": config["style_factors.val"]["standardize"].lower() == "true",
                "score_col": config["style_factors.val"]["score_col"],
                "bp_col": config["style_factors.val"]["bp_col"],
                "sp_col": config["style_factors.val"]["sp_col"],
                "cf_col": config["style_factors.val"]["cf_col"],
            }
            if config["style_factors.val"]["winsor_factor"].lower() != "null":
                settings["style_factors"]["val"]["winsor_factor"] = float(config["style_factors.val"]["winsor_factor"])
            else:
                settings["style_factors"]["val"]["winsor_factor"] = None
    except KeyError as e:
        raise ValueError(
            "Failed to read `style_factors.val` section in config; please see "
            "documentation in the sample config.ini file"
        ) from e
    except ValueError as e:
        raise TypeError("'winsor_factor' in `style_factors.val` section must be a float or null type") from e
    return settings
