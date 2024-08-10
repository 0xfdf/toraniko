from __future__ import annotations

import sys

__version__ = "1.1.2"


# Credit to Ritchie Vink and Polars: this implementation comes almost directly from
# https://github.com/pola-rs/polars/blob/main/py-polars/polars/meta/versions.py
def show_versions() -> None:
    """
    Print out the version of Toraniko and its optional dependencies.

    Examples
    --------
    >>> toraniko.show_versions()
    --------Version info---------
    Toraniko:   1.1.2
    Platform:   macOS-14.5-arm64-arm-64bit
    Python:     3.11.9 (main, Apr 19 2024, 11:43:47) [Clang 14.0.6 ]

    ----Optional dependencies----
    numpy:      1.26.4
    polars:     1.0.0
    """  # noqa: W505
    # Note: we import 'platform' here (rather than at the top of the
    # module) as a micro-optimization for polars' initial import
    import platform

    deps = _get_dependency_info()
    core_properties = ("Toraniko", "Index type", "Platform", "Python")
    keylen = max(len(x) for x in [*core_properties, *deps.keys()]) + 1

    print("--------Version info---------")
    print(f"{'Toraniko:':{keylen}s} {__version__}")
    print(f"{'Platform:':{keylen}s} {platform.platform()}")
    print(f"{'Python:':{keylen}s} {sys.version}")

    print("\n----Optional dependencies----")
    for name, v in deps.items():
        print(f"{name:{keylen}s} {v}")


def _get_dependency_info() -> dict[str, str]:
    # see the list of dependencies in pyproject.toml
    opt_deps = ["numpy", "polars"]
    return {f"{name}:": _get_dependency_version(name) for name in opt_deps}


def _get_dependency_version(dep_name: str) -> str:
    # note: we import 'importlib' here as a significant optimisation for initial import
    import importlib
    import importlib.metadata

    try:
        module = importlib.import_module(dep_name)
    except ImportError:
        return "<not installed>"

    if hasattr(module, "__version__"):
        module_version = module.__version__
    else:
        module_version = importlib.metadata.version(dep_name)  # pragma: no cover

    return module_version
