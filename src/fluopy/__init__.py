"""
.. currentmodule:: fluopy

Welcome to the base module of fluopy!
"""

import logging
from importlib import import_module

__all__: list[str] = []

logger = logging.getLogger(__name__)


try:
    from fluopy._version import version as __version__
except ImportError:
    __version__ = "not-installed"

from .analysis import *
from .blinking import *
from .distributions import *
from .emissions import *
from .fcs import *
from .plotting import *
from .fitting import *
from .fluo_data import *
from .fluorophores import *
from .photophysics import *
from .kappa_squared import *
from .miscellaneous import *
from .prediction import *
from .routines import *
from .simulation import *
from .tcspc import *
from .transitions import *

submodules: list[str] = [
    "analysis",
    "blinking",
    "distributions",
    "emissions",
    "fcs",
    "plotting",
    "fitting",
    "fluo_data",
    "fluorophores",
    "photophysics",
    "kappa_squared",
    "miscellaneous",
    "prediction",
    "routines",
    "simulation",
    "tcspc",
    "transitions",
]

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="fluopy")
    if hasattr(module_, "__all__"):
        __all__.extend(module_.__all__)
