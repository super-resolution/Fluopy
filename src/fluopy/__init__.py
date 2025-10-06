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
from .figure import *
from .fitting import *
from .fluo_data import *
from .fluorophores import *
from .formulas import *
from .kappa_squared import *
from .miscellaneous import *
from .network import *
from .prediction import *
from .routines import *
from .simulation import *
from .simulation_tcspc import *
from .transitions import *


submodules: list[str] = [
    "analysis",
    "blinking",
    "distributions",
    "emissions",
    "fcs",
    "figure",
    "fitting",
    "fluo_data",
    "fluorophores",
    "formulas",
    "kappa_squared",
    "miscellaneous",
    "network",
    "prediction",
    "routines",
    "simulation",
    "simulation_tcspc",
    "transitions",
]

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="fluopy")
    if hasattr(module_, "__all__"):
        __all__.extend(module_.__all__)
