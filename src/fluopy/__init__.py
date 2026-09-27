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
from .emissions import *
from .fcs import *
from .plotting import *
from .fluo_data import *
from .fluorophores import *
from .photophysics import *
from .kappa_squared import *
from .prediction import *
from .simulation import *
from .tcspc import *
from .transitions import *

submodules: list[str] = [
    "analysis",
    "blinking",
    "emissions",
    "fcs",
    "plotting",
    "fluo_data",
    "fluorophores",
    "photophysics",
    "kappa_squared",
    "prediction",
    "simulation",
    "tcspc",
    "transitions",
]

for submodule in submodules:
    module_ = import_module(name=f".{submodule}", package="fluopy")
    if hasattr(module_, "__all__"):
        __all__.extend(module_.__all__)
