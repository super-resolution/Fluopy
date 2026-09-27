"""
.. currentmodule:: fluopy

Welcome to the base module of fluopy!
"""

import logging

logger = logging.getLogger(__name__)


try:
    from fluopy._version import version as __version__
except ImportError:
    __version__ = "not-installed"

from . import (
    analysis,
    blinking,
    emissions,
    fcs,
    fluo_data,
    fluorophores,
    kappa_squared,
    photophysics,
    plotting,
    prediction,
    simulation,
    tcspc,
    transitions,
)

__all__: list[str] = [
    "analysis",
    "blinking",
    "emissions",
    "fcs",
    "fluo_data",
    "fluorophores",
    "kappa_squared",
    "photophysics",
    "plotting",
    "prediction",
    "simulation",
    "tcspc",
    "transitions",
]
