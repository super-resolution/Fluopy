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

from . import analysis
from . import blinking
from . import distributions
from . import emissions
from . import fcs
from . import figure
from . import fitting
from . import fluorophores
from . import formulas
from . import miscellaneous
from . import prediction
from . import routines
from . import simulation
from . import simulation_tcspc
from . import transitions
