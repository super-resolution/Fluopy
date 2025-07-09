"""
Type definitions
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

# type definition for random number generator seed
RandomGeneratorSeed = (
    None
    | int
    | Sequence[int]
    | np.random.SeedSequence
    | np.random.BitGenerator
    | np.random.Generator
)
