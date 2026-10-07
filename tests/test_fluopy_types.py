from collections.abc import Sequence
from types import NoneType
from typing import get_args

import numpy as np

from fluopy.fluopy_types import RandomGeneratorSeed


def test_random_generator_seed_types():
    assert get_args(RandomGeneratorSeed) == (
        NoneType,
        int,
        Sequence[int],
        np.random.SeedSequence,
        np.random.BitGenerator,
        np.random.Generator,
    )
