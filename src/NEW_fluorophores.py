from dataclasses import dataclass, field
from collections.abc import Collection
from typing import ClassVar
import numpy as np


@dataclass
class Fluorophore:
    count: ClassVar[int] = 0

    id: int = field(init=False)
    name: str = field()
    position: Collection[float, float] = field()

    def __post_init__(self):
        Fluorophore.count += 1
        object.__setattr__(self, 'id', Fluorophore.count)

# do distances as optional setattr if positions are given, if not it has to be provided
# fluorophoresystem HAS transitionset??? Composition
@dataclass
class FluorophoreSystem:
    fluorophores: Collection[Fluorophore] = field()
    distances: dict = field(init=False)
    count: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'distances', get_distances(self.fluorophores))
        object.__setattr__(self, 'count', len(self.fluorophores))


def get_distances(fluorophores):
    distances = dict()
    for i, fluorophore in enumerate(fluorophores):
        position_1 = fluorophore.position
        for j, fluorophore in enumerate(fluorophores):
            if i != j and (i, j) not in distances:
                position_2 = fluorophore.position
                distances[(i, j)] = np.linalg.norm(position_1 - position_2)

    return distances
