from dataclasses import dataclass, field
from collections.abc import Collection
from typing import ClassVar
import numpy as np
import src.custom_plot as cp


@dataclass
class Fluorophore:
    count: ClassVar[int] = 0

    id: int = field(init=False)
    name: str = field()
    position: Collection[float, float] = field()

    def __post_init__(self):
        Fluorophore.count += 1
        object.__setattr__(self, 'id', Fluorophore.count)


@dataclass
class FluorophoreSystem:
    fluorophores: Collection[Fluorophore] = field()
    distances: dict = field(init=False)
    count: int = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'distances', get_distances(self.fluorophores))
        object.__setattr__(self, 'count', len(self.fluorophores))

    def plot(self, **kwargs):
        positions = np.empty(shape=(2, self.count))
        labels = []
        for i, fluorophore in enumerate(self.fluorophores):
            positions[:, i] = fluorophore.position
            labels.append(fluorophore.name)
        fig, ax = plot_scatter(data=positions, labels=labels, **kwargs)
        return fig, ax

    def load_transition_set(self, irradiance, wavelength, **dstorm_parameters):
        if self.fluorophores[0].name.lower() == 'cy5':
            import src.NEW_cy5 as cy5
            transition_set = cy5.construct_transition_set(irradiance, wavelength, self.distances, **dstorm_parameters)
            return transition_set


def get_distances(fluorophores):
    distances = dict()
    for i, fluorophore in enumerate(fluorophores):
        position_1 = fluorophore.position
        for j, fluorophore in enumerate(fluorophores):
            if i != j and (i, j) not in distances:
                position_2 = fluorophore.position
                distances[(i, j)] = np.linalg.norm(position_1 - position_2)

    return distances


def plot_scatter(data, labels, **kwargs):
    kwargs.setdefault('type_', 'scatter')
    kwargs.setdefault('xlabel', 'x [nm]')
    kwargs.setdefault('ylabel', 'y [nm]')
    fig, ax = cp.universal_figure(data=data, **kwargs)
    for i, label in enumerate(labels):
        ax[0, 0].annotate(label, (data[0, i], data[1, i]))
    ax[0, 0].margins(0.2, 0.2)
    ax[0, 0].set_aspect('equal', adjustable='box')

    return fig, ax

