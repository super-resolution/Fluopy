from dataclasses import dataclass, field
from collections.abc import Collection
import numpy as np
from typing import Optional
import src.custom_plot as cp


@dataclass
class Cy5:
    maximum_extinction_coefficient: float = 2.5e5
    quantum_yield: float = 0.27
    fluorescence_lifetime: float = 1e-9
    isc_st_rate: float = 8.3e5
    isc_ts_rate: float = 5e5
    photobleach_s1_rate: float = 0
    photobleach_t1_rate: float = 0
    photobleach_t2_rate: float = 0
    iso_rate: float = 2e7
    biso_cross_section: float = 1.7e-17

    fret_kappa_sq: float = 2/3

    j_homo_fret: float = 1.55e16
    j_cis_fret: float = 3e16
    j_triplet_fret: float = 9e15
    j_off_fret: float = 1e15

    dstorm_red_rate_mol: float = 9.6e7
    dstorm_oxi_rate: float = 2e-1


@dataclass
class Fluorophore:
    id: int = field(init=False)
    name: str = field()
    position: Collection[float, float] = field()
    attributes: Optional[Cy5] = field(init=False)

    def __post_init__(self):
        if self.name == 'cy5':
            object.__setattr__(self, 'attributes', Cy5)


@dataclass
class FluorophoreSystem:
    fluorophores: Collection[Fluorophore] = field()
    distances: dict = field(init=False)
    count: int = field(init=False)

    def __post_init__(self):
        for i, fluorophore in enumerate(self.fluorophores):
            fluorophore.id = i
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


def get_distances(fluorophores):
    distances = dict()
    for fluorophore_1 in fluorophores:
        for fluorophore_2 in fluorophores:
            if fluorophore_1.id != fluorophore_2.id and (fluorophore_1.id, fluorophore_2.id) not in distances:
                distances[(fluorophore_1.id, fluorophore_2.id)] = \
                    np.linalg.norm(fluorophore_1.position - fluorophore_2.position)

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


def get_positions_from_distance(distance, count):
    position_1 = np.array([0, 0])
    position_2 = np.array([distance, 0])
    if count == 2:
        positions = position_1, position_2
    elif count == 3:
        position_3 = triangle_third_position(position_1, position_2)
        positions = position_1, position_2, position_3
    elif count == 4:
        position_3 = np.array([0, distance])
        position_4 = np.array([distance, distance])
        positions = position_1, position_2, position_3, position_4
    else:
        raise AttributeError('count has to be one of 2, 3 or 4.')

    return positions


def triangle_third_position(position_1, position_2):
    distance = np.linalg.norm(position_1 - position_2)
    x1, y1 = position_1
    x2, y2 = position_2
    x3 = (x1 + x2 + np.sqrt(3) * (y1 - y2)) / 2
    y3 = (y1 + y2 + np.sqrt(3) * (x1 - x2)) / -2
    return np.array([x3, y3])


def construct_fluorophores(name, distance, count):
    positions = get_positions_from_distance(distance, count)
    fluorophores = [Fluorophore(name=name, position=position) for position in positions]
    return fluorophores
