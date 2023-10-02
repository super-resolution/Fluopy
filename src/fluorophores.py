"""
Module fluorophores
"""
from dataclasses import dataclass, field
from collections.abc import Collection
import numpy as np
from typing import Optional
import src.figure as fi
from abc import ABC


@dataclass
class FluorophoreData(ABC):
    """
    Abstract class, intended to be subclassed by fluorophore dataclasses that represent real fluorophores.
    """
    pass


@dataclass
class Cy5(FluorophoreData):
    """
    Contains constant attributes of the fluorophore Cy5.
    """
    maximum_extinction_coefficient: float = 2.5e5
    quantum_yield: float = 0.27
    fluorescence_lifetime: float = 1e-9

    # intersystem crossing
    isc_st_rate: float = 8.3e5
    isc_ts_rate: float = 5e3

    # photobleaching
    photobleach_t1_rate: float = 1
    photobleach_t2_rate: float = 0

    # cis/trans isomerization
    iso_rate: float = 2e7
    biso_cross_section: float = 1.7e-17

    # dipole orientation factor
    fret_kappa_sq: float = 2/3
    # spectral overlap integral
    j_homo_fret: float = 1.55e16
    j_cis_fret: float = 3e16
    j_triplet_fret: float = 9e15
    j_off_fret: float = 1e15

    # dstorm-specific attributes
    dstorm_pet_t_rate_mol: float = 1e8
    dstorm_pet_s_rate_mol: float = 1e9
    dstorm_th_el_rate: float = 2e-2


@dataclass
class Fluorophore:
    """
    Contains attributes of a fluorophore.

    Attributes
    ----------
    id : int
        The id of the fluorophore. Not None if fluorophore is part of a FluorophoreSystem.
    name : str
        Name of the fluorophore.
    position : Collection
        The position of the fluorophore in 2D space.
    attribute_container : None, FluorophoreData
        Not None if the fluorophore has a defined FluorophoreData dataclass.
    """
    id: int = field(init=False)
    name: str = field()
    position: Collection[float, float] = field()
    attribute_container: Optional[FluorophoreData] = None

    def __post_init__(self):
        object.__setattr__(self, 'id', None)
        object.__setattr__(self, 'position', np.asarray(self.position))
        if self.name.lower() == 'cy5':
            object.__setattr__(self, 'attribute_container', Cy5)


@dataclass
class FluorophoreSystem:
    """
    Container for attributes of multiple, interrelated fluorophores.

    Attributes
    ----------
    fluorophores : Collection
        Contains all given fluorophores of type Fluorophore.
    distances : dict
        Contains tuples of 2 fluorophore ids as keys and their distance as values.
    count : int
        The total number of fluorophores given.
    """
    fluorophores: Collection[Fluorophore] = field()
    distances: dict = field(init=False)
    count: int = field(init=False)

    def __post_init__(self):
        for i, fluorophore in enumerate(self.fluorophores):
            fluorophore.id = i
        object.__setattr__(self, 'distances', get_distances([fluo.position for fluo in self.fluorophores]))
        object.__setattr__(self, 'count', len(self.fluorophores))

    def plot(self, quadratic=True, **kwargs):
        """
        Plot the positions of fluorophores.

        Parameters
        ----------
        quadratic : bool
            Whether to display the plot with same x and y axis scaling.
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        positions = np.empty(shape=(2, self.count))
        labels = []
        for i, fluorophore in enumerate(self.fluorophores):
            positions[:, i] = fluorophore.position
            labels.append(fluorophore.name)
        kwargs.setdefault('type_', 'scatter')
        kwargs.setdefault('xlabel', 'x [nm]')
        kwargs.setdefault('ylabel', 'y [nm]')
        axes = fi.universal_figure(data=positions, **kwargs)
        for i, label in enumerate(labels):
            axes[0, 0].annotate(label, (positions[0, i], positions[1, i]))
        axes[0, 0].margins(0.2, 0.2)
        if quadratic:
            axes[0, 0].set_aspect('equal', adjustable='box')

        return axes


def get_distances(positions):
    """
    Gets distances between positions.

    Parameters
    ----------
    positions : Collection
        Contains coordinate pairs (2D).

    Returns
    -------
    distances : dict
        Contains tuples of ids (order as positions) as keys and their distance as values.
    """
    distances = dict()
    positions = np.asarray(positions)
    for i, position_1 in enumerate(positions):
        for j, position_2 in enumerate(positions):
            if i != j and (i, j) not in distances:
                distances[(i, j)] = np.round(np.linalg.norm(position_1 - position_2), 3)

    return distances


def triangle_third_position(position_1=None, position_2=None):
    """
    Get the third position of an equilateral triangle based on positions of two vertices. There are two solutions to
    such a position but only one is considered here.

    Parameters
    ----------
    position_1 : 1-D array_like
        The position of the first vertex.
    position_2 : 1-D array_like
        The position of the second vertex.

    Returns
    -------
    position_3 : 1-D array_like
        The position of the third vertex.
    """
    if position_1 is None:
        position_1 = np.array([0, 0])
    if position_2 is None:
        position_2 = np.array([0, 10])
    x1, y1 = position_1
    x2, y2 = position_2
    x3 = (x1 + x2 + np.sqrt(3) * (y1 - y2)) / 2
    y3 = (y1 + y2 + np.sqrt(3) * (x1 - x2)) / -2
    position_3 = np.array([x3, y3])

    return position_3


def get_positions_from_distance(distance, count):
    """
    Gets positions of up to 4 fluorophores based on a single distance. If it is 3 fluorophores, they are positioned in
    an equilateral triangle. If it is 4 fluorophores, they are positioned in a square.

    Parameters
    ----------
    distance : float
        Minimum distance between fluorophores.
    count : int
        Number of fluorophores.

    Returns
    -------
    positions : np.ndarray
        Contains np.ndarrays of x and y for each fluorophore.
    """
    position_1 = np.array([0, 0])
    position_2 = np.array([distance, 0])
    if count == 1:
        positions = np.array([position_1])
    elif count == 2:
        positions = np.array([position_1, position_2])
    elif count == 3:
        position_3 = triangle_third_position(position_1=position_1, position_2=position_2)
        positions = np.array([position_1, position_2, position_3])
    elif count == 4:
        position_3 = np.array([0, distance])
        position_4 = np.array([distance, distance])
        positions = np.array([position_1, position_2, position_3, position_4])
    else:
        raise AttributeError('count has to be one of 1, 2, 3 or 4.')

    return positions


def construct_fluorophores(name='cy5', distance=10, count=3):
    """
    Constructs a collection of fluorophores of up to 4 fluorophores of the same kind.

    Parameters
    ----------
    name : str
        Name of the fluorophore.
    distance : float
        Minimum distance between fluorophores.
    count : int
        Number of fluorophores.

    Returns
    -------
    fluorophores : Collection
        Contains fluorophores of type Fluorophore.
    """
    positions = get_positions_from_distance(distance=distance, count=count)
    fluorophores = [Fluorophore(name=name, position=position) for position in positions]

    return fluorophores
