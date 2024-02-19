"""
Module fluorophores
"""
from dataclasses import dataclass, field
from collections.abc import Collection
import numpy as np
import warnings
from pathlib import Path
import os
import pandas as pd
from typing import Optional
import src.figure as fi
import src.fluo_data as fd


@dataclass
class Fluorophore:
    """
    Contains attributes of a fluorophore.

    Attributes
    ----------
    identity : int
        The id of the fluorophore. Not None if fluorophore is part of a FluorophoreSystem.
    name : str
        Name of the fluorophore.
    position : Collection
        The position of the fluorophore in 2D space.
    constants : None, FluorophoreData
        Not None if the fluorophore has a defined FluorophoreData dataclass.
    """
    identity: int = field(init=False)
    name: str = field()
    position: Collection[float, float] = field()
    constants: Optional[fd.FluorophoreData] = None

    def __post_init__(self):
        object.__setattr__(self, 'identity', None)
        object.__setattr__(self, 'position', np.asarray(self.position))
        fluorophore_dataclasses = [name for name in dir(fd) if isinstance(getattr(fd, name), type)]
        class_name = [name for name in fluorophore_dataclasses if name.lower() == self.name.lower()]
        if len(class_name) == 1:
            object.__setattr__(self, 'constants', getattr(fd, class_name[0])())        
        elif len(class_name) == 0:
            warnings.warn(f'Fluorophore {self.name} not known. Parameters have to be defined manually.')
        else:
            raise ValueError('Multiple fluorophore dataclasses found.')


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
            fluorophore.identity = i
        object.__setattr__(self, 'distances', get_distances([fluo.position for fluo in self.fluorophores]))
        object.__setattr__(self, 'count', len(self.fluorophores))

    def load_transitions(self, irradiance=2, wavelength=600, bleaching=False, energy_transfer=True, dstorm=True,
                         **dstorm_parameters):
        """
        Derives transitions based on fluorophore and the experimental conditions to be mimicked.

        Parameters
        ----------
        irradiance : float
            Irradiance in kW/cm².
        wavelength : float
            Wavelength in nm.
        bleaching : bool
            Whether to incooperate bleaching as a possible transition.
        energy_transfer : bool
            Whether to incooperate energy transfers as possible transitions.
        dstorm : bool
            Whether to incooperate dstorm photoswitching as possible transitions.
        dstorm_parameters : dict
            May contain the following keys: reducing_agent, concentration, k_pet, ph.
            Only needed if dstorm is True.

        Returns
        -------
        transitions : Collection
            Contains transitions of type Transition.
        """
        transitions = {}
        from src.transitions import derive_transitions
        for fluorophore in self.fluorophores:
            if fluorophore.constants is None:
                warnings.warn(f'load_transitions() not available for this kind of fluorophore: {fluorophore.name}.')
            elif fluorophore.name not in transitions:
                transitions[fluorophore.name] = \
                    derive_transitions(fluorophore_data=fluorophore.constants,irradiance=irradiance, wavelength=wavelength,
                                       bleaching=bleaching, energy_transfer=energy_transfer, distances=self.distances, 
                                       dstorm=dstorm, **dstorm_parameters)
            else:
                continue
        
        return transitions

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


def get_relative_extinction(wavelength, file_directory):
    """
    Get the relative extinction from a file given the wavelength.

    Parameters
    ----------
    wavelength : float
        The wavelength in nm.
    file_directory : str
        The name of the tail directory.

    Returns
    -------
    relative_extinction : float
        The relative extinction at the given wavelength.
    """
    path_absorption = os.path.join(Path(__file__).parent, 'fluorophore_collection', file_directory, 'rel_absorption.csv')
    dataframe_absorption = pd.read_csv(filepath_or_buffer=path_absorption, index_col=0)
    relative_extinction =  dataframe_absorption.loc[int(wavelength), 'relative extinction']
    
    return relative_extinction


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


def get_positions_from_distance(distance, count, shape='triangle'):
    """
    Gets positions of up to 4 fluorophores based on a single distance. If it is 3 fluorophores, they are positioned
    either in an equilateral triangle or in a square with a missing vertex. If it is 4 fluorophores, they are
    positioned in a square.

    Parameters
    ----------
    distance : float
        Minimum distance between fluorophores.
    count : int
        Number of fluorophores.
    shape : str
        One of 'triangle', 'square'. Only needed if count is 3.

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
        if shape == 'triangle':
            position_3 = triangle_third_position(position_1=position_1, position_2=position_2)
        elif shape == 'square':
            position_3 = np.array([0, distance])
        else:
            raise ValueError(f"shape {shape} not known. Can either be 'triangle' or 'square'.")
        positions = np.array([position_1, position_2, position_3])
    elif count == 4:
        position_3 = np.array([0, distance])
        position_4 = np.array([distance, distance])
        positions = np.array([position_1, position_2, position_3, position_4])
    else:
        raise AttributeError('count has to be one of 1, 2, 3 or 4.')

    return positions


def construct_fluorophores(name='cy5', distance=10, count=3, shape='triangle'):
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
    shape : str
        One of 'triangle', 'square'. Only needed if count is 3.

    Returns
    -------
    fluorophores : Collection
        Contains fluorophores of type Fluorophore.
    """
    positions = get_positions_from_distance(distance=distance, count=count, shape=shape)
    fluorophores = [Fluorophore(name=name, position=position) for position in positions]

    return fluorophores
