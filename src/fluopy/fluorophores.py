"""
Module fluorophores
"""

import warnings
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from . import figure as fi
from . import fluo_data as fd


@dataclass
class Fluorophore:
    """
    Contains attributes of a fluorophore.

    Attributes
    ----------
    identity : int
        The id of the fluorophore. Not None if fluorophore is part of a
        FluorophoreSystem.
    name : str
        Name of the fluorophore.
    position : Collection
        The position of the fluorophore in 2D space.
    constants : None, FluorophoreData
        Not None if the fluorophore has a defined FluorophoreData
        dataclass.
    """

    identity: int = field(init=False)
    name: str = field()
    position: Collection[float, float] = field()
    constants: Optional[fd.FluorophoreData] = None

    def __post_init__(self):
        object.__setattr__(self, "identity", None)
        object.__setattr__(self, "position", np.asarray(self.position))
        fluorophore_dataclasses = [
            name for name in dir(fd) if isinstance(getattr(fd, name), type)
        ]
        class_name = [
            name
            for name in fluorophore_dataclasses
            if name.lower() == self.name.lower()
        ]
        if len(class_name) == 1:
            object.__setattr__(self, "constants", getattr(fd, class_name[0])())
        elif len(class_name) == 0:
            warnings.warn(
                f"Fluorophore {self.name} not known. Parameters have to be defined "
                "manually."
            )
        else:
            raise ValueError("Multiple fluorophore dataclasses found.")


@dataclass
class FluorophoreSystem:
    """
    Container for attributes of multiple, interrelated fluorophores.

    Attributes
    ----------
    fluorophores : Collection
        Contains all given fluorophores of type Fluorophore.
    multi_type : bool
        Whether there are multiple types of fluorophores in the system.
    distances : dict
        Contains tuples of 2 fluorophore ids as keys and their distance as values.
    count : int
        The total number of fluorophores given.
    """

    fluorophores: Collection[Fluorophore] = field()
    multi_type: bool = field(init=False)
    distances: dict = field(init=False)
    count: int = field(init=False)

    def __post_init__(self):
        for i, fluorophore in enumerate(self.fluorophores):
            fluorophore.identity = i
        if all(
            fluorophore.name == self.fluorophores[0].name
            for fluorophore in self.fluorophores
        ):
            object.__setattr__(self, "multi_type", False)
        else:
            object.__setattr__(self, "multi_type", True)
        object.__setattr__(
            self,
            "distances",
            get_distances([fluo.position for fluo in self.fluorophores]),
        )
        if 0 in self.distances.values():
            raise ValueError(
                "at least two fluorophores share the same position. Also "
                "check for duplicates."
            )
        object.__setattr__(self, "count", len(self.fluorophores))

    def load_transitions(
        self,
        summarize=False,
        irradiance=2,
        wavelength=600,
        bleaching=False,
        energy_transfer=True,
        dstorm=True,
        energy_transfer_parameters=None,
        dstorm_parameters=None,
    ):
        """
        Derives transitions based on fluorophore and the experimental conditions to be
        mimicked.

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
        energy_transfer_parameters : dict, optional
            May contain the following keys: dipole_orientation_factor, refractive_index,
            overwrite, exclude, include.
            Only used if energy_transfer is True.
            - overwrite : dict
                Contains the type of acceptor state as key and a list with a factor for
                the rate as well as an efficiency (of not recylcing acceptor state) as
                value.
            - exclude : list
                Contains the type of acceptor state (lowercase) to be excluded.
            - include : dict
                Contains the type of acceptor state as key and a list of tuples as
                values. The tuples contain the transition type and an efficiency. If the
                summed efficiencies is e.g., 0.5, all other energy transfers affecting
                the acceptor state are multiplied by 1-0.5.
        dstorm_parameters : dict, optional
            May contain the following keys: reducing_agent, concentration, k_pet, ph.
            Only used if dstorm is True.

        Returns
        -------
        transitions : dict
            Contains lists of transitions of type Transition as values and fluorophores
            or fluorophore-combinations as keys.
        """
        transitions = {}
        from .transitions import (
            derive_energy_transfer_transitions,
            derive_transitions,
        )

        skip_warnings = []
        et_pairs = {}
        for (donor, acceptor), distance in self.distances.items():
            if (
                f"{self.fluorophores[donor].name}, "
                f"{self.fluorophores[acceptor].name}, "
                f"{distance}" not in et_pairs
            ):
                et_pairs[
                    f"{self.fluorophores[donor].name}, "
                    f"{self.fluorophores[acceptor].name}, "
                    f"{distance}"
                ] = [(donor, acceptor)]
            else:
                et_pairs[
                    f"{self.fluorophores[donor].name}, "
                    f"{self.fluorophores[acceptor].name}, "
                    f"{distance}"
                ] += [(donor, acceptor)]

        if energy_transfer_parameters is None:
            energy_transfer_parameters = {}
        else:
            if any(
                key in energy_transfer_parameters
                for key in ["overwrite", "exclude", "include"]
            ):
                if self.multi_type:
                    warnings.warn(
                        "'overwrite', 'exclude' or 'include' in "
                        "energy_transfer_parameters will effect all types of "
                        "fluorophores."
                    )
        energy_transfer_parameters.setdefault("dipole_orientation_factor", 2 / 3)
        energy_transfer_parameters.setdefault("refractive_index", 1.33)

        if dstorm_parameters is None:
            dstorm_parameters = {}
        dstorm_parameters.setdefault("reducing_agent", "mea")
        dstorm_parameters.setdefault("concentration", 143)
        dstorm_parameters.setdefault("ph", 7.5)

        for fluorophore in self.fluorophores:
            fluorophore_ids = [
                f.identity for f in self.fluorophores if f.name == fluorophore.name
            ]
            if fluorophore.constants is None:
                if fluorophore not in skip_warnings:
                    skip_warnings.append(fluorophore)
                    warnings.warn(
                        "load_transitions() not available for this kind of "
                        f"fluorophore: {fluorophore.name}."
                    )
                else:
                    continue
            else:
                if fluorophore.name not in transitions:
                    transitions[fluorophore.name] = derive_transitions(
                        summarize=summarize,
                        fluorophore_data=fluorophore.constants,
                        fluorophore_ids=fluorophore_ids,
                        irradiance=irradiance,
                        wavelength=wavelength,
                        bleaching=bleaching,
                        dstorm=dstorm,
                        **dstorm_parameters,
                    )
                if energy_transfer:
                    donor = fluorophore
                    for acceptor in self.fluorophores:
                        if acceptor.constants is None:
                            if acceptor not in skip_warnings:
                                skip_warnings.append(acceptor)
                                warnings.warn(
                                    "load_transitions() not available for this kind of "
                                    f"fluorophore: {acceptor.name}."
                                )
                            else:
                                continue
                        elif (donor.identity, acceptor.identity) in self.distances:
                            distance = self.distances[donor.identity, acceptor.identity]
                            fluorophore_ids = et_pairs[
                                f"{donor.name}, {acceptor.name}, {distance}"
                            ]

                            transitions[
                                f"D: {donor.name}, A: {acceptor.name}, dist: {distance}"
                            ] = derive_energy_transfer_transitions(
                                donor_data=donor.constants,
                                acceptor_data=acceptor.constants,
                                fluorophore_ids=fluorophore_ids,
                                distance=distance,
                                **energy_transfer_parameters,
                            )

        return transitions

    def plot(self, quadratic=True, **kwargs):
        """
        Plot the positions of fluorophores.

        Parameters
        ----------
        quadratic : bool
            Whether to display the plot with same x and y axis scaling.
        kwargs : fluopy.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        positions = np.empty(shape=(2, self.count))
        labels = []
        for i, fluorophore in enumerate(self.fluorophores):
            positions[:, i] = fluorophore.position
            labels.append(fluorophore.name + f" ({fluorophore.identity})")
        kwargs.setdefault("type_", "scatter")
        kwargs.setdefault("xlabel", "x [nm]")
        kwargs.setdefault("ylabel", "y [nm]")
        axes = fi.universal_figure(data=positions, **kwargs)
        for i, label in enumerate(labels):
            axes[0, 0].annotate(label, (positions[0, i], positions[1, i]))
        axes[0, 0].margins(0.2, 0.2)
        if quadratic:
            axes[0, 0].set_aspect("equal", adjustable="box")

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
        Contains tuples of ids (order as positions) as keys and their distance as
        values.
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
    Get the third position of an equilateral triangle based on positions of two
    vertices. There are two solutions to such a position but only one is considered
    here.

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


def get_positions_from_distance(distance=1, count=1, shape="triangle"):
    """
    Gets positions of up to 4 fluorophores based on a single distance. If it is 3
    fluorophores, they are positioned either in an equilateral triangle or in a square
    with a missing vertex. If it is 4 fluorophores, they are positioned in a square.

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
    if count == 1:
        positions = np.array([position_1])
    elif count in [2, 3, 4]:
        position_2 = np.array([distance, 0])
        if count == 2:
            positions = np.array([position_1, position_2])
        elif count == 3:
            if shape == "triangle":
                position_3 = triangle_third_position(
                    position_1=position_1, position_2=position_2
                )
            elif shape == "square":
                position_3 = np.array([0, distance])
            else:
                raise ValueError(
                    f"shape {shape} not known. Can either be 'triangle' or 'square'."
                )
            positions = np.array([position_1, position_2, position_3])
        elif count == 4:
            position_3 = np.array([0, distance])
            position_4 = np.array([distance, distance])
            positions = np.array([position_1, position_2, position_3, position_4])
    else:
        warnings.warn(
            "If count is above 4, all fluorophores are positioned at the same"
            " location. This indicates no support for energy transfers."
        )
        positions = [[position_1[0], position_1[1] + i] for i in range(count)]

    return positions


def construct_fluorophores(name="cy5", distance=10, count=3, shape="triangle"):
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
