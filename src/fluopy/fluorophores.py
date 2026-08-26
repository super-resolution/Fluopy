"""
Define and work with fluorophores.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import numpy.typing as npt

from . import figure as fi
from . import fluo_data as fd
from .transitions import (
    derive_energy_transfer_transitions,
    derive_transitions,
)

if TYPE_CHECKING:
    from fluopy.transitions import Transition

__all__: list[str] = ["Fluorophore", "FluorophoreSystem"]

logger = logging.getLogger(__name__)


@dataclass(eq=False)
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
    position : Collection[float]
        The position of the fluorophore in space in nm.
    constants : FluorophoreData | None
        If None an instance of FluorophoreData with the same name is inserted
        if available in fluopy.fluo_data.
    """

    identity: int | None = field(init=False, default=None)
    name: str = field()
    position: npt.NDArray[np.float64] = field()
    constants: fd.FluorophoreData | None = None

    def __post_init__(self) -> None:
        position = np.asarray(self.position, dtype=float).copy()

        if position.ndim != 1:
            raise ValueError("fluorophore position must be one-dimensional.")
        if position.size == 0:
            raise ValueError("fluorophore position must not be empty.")
        if not np.all(np.isfinite(position)):
            raise ValueError("fluorophore position must be finite.")

        position.setflags(write=False)
        self.position = position

        if self.constants is None:
            if self.name in dir(fd) and isinstance(
                getattr(fd, self.name), fd.FluorophoreData
            ):
                self.constants = getattr(fd, self.name)
            else:
                logger.warning(
                    f"There is no FluorophoreData for Fluorophore {self.name} in fluopy.fluo_data. "
                    f"Parameters have to be defined manually.",
                    stacklevel=2,
                )

    def get_identity(self) -> int:
        if self.identity is None:
            raise RuntimeError(
                "fluorophore identity is only available after adding it "
                "to a FluorophoreSystem."
            )
        return self.identity


@dataclass
class FluorophoreSystem:
    """
    Container for attributes of multiple, interrelated fluorophores.

    Attributes
    ----------
    fluorophores : Sequence[Fluorophore]
        Contains all given fluorophores of type Fluorophore.
    multi_type : bool
        Whether there are multiple types of fluorophores in the system.
    distances :  dict[tuple[int, int], np.float64]
        Contains tuples of 2 fluorophore ids as keys and their distance as values.
        The distances are given in nm and are rounded to 3 decimals.
    count : int
        The total number of fluorophores given.
    """

    fluorophores: Sequence[Fluorophore] = field()
    multi_type: bool = field(init=False)
    distances: dict[tuple[int, int], np.float64] = field(init=False)
    count: int = field(init=False)

    def __post_init__(self) -> None:
        self.fluorophores = tuple(self.fluorophores)
        if not self.fluorophores:
            raise ValueError(
                "a fluorophore system must contain at least one fluorophore."
            )

        if not all(
            isinstance(fluorophore, Fluorophore) for fluorophore in self.fluorophores
        ):
            raise TypeError("fluorophores must contain only Fluorophore objects.")

        dimensions = {fluorophore.position.size for fluorophore in self.fluorophores}

        if len(dimensions) != 1:
            raise ValueError("all fluorophore positions must have the same dimension.")

        constants_by_name: dict[str, fd.FluorophoreData | None] = {}
        for fluorophore in self.fluorophores:
            if fluorophore.name in constants_by_name:
                if fluorophore.constants is not constants_by_name[fluorophore.name]:
                    raise ValueError(
                        "fluorophores with the same name must share the same "
                        f"FluorophoreData object: {fluorophore.name}"
                    )
            else:
                constants_by_name[fluorophore.name] = fluorophore.constants

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
            get_distances(positions=[fluo.position for fluo in self.fluorophores]),
        )
        if 0 in self.distances.values():
            raise ValueError(
                "at least two fluorophores are indistinguishable at the 0.001 nm "
                "distance resolution. Also check for duplicates."
            )
        object.__setattr__(self, "count", len(self.fluorophores))

    def load_transitions(
        self,
        summarize: bool = False,
        irradiance: float = 2,
        wavelength: float = 600,
        bleaching: bool = False,
        energy_transfer: bool = True,
        dstorm: bool = True,
        energy_transfer_parameters: (
            dict[
                Literal[
                    "dipole_orientation_factor",
                    "refractive_index",
                    "overwrite",
                    "exclude",
                    "include",
                ],
                Any,
            ]
            | None
        ) = None,
        dstorm_parameters: dict[str, Any] | None = None,
    ) -> dict[str, list[Transition]]:
        """
        Derives transitions based on fluorophore and the experimental conditions to be
        mimicked.

        This method derives only the transition types implemented by Fluopy.
        User-defined SingleState, PairedState, and TransitionType objects are not
        discovered automatically. Transitions using them must be added manually to
        the returned dictionary before creating a TransitionSet.

        Parameters
        ----------
        summarize
            Whether to summarize some transitions into fewer.
        irradiance
            Irradiance in kW/cm².
        wavelength
            Wavelength in nm.
        bleaching
            Whether to incorporate bleaching as a possible transition.
        energy_transfer
            Whether to incorporate energy transfers as possible transitions.
        dstorm
            Whether to incorporate dstorm photoswitching as possible transitions.
        energy_transfer_parameters
            May contain the following keys: dipole_orientation_factor, refractive_index,
            overwrite, exclude, include.
            Only used if energy_transfer is True.
            - overwrite : dict
                Contains the type of acceptor state as key and a list with a factor for
                the rate as well as an efficiency (of not recycling acceptor state) as
                value.
            - exclude : list
                Contains the type of acceptor state (lowercase) to be excluded.
            - include : dict
                Contains the type of acceptor state as key and a list of tuples as
                values. The tuples contain the transition type and an efficiency. If the
                summed efficiencies is e.g., 0.5, all other energy transfers affecting
                the acceptor state are multiplied by 1-0.5.
        dstorm_parameters
            May contain the following keys: reducing_agent, concentration, k_pet, ph.
            Only used if dstorm is True.

        Notes
        -----
        General transitions are derived for excitation, fluorescence, internal
        conversion, intersystem crossing, reverse intersystem crossing, and cis-trans
        isomerization. Setting dstorm to True adds PET, adduct, uncaging, elimination,
        and radical-state transitions. Setting bleaching to True adds triplet-state
        photobleaching.

        Energy-transfer transitions are derived from acceptor absorption spectra with
        the supported state keys s0, s1, t1, cis, and off. These produce standard FRET,
        singlet-singlet annihilation, singlet-triplet annihilation, cis-state transfer,
        and off-state transfer, respectively.

        A transition can be returned with rate zero when its corresponding
        FluorophoreData value is zero or its spectra do not overlap. TransitionSet
        removes zero-rate transitions by default.

        User-defined transition types are not discovered automatically and must be
        added manually to the returned dictionary.

        See Also
        --------
        fluopy.transitions.derive_transitions
        fluopy.transitions.derive_energy_transfer_transitions

        Returns
        -------
        transitions : dict[str, list[Transition]]
            Contains lists of transitions of type Transition as values and fluorophores
            or fluorophore-combinations (D: <donor>, A: <acceptor>, dist: <distance>) as
            keys.
        """
        transitions: dict[str, list[Transition]] = {}
        warned_names: set[str] = set()
        et_pairs: dict[str, list[tuple[int, int]]] = {}

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

        energy_transfer_parameters = dict(energy_transfer_parameters or {})

        if self.multi_type and any(
            key in energy_transfer_parameters
            for key in ["overwrite", "exclude", "include"]
        ):
            logger.warning(
                "'overwrite', 'exclude' or 'include' in "
                "energy_transfer_parameters will affect all types of "
                "fluorophores.",
                stacklevel=2,
            )
        energy_transfer_parameters.setdefault("dipole_orientation_factor", 2 / 3)
        energy_transfer_parameters.setdefault("refractive_index", 1.33)

        dstorm_parameters = dict(dstorm_parameters or {})
        dstorm_parameters.setdefault("reducing_agent", "mea")
        dstorm_parameters.setdefault("concentration", 143)
        dstorm_parameters.setdefault("ph", 7.5)

        for fluorophore in self.fluorophores:
            fluorophore_ids: list[int] = [
                f.get_identity()
                for f in self.fluorophores
                if f.name == fluorophore.name
            ]
            donor_data = fluorophore.constants
            if donor_data is None:
                if fluorophore.name not in warned_names:
                    warned_names.add(fluorophore.name)
                    logger.warning(
                        "load_transitions() not available for this kind of "
                        f"fluorophore: {fluorophore.name}.",
                        stacklevel=2,
                    )
                continue
            else:
                if fluorophore.name not in transitions:
                    transitions[fluorophore.name] = derive_transitions(
                        summarize=summarize,
                        fluorophore_data=donor_data,
                        fluorophore_ids=fluorophore_ids,
                        irradiance=irradiance,
                        wavelength=wavelength,
                        bleaching=bleaching,
                        dstorm=dstorm,
                        **dstorm_parameters,
                    )
                if energy_transfer:
                    donor_fluorophore = fluorophore
                    for acceptor_fluorophore in self.fluorophores:
                        if acceptor_fluorophore.constants is None:
                            if acceptor_fluorophore.name not in warned_names:
                                warned_names.add(acceptor_fluorophore.name)
                                logger.warning(
                                    "load_transitions() not available for this kind of "
                                    f"fluorophore: {acceptor_fluorophore.name}.",
                                    stacklevel=2,
                                )
                            continue
                        pair = (
                            donor_fluorophore.get_identity(),
                            acceptor_fluorophore.get_identity(),
                        )
                        if pair in self.distances:
                            distance = self.distances[pair]
                            energy_transfer_ids = et_pairs[
                                f"{donor_fluorophore.name}, {acceptor_fluorophore.name}, {distance}"
                            ]

                            transitions[
                                f"D: {donor_fluorophore.name}, A: {acceptor_fluorophore.name}, dist: {distance}"
                            ] = derive_energy_transfer_transitions(
                                donor_data=donor_data,
                                acceptor_data=acceptor_fluorophore.constants,
                                fluorophore_ids=energy_transfer_ids,
                                distance=distance,
                                **energy_transfer_parameters,
                            )

        return transitions

    def plot(self, quadratic: bool = True, **kwargs: Any) -> fi.AxesArray:
        """
        Plot the positions of fluorophores.

        Parameters
        ----------
        quadratic
            Whether to display the plot with same x and y axis scaling.
        kwargs
            fluopy.figure.universal_figure arguments

        Returns
        -------
        axes
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.fluorophores[0].position.shape[0] != 2:
            raise ValueError("Only 2D positions can be plotted.")
        positions = np.empty(shape=(2, self.count), dtype=float)
        labels: list[str] = []
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


def get_distances(
    positions: npt.ArrayLike,
) -> dict[tuple[int, int], np.float64]:
    """
    Gets distances between positions.

    Parameters
    ----------
    positions
        Contains coordinate pairs (2D).

    Returns
    -------
    distances : dict[tuple[int, int], np.float64]
        Contains tuples of ids (order as positions) as keys and their distance as
        values.
    """
    distances: dict[tuple[int, int], np.float64] = {}
    positions = np.asarray(positions)
    for i, position_1 in enumerate(positions):
        for j, position_2 in enumerate(positions):
            if i != j and (i, j) not in distances:
                distances[(i, j)] = np.round(
                    np.linalg.norm(position_1 - position_2), decimals=3
                )

    return distances


def triangle_third_position(
    position_1: npt.ArrayLike | None = None, position_2: npt.ArrayLike | None = None
) -> npt.NDArray[np.float64]:
    """
    Get the third position of an equilateral triangle based on positions of two
    vertices. There are two solutions to such a position but only one is considered
    here.

    Parameters
    ----------
    position_1
        The position of the first vertex. Shape (2,).
    position_2
        The position of the second vertex. Shape (2,).

    Returns
    -------
    npt.NDArray[np.float64]
        The position of the third vertex. Shape (2,).
    """
    if position_1 is None:
        position_1 = np.array([0, 0])
    else:
        position_1 = np.asarray(position_1)
    if position_2 is None:
        position_2 = np.array([0, 10])
    else:
        position_2 = np.asarray(position_2)
    x1, y1 = position_1
    x2, y2 = position_2
    x3 = (x1 + x2 + np.sqrt(3) * (y1 - y2)) / 2
    y3 = (y1 + y2 + np.sqrt(3) * (x1 - x2)) / -2
    position_3 = np.array([x3, y3])

    return position_3


def get_positions_from_distance(
    distance: float = 1,
    count: int = 1,
    shape: Literal["triangle", "square"] = "triangle",
) -> npt.NDArray[np.float64]:
    """
    Gets positions of up to 4 fluorophores based on a single distance. If it is 3
    fluorophores, they are positioned either in an equilateral triangle or in a square
    with a missing vertex. If it is 4 fluorophores, they are positioned in a square.

    Parameters
    ----------
    distance
        Minimum distance between fluorophores.
    count
        Number of fluorophores.
    shape
        One of 'triangle', 'square'. Only needed if count is 3.

    Returns
    -------
    positions : npt.NDArray[np.float64]
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
        else:  # count == 4:
            position_3 = np.array([0, distance])
            position_4 = np.array([distance, distance])
            positions = np.array([position_1, position_2, position_3, position_4])
    else:
        logger.warning(
            "If count is above 4, all fluorophores are positioned at the same"
            " location. This indicates no support for energy transfers.",
            stacklevel=2,
        )
        positions = np.array([[position_1[0], position_1[1] + i] for i in range(count)])

    return positions


def construct_fluorophores(
    name: str = "cy5",
    distance: float = 10,
    count: int = 3,
    shape: Literal["triangle", "square"] = "triangle",
) -> list[Fluorophore]:
    """
    Constructs up to 4 fluorophores of the same kind.

    Parameters
    ----------
    name
        Name of the fluorophore.
    distance
        Minimum distance between fluorophores.
    count
        Number of fluorophores.
    shape
        One of 'triangle', 'square'. Only needed if count is 3.

    Returns
    -------
    fluorophores : list[Fluorophore]
        Contains fluorophores of type Fluorophore.
    """
    positions = get_positions_from_distance(distance=distance, count=count, shape=shape)
    fluorophores = [Fluorophore(name=name, position=position) for position in positions]

    return fluorophores
