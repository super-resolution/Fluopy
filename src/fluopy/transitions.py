"""
Define and handle photophysical transitions.
"""

from __future__ import annotations

import copy
import logging
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from itertools import product
from numbers import Real
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Self

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import _graphs as net
from . import photophysics as fo
from ._statistics import parse_paired_transition_label
from .fluo_data import FluorophoreData, Spectrum

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes

    from fluopy.fluorophores import Fluorophore, FluorophoreSystem


__all__: list[str] = [
    "SingleState",
    "BUILTIN_SINGLE_STATES",
    "PairedState",
    "BUILTIN_PAIRED_STATES",
    "TransitionType",
    "BUILTIN_TRANSITION_TYPES",
    "Transition",
    "TransitionSet",
]

logger = logging.getLogger(__name__)


StateCombination = tuple[int, ...]
TransitionRateRecord = list[object]


@dataclass(frozen=True, slots=True)
class SingleState:
    """
    Contains the name and numerical identifier of a photophysical state.

    Attributes
    ----------
    name
        Name of the state.
    value
        Unique numerical identifier of the state.
    """

    name: str
    value: int

    S0: ClassVar[SingleState]
    S1: ClassVar[SingleState]
    S2: ClassVar[SingleState]
    T1: ClassVar[SingleState]
    T2: ClassVar[SingleState]
    B: ClassVar[SingleState]
    cis: ClassVar[SingleState]
    OFF: ClassVar[SingleState]
    OFF2: ClassVar[SingleState]
    R: ClassVar[SingleState]


SingleState.S0 = SingleState("S0", 0)
SingleState.S1 = SingleState("S1", 1)
SingleState.S2 = SingleState("S2", 2)
SingleState.T1 = SingleState("T1", 3)
SingleState.T2 = SingleState("T2", 4)
SingleState.B = SingleState("B", 5)
SingleState.cis = SingleState("cis", 6)
SingleState.OFF = SingleState("OFF", 7)
SingleState.OFF2 = SingleState("OFF2", 8)
SingleState.R = SingleState("R", 9)

BUILTIN_SINGLE_STATES = (
    SingleState.S0,
    SingleState.S1,
    SingleState.S2,
    SingleState.T1,
    SingleState.T2,
    SingleState.B,
    SingleState.cis,
    SingleState.OFF,
    SingleState.OFF2,
    SingleState.R,
)


@dataclass(frozen=True, slots=True)
class PairedState:
    """
    Contains the two component states of a paired transition.

    Built-in paired states are available as class attributes and in
    BUILTIN_PAIRED_STATES. Additional paired states can be constructed directly.

    PairedState describes the structure of a transition and does not require the two
    components to form a physical donor-acceptor pair. The donor and acceptor names
    identify their order because most supported pair mechanisms naturally use these
    roles. The interaction or rate mechanism, such as FRET or PET, is specified by
    TransitionType.mechanism.

    Attributes
    ----------
    name
        Name of the paired state.
    donor
        State of the first component, conventionally the donor or active component.
        Paired-transition frequencies are assigned to this component's transition
        group during analysis and prediction.
    acceptor
        State of the second component, conventionally the acceptor or passive
        component.
    """

    name: str
    donor: SingleState
    acceptor: SingleState

    S1_S0: ClassVar[PairedState]
    S0_S1: ClassVar[PairedState]
    S1_T1: ClassVar[PairedState]
    S1_Cis: ClassVar[PairedState]
    S0_Cis: ClassVar[PairedState]
    S1_OFF: ClassVar[PairedState]
    S0_S0: ClassVar[PairedState]
    S0_T2: ClassVar[PairedState]
    S1_S1: ClassVar[PairedState]
    S0_T1: ClassVar[PairedState]
    S0_OFF2: ClassVar[PairedState]
    S0_OFF: ClassVar[PairedState]
    S0_B: ClassVar[PairedState]
    S1_R: ClassVar[PairedState]
    S0_R: ClassVar[PairedState]

    def __post_init__(self) -> None:
        """Validate the first and second component states."""
        if not isinstance(self.donor, SingleState) or not isinstance(
            self.acceptor, SingleState
        ):
            raise TypeError("donor and acceptor must both be SingleState objects.")

    @property
    def value(self) -> tuple[SingleState, SingleState]:
        """
        Return the donor and acceptor states.
        """
        return self.donor, self.acceptor

    @property
    def single_state_values(self) -> tuple[int, int]:
        """
        Return the numerical donor and acceptor state values.
        """
        return self.donor.value, self.acceptor.value


PairedState.S1_S0 = PairedState("S1_S0", SingleState.S1, SingleState.S0)
PairedState.S0_S1 = PairedState("S0_S1", SingleState.S0, SingleState.S1)
PairedState.S1_T1 = PairedState("S1_T1", SingleState.S1, SingleState.T1)
PairedState.S1_Cis = PairedState("S1_Cis", SingleState.S1, SingleState.cis)
PairedState.S0_Cis = PairedState("S0_Cis", SingleState.S0, SingleState.cis)
PairedState.S1_OFF = PairedState("S1_OFF", SingleState.S1, SingleState.OFF)
PairedState.S0_S0 = PairedState("S0_S0", SingleState.S0, SingleState.S0)
PairedState.S0_T2 = PairedState("S0_T2", SingleState.S0, SingleState.T2)
PairedState.S1_S1 = PairedState("S1_S1", SingleState.S1, SingleState.S1)
PairedState.S0_T1 = PairedState("S0_T1", SingleState.S0, SingleState.T1)
PairedState.S0_OFF2 = PairedState("S0_OFF2", SingleState.S0, SingleState.OFF2)
PairedState.S0_OFF = PairedState("S0_OFF", SingleState.S0, SingleState.OFF)
PairedState.S0_B = PairedState("S0_B", SingleState.S0, SingleState.B)
PairedState.S1_R = PairedState("S1_R", SingleState.S1, SingleState.R)
PairedState.S0_R = PairedState("S0_R", SingleState.S0, SingleState.R)


BUILTIN_PAIRED_STATES = (
    PairedState.S1_S0,
    PairedState.S0_S1,
    PairedState.S1_T1,
    PairedState.S1_Cis,
    PairedState.S0_Cis,
    PairedState.S1_OFF,
    PairedState.S0_S0,
    PairedState.S0_T2,
    PairedState.S1_S1,
    PairedState.S0_T1,
    PairedState.S0_OFF2,
    PairedState.S0_OFF,
    PairedState.S0_B,
    PairedState.S1_R,
    PairedState.S0_R,
)


@dataclass(frozen=True, slots=True)
class TransitionType:
    """
    Contains constant attributes of a photophysical transition.

    Built-in transition types are available as class attributes and in
    BUILTIN_TRANSITION_TYPES. Additional transition types can be constructed
    directly.

    Attributes
    ----------
    abbreviation
        Abbreviation of the transition.
    initial_state
        Initial state of the transition.
    final_state
        Final state of the transition.
    photon
        Whether the transition emits a photon.
    mechanism
        Non-empty name of the interaction or rate mechanism for a paired transition.
        Single-state transitions use None.
    """

    abbreviation: str
    initial_state: SingleState | PairedState
    final_state: SingleState | PairedState
    photon: bool
    mechanism: str | None = None

    EXCITATION: ClassVar[TransitionType]
    FLUORESCENT_EMISSION: ClassVar[TransitionType]
    SINGLET_QUENCHING: ClassVar[TransitionType]
    INTERSYSTEM_CROSSING_ST: ClassVar[TransitionType]
    INTERSYSTEM_CROSSING_TS: ClassVar[TransitionType]
    INTERNAL_CONVERSION_S: ClassVar[TransitionType]
    REVERSE_INTERSYSTEM_CROSSING: ClassVar[TransitionType]
    PHOTOBLEACHING_1: ClassVar[TransitionType]
    PHOTOBLEACHING_2: ClassVar[TransitionType]

    ET_CYCLE_T: ClassVar[TransitionType]
    ET_CYCLE_S: ClassVar[TransitionType]
    ADDUCT_T: ClassVar[TransitionType]
    ADDUCT_S: ClassVar[TransitionType]
    THERM_ELIMINATION: ClassVar[TransitionType]
    PHOTO_UNCAGING: ClassVar[TransitionType]
    RAD_ESCAPE: ClassVar[TransitionType]
    RAD_RELAX: ClassVar[TransitionType]

    ISOMERIZATION: ClassVar[TransitionType]
    PHOTO_BISO: ClassVar[TransitionType]
    THERM_BISO: ClassVar[TransitionType]

    FRET: ClassVar[TransitionType]
    CIS_FRET_1: ClassVar[TransitionType]
    CIS_FRET_2: ClassVar[TransitionType]
    OFF_FRET_1: ClassVar[TransitionType]
    OFF_FRET_2: ClassVar[TransitionType]
    S_S_ANNIHILATION: ClassVar[TransitionType]
    S_T_ANNIHILATION: ClassVar[TransitionType]
    S_T_ANNI_RISC: ClassVar[TransitionType]
    S_T_ANNI_BLEACH: ClassVar[TransitionType]
    R_FRET_1: ClassVar[TransitionType]
    R_FRET_2: ClassVar[TransitionType]

    H2O_ATTACK_S: ClassVar[TransitionType]
    H2O_ATTACK_T: ClassVar[TransitionType]
    BACK_REACTION: ClassVar[TransitionType]

    S1_S0_TRANSITIONS: ClassVar[TransitionType]
    CIS_S0_TRANSITIONS: ClassVar[TransitionType]
    T1_S0_TRANSITIONS: ClassVar[TransitionType]
    OFF_S0_TRANSITIONS: ClassVar[TransitionType]

    def __post_init__(self) -> None:
        """Validate the state kinds and paired-transition mechanism."""
        states = (self.initial_state, self.final_state)
        if not all(isinstance(state, SingleState | PairedState) for state in states):
            raise TypeError(
                "initial_state and final_state must be SingleState or PairedState."
            )
        if isinstance(self.initial_state, SingleState) != isinstance(
            self.final_state, SingleState
        ):
            raise TypeError(
                "initial_state and final_state must both be SingleState or both be "
                "PairedState."
            )
        if isinstance(self.initial_state, PairedState):
            if not isinstance(self.mechanism, str) or not self.mechanism:
                raise ValueError(
                    "a paired transition type must specify a non-empty mechanism."
                )
        elif self.mechanism is not None:
            raise ValueError("a single-state transition type must use mechanism=None.")


# general
TransitionType.EXCITATION = TransitionType("EXC", SingleState.S0, SingleState.S1, False)
TransitionType.FLUORESCENT_EMISSION = TransitionType(
    "FLU", SingleState.S1, SingleState.S0, True
)
TransitionType.SINGLET_QUENCHING = TransitionType(
    "SQ", SingleState.S1, SingleState.S0, False
)
TransitionType.INTERSYSTEM_CROSSING_ST = TransitionType(
    "ISC_ST", SingleState.S1, SingleState.T1, False
)
TransitionType.INTERSYSTEM_CROSSING_TS = TransitionType(
    "ISC_TS", SingleState.T1, SingleState.S0, False
)
TransitionType.INTERNAL_CONVERSION_S = TransitionType(
    "IC", SingleState.S1, SingleState.S0, False
)
TransitionType.REVERSE_INTERSYSTEM_CROSSING = TransitionType(
    "RISC", SingleState.T1, SingleState.S1, False
)
TransitionType.PHOTOBLEACHING_1 = TransitionType(
    "BLE", SingleState.T1, SingleState.B, False
)
TransitionType.PHOTOBLEACHING_2 = TransitionType(
    "BLE2", SingleState.T2, SingleState.B, False
)

# dstorm
TransitionType.ET_CYCLE_T = TransitionType(
    "PET_TS", SingleState.T1, SingleState.S0, False
)
TransitionType.ET_CYCLE_S = TransitionType(
    "PET_SS", SingleState.S1, SingleState.S0, False
)
TransitionType.ADDUCT_T = TransitionType(
    "PET_TO", SingleState.T1, SingleState.OFF, False
)
TransitionType.ADDUCT_S = TransitionType(
    "PET_SO", SingleState.S1, SingleState.OFF, False
)
TransitionType.THERM_ELIMINATION = TransitionType(
    "TE", SingleState.OFF, SingleState.S0, False
)
TransitionType.PHOTO_UNCAGING = TransitionType(
    "PU", SingleState.OFF, SingleState.S0, False
)
TransitionType.RAD_ESCAPE = TransitionType(
    "PET_TR", SingleState.T1, SingleState.R, False
)
TransitionType.RAD_RELAX = TransitionType("OXI", SingleState.R, SingleState.S0, False)

# cis trans isomerization
TransitionType.ISOMERIZATION = TransitionType(
    "ISO", SingleState.S1, SingleState.cis, False
)
TransitionType.PHOTO_BISO = TransitionType(
    "PBISO", SingleState.cis, SingleState.S0, False
)
TransitionType.THERM_BISO = TransitionType(
    "TBISO", SingleState.cis, SingleState.S0, False
)

# paired transitions
TransitionType.FRET = TransitionType(
    "FRET", PairedState.S1_S0, PairedState.S0_S1, False, mechanism="FRET"
)
TransitionType.CIS_FRET_1 = TransitionType(
    "CET_1", PairedState.S1_Cis, PairedState.S0_Cis, False, mechanism="FRET"
)
TransitionType.CIS_FRET_2 = TransitionType(
    "CET_2", PairedState.S1_Cis, PairedState.S0_S0, False, mechanism="FRET"
)
TransitionType.OFF_FRET_1 = TransitionType(
    "OET_1", PairedState.S1_OFF, PairedState.S0_OFF, False, mechanism="FRET"
)
TransitionType.OFF_FRET_2 = TransitionType(
    "OET_2", PairedState.S1_OFF, PairedState.S0_S0, False, mechanism="FRET"
)
TransitionType.S_S_ANNIHILATION = TransitionType(
    "SSA",
    PairedState.S1_S1,
    PairedState.S0_S1,
    False,
    mechanism="FRET",
)
TransitionType.S_T_ANNIHILATION = TransitionType(
    "STA",
    PairedState.S1_T1,
    PairedState.S0_T1,
    False,
    mechanism="FRET",
)
TransitionType.S_T_ANNI_RISC = TransitionType(
    "STA_2",
    PairedState.S1_T1,
    PairedState.S0_S1,
    False,
    mechanism="FRET",
)
TransitionType.S_T_ANNI_BLEACH = TransitionType(
    "STA_B",
    PairedState.S1_T1,
    PairedState.S0_B,
    False,
    mechanism="FRET",
)
TransitionType.R_FRET_1 = TransitionType(
    "RET_1", PairedState.S1_R, PairedState.S0_R, False, mechanism="FRET"
)
TransitionType.R_FRET_2 = TransitionType(
    "RET_2", PairedState.S1_R, PairedState.S0_S0, False, mechanism="FRET"
)

# rhodamines
TransitionType.H2O_ATTACK_S = TransitionType(
    "H2OS", SingleState.S1, SingleState.OFF, False
)
TransitionType.H2O_ATTACK_T = TransitionType(
    "H2OT", SingleState.T1, SingleState.OFF, False
)
TransitionType.BACK_REACTION = TransitionType(
    "BR", SingleState.OFF, SingleState.S0, False
)

# summarize
TransitionType.S1_S0_TRANSITIONS = TransitionType(
    "S1S0SUM", SingleState.S1, SingleState.S0, False
)
TransitionType.CIS_S0_TRANSITIONS = TransitionType(
    "cisS0SUM", SingleState.cis, SingleState.S0, False
)
TransitionType.T1_S0_TRANSITIONS = TransitionType(
    "T1S0SUM", SingleState.T1, SingleState.S0, False
)
TransitionType.OFF_S0_TRANSITIONS = TransitionType(
    "OFFS0SUM", SingleState.OFF, SingleState.S0, False
)


BUILTIN_TRANSITION_TYPES = (
    TransitionType.EXCITATION,
    TransitionType.FLUORESCENT_EMISSION,
    TransitionType.SINGLET_QUENCHING,
    TransitionType.INTERSYSTEM_CROSSING_ST,
    TransitionType.INTERSYSTEM_CROSSING_TS,
    TransitionType.INTERNAL_CONVERSION_S,
    TransitionType.REVERSE_INTERSYSTEM_CROSSING,
    TransitionType.PHOTOBLEACHING_1,
    TransitionType.PHOTOBLEACHING_2,
    TransitionType.ET_CYCLE_T,
    TransitionType.ET_CYCLE_S,
    TransitionType.ADDUCT_T,
    TransitionType.ADDUCT_S,
    TransitionType.THERM_ELIMINATION,
    TransitionType.PHOTO_UNCAGING,
    TransitionType.RAD_ESCAPE,
    TransitionType.RAD_RELAX,
    TransitionType.ISOMERIZATION,
    TransitionType.PHOTO_BISO,
    TransitionType.THERM_BISO,
    TransitionType.FRET,
    TransitionType.CIS_FRET_1,
    TransitionType.CIS_FRET_2,
    TransitionType.OFF_FRET_1,
    TransitionType.OFF_FRET_2,
    TransitionType.S_S_ANNIHILATION,
    TransitionType.S_T_ANNIHILATION,
    TransitionType.S_T_ANNI_RISC,
    TransitionType.S_T_ANNI_BLEACH,
    TransitionType.R_FRET_1,
    TransitionType.R_FRET_2,
    TransitionType.H2O_ATTACK_S,
    TransitionType.H2O_ATTACK_T,
    TransitionType.BACK_REACTION,
    TransitionType.S1_S0_TRANSITIONS,
    TransitionType.CIS_S0_TRANSITIONS,
    TransitionType.T1_S0_TRANSITIONS,
    TransitionType.OFF_S0_TRANSITIONS,
)


@dataclass(slots=True)
class Transition:
    """
    Contains constant and variable attributes of photophysical transitions.

    Attributes
    ----------
    identity
        The id of the transition. Not None if transition is part of a TransitionSet.
    transition_type
        The photophysical type of the transitions with its constant attributes.
    abbreviation
        The abbreviation of the transition.
    initial_state
        The initial state of the transition.
    final_state
        The final state of the transition.
    rate
        The rate of the transition.
    photon
        Whether the transition emits a photon.
    fluorophore_ids
        Immutable sequence containing the identities of relevant fluorophores.
        For a paired transition, tuples of fluorophore pairs corresponding to the
        ordered components of its PairedState.
    """

    identity: int | None = field(init=False, default=None)
    transition_type: TransitionType = field()
    abbreviation: str = field(init=False)
    initial_state: SingleState | PairedState = field(init=False)
    final_state: SingleState | PairedState = field(init=False)
    rate: float = field()
    photon: bool = field(init=False)
    fluorophore_ids: Sequence[int] | Sequence[tuple[int, int]] = field()

    def __setattr__(self, name: str, value: object) -> None:
        """
        Set an attribute while preventing replacement of fluorophore_ids.

        Parameters
        ----------
        name
            Attribute name.
        value
            Value assigned to the attribute.
        """
        if name == "fluorophore_ids" and hasattr(self, "fluorophore_ids"):
            raise AttributeError("fluorophore_ids is read-only.")
        object.__setattr__(self, name, value)

    def __post_init__(self) -> None:
        """Validate and normalize the transition attributes."""
        if not isinstance(self.transition_type, TransitionType):
            raise TypeError("transition_type must be a TransitionType object.")
        if not isinstance(self.rate, Real):
            raise ValueError("rate must be a finite, non-negative scalar.")

        rate = float(self.rate)
        if not np.isfinite(rate) or rate < 0:
            raise ValueError("rate must be a finite, non-negative scalar.")

        self.rate = rate
        object.__setattr__(self, "fluorophore_ids", tuple(self.fluorophore_ids))
        if not self.fluorophore_ids:
            raise ValueError("fluorophore_ids must not be empty.")
        self.abbreviation = self.transition_type.abbreviation
        self.initial_state = self.transition_type.initial_state
        self.final_state = self.transition_type.final_state
        self.photon = self.transition_type.photon
        for fluorophore_id in self.fluorophore_ids:
            if isinstance(self.initial_state, PairedState):
                if (
                    not isinstance(fluorophore_id, tuple)
                    or len(fluorophore_id) != 2
                    or not all(isinstance(value, int) for value in fluorophore_id)
                ):
                    raise ValueError(
                        f"{self.abbreviation} is a paired transition, "
                        "fluorophore_ids has to be a sequence of fluorophore "
                        "identity pairs."
                    )
            else:
                if not isinstance(fluorophore_id, int):
                    raise ValueError(
                        f"{self.abbreviation} is a single-state transition, "
                        "fluorophore_ids has to be a sequence of ints."
                    )
        if len(set(self.fluorophore_ids)) != len(self.fluorophore_ids):
            raise ValueError("fluorophore_ids must not contain duplicates.")

    def to_dict(self) -> dict[str, object]:
        """
        Return the transition fields as a shallow dictionary.
        """
        return {item.name: getattr(self, item.name) for item in fields(self)}

    def get_identity(self) -> int:
        """
        Return the identity assigned by the containing TransitionSet.

        Returns
        -------
        int
            Transition identity.
        """
        if self.identity is None:
            raise RuntimeError(
                "transition identity is only available after adding it "
                "to a TransitionSet."
            )
        return self.identity

    def get_single_fluorophore_ids(self) -> list[int]:
        """
        Return the fluorophore identities of a non-paired transition.

        Returns
        -------
        list[int]
            Fluorophore identities.
        """
        fluorophore_ids: list[int] = []

        for fluorophore_id in self.fluorophore_ids:
            if not isinstance(fluorophore_id, int):
                raise RuntimeError(
                    "a non-paired transition must contain integer "
                    "fluorophore identities."
                )
            fluorophore_ids.append(fluorophore_id)

        return fluorophore_ids

    def get_fluorophore_pairs(self) -> list[tuple[int, int]]:
        """
        Return the fluorophore pairs of a paired transition.

        Returns
        -------
        list[tuple[int, int]]
            Ordered first- and second-component identity pairs.
        """
        fluorophore_pairs: list[tuple[int, int]] = []

        for fluorophore_pair in self.fluorophore_ids:
            if not isinstance(fluorophore_pair, tuple):
                raise RuntimeError(
                    "a paired transition must contain fluorophore identity pairs."
                )
            fluorophore_pairs.append(fluorophore_pair)

        return fluorophore_pairs


def get_states_by_value(
    transitions: dict[str, list[Transition]],
) -> dict[int, SingleState]:
    """
    Collect single states by value and reject ambiguous names or values.

    Parameters
    ----------
    transitions
        Contains lists of transitions as values and fluorophore names or
        fluorophore-combination as keys. Fluorophore-combination keys require the
        format 'D: {name of donor}, A: {name of acceptor}, dist: {distance between them
        in nm}'.

    Returns
    -------
    dict[int, SingleState]
        Built-in and custom single states indexed by their numerical value.
    """
    states_by_value = {state.value: state for state in BUILTIN_SINGLE_STATES}
    values_by_name = {state.name: state.value for state in BUILTIN_SINGLE_STATES}

    for transition_collection in transitions.values():
        for transition in transition_collection:
            for state in (
                transition.initial_state,
                transition.final_state,
            ):
                single_states: tuple[SingleState, ...]
                if isinstance(state, PairedState):
                    single_states = state.value
                else:
                    single_states = (state,)

                for single_state in single_states:
                    existing_state = states_by_value.get(single_state.value)
                    if (
                        existing_state is not None
                        and existing_state.name != single_state.name
                    ):
                        raise ValueError(
                            f"state value {single_state.value} is already "
                            f"assigned to {existing_state.name}, and cannot "
                            f"be assigned to {single_state.name}."
                        )

                    existing_value = values_by_name.get(single_state.name)
                    if (
                        existing_value is not None
                        and existing_value != single_state.value
                    ):
                        raise ValueError(
                            f"state name {single_state.name} is already "
                            f"assigned to value {existing_value}, and cannot "
                            f"be assigned to {single_state.value}."
                        )

                    states_by_value[single_state.value] = single_state
                    values_by_name[single_state.name] = single_state.value

    return states_by_value


class TransitionSet:
    """
    Collection of all relevant transitions and related attributes. Allows optional
    post-init-modification and (subsequent) finalization.

    Attributes
    ----------
    transitions : dict[str, list[Transition]]
        Contains lists of retained transitions of type Transition as values and
        fluorophores or fluorophore-combinations as keys. Fluorophore-combination keys
        require the format 'D: {name of donor}, A: {name of acceptor}, dist: {distance
        between them in nm}'.
    fluorophore_system : fluopy.fluorophores.FluorophoreSystem
        Container for attributes of multiple, interrelated fluorophores.
    combined_state_transitions_df : pd.DataFrame
        Contains realizable combined_state_transitions with their id as index and their
        other attributes as columns.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum
        of rates of all possible combined_state_transitions.
    single_states : dict[str, npt.NDArray[np.int64]]
        Contains the values of all relevant SingleStates as values. Name of
        fluorophores as keys.
    absorbing_states : Mapping[int, npt.NDArray[np.int64]]
        Individually absorbing state values indexed by fluorophore identity. These are
        derived from the retained transition topology, including transitions retained
        with rate 0.
    terminal_state_combinations : frozenset[tuple[int, ...]]
        Combined states without a positive-rate outgoing transition. Unlike
        absorbing_states, this collection is derived from the active rates.
    transition_df : pd.DataFrame
        Dataframe of all retained transitions containing their id as second level index
        and their other attributes as columns. Name of fluorophores as first level
        index. Includes zero-rate transitions when keep_zero_rates is True.
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each
        possible combined_state_transition at the corresponding index pair.
    """

    _combined_state_transitions_df: pd.DataFrame | None
    _row_sums: npt.NDArray[np.float64] | None
    _transition_matrix: npt.NDArray[np.float64] | None
    _terminal_state_combinations: frozenset[StateCombination] | None
    _absorbing_states: dict[int, npt.NDArray[np.int64]]

    def __init__(
        self,
        transitions: dict[str, list[Transition]],
        fluorophore_system: FluorophoreSystem,
        keep_zero_rates: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        transitions
            Contains lists of transitions of type Transition as values and fluorophores
            or fluorophore-combinations as keys. Fluorophore-combination keys require
            the format 'D: {name of donor}, A: {name of acceptor}, dist: {distance
            between them in nm}'.
        fluorophore_system
            Container for attributes of multiple, interrelated fluorophores.
        keep_zero_rates
            Whether to retain transitions with rate 0. Retained zero-rate transitions
            remain part of the state space and structural absorbing-state detection,
            but do not contribute to the active transition rates.
        """
        transitions = {
            key: [copy.copy(transition) for transition in transition_collection]
            for key, transition_collection in transitions.items()
        }
        self.fluorophore_system = fluorophore_system

        self.transition_df = pd.DataFrame()
        retained_transitions: dict[str, list[Transition]] = {}
        i = 0
        for fluorophore_comb, f_transitions in transitions.items():
            keep_transitions = []
            df_constructor = []
            for transition in f_transitions:
                if isinstance(transition.initial_state, PairedState):
                    paired_label = parse_paired_transition_label(fluorophore_comb)
                    if paired_label is None:
                        raise ValueError(
                            "paired transitions have to be defined with "
                            "the key 'D: {name of donor}, A: {name of acceptor}, dist: "
                            "{distance between them in nm}'."
                        )
                    d, a, dist = paired_label
                    for d_t, a_t in transition.get_fluorophore_pairs():
                        if not 0 <= d_t < self.fluorophore_system.count:
                            raise ValueError(
                                f"fluorophore identity {d_t} is outside the system."
                            )
                        if not 0 <= a_t < self.fluorophore_system.count:
                            raise ValueError(
                                f"fluorophore identity {a_t} is outside the system."
                            )
                        if d_t == a_t:
                            raise ValueError(
                                "paired transitions require two distinct fluorophore "
                                "identities."
                            )
                        if self.fluorophore_system.fluorophores[d_t].name != d:
                            raise ValueError(
                                f"{d} indicated to be at identity {d_t}, "
                                f"{self.fluorophore_system.fluorophores[d_t].name} "
                                "found."
                            )
                        elif self.fluorophore_system.fluorophores[a_t].name != a:
                            raise ValueError(
                                f"{a} indicated to be at identity {a_t}, "
                                f"{self.fluorophore_system.fluorophores[a_t].name} "
                                "found."
                            )
                        actual_dist = self.fluorophore_system.distances[(d_t, a_t)]
                        if float(dist) != actual_dist:
                            raise ValueError(
                                f"{dist} nm indicated, {actual_dist} nm found."
                            )
                else:
                    for fluorophore_id in transition.get_single_fluorophore_ids():
                        if not 0 <= fluorophore_id < self.fluorophore_system.count:
                            raise ValueError(
                                f"fluorophore identity {fluorophore_id} is outside "
                                "the system."
                            )
                        if (
                            self.fluorophore_system.fluorophores[fluorophore_id].name
                            != fluorophore_comb
                        ):
                            raise ValueError(
                                f"{fluorophore_comb} indicated to be at identity {fluorophore_id}, "
                                f"{self.fluorophore_system.fluorophores[fluorophore_id].name} found."
                            )
                if not keep_zero_rates:
                    if transition.rate != 0:
                        transition.identity = i
                        i += 1
                        keep_transitions.append(transition)
                        df_constructor.append(transition.to_dict())
                else:
                    transition.identity = i
                    i += 1
                    keep_transitions.append(transition)
                    df_constructor.append(transition.to_dict())
            if keep_transitions:
                retained_transitions[fluorophore_comb] = keep_transitions
                transition_df = pd.DataFrame(df_constructor)
                transition_df = transition_df.set_index("identity")
                transition_df = pd.concat(
                    {fluorophore_comb: transition_df}, names=["Fluorophore"]
                )
                self.transition_df = pd.concat([self.transition_df, transition_df])
        self.transitions = retained_transitions
        if self.transition_df.empty:
            self.transition_df = pd.DataFrame(
                columns=[
                    item.name for item in fields(Transition) if item.name != "identity"
                ],
                index=pd.MultiIndex.from_tuples([], names=["Fluorophore", "identity"]),
            )
        self.states_by_value = get_states_by_value(self.transitions)
        self.single_states = get_single_states(
            self.transitions,
            self.fluorophore_system,
        )
        absorbing_states = get_absorbing_states(
            self.transitions,
            self.single_states,
            self.fluorophore_system,
        )
        self._absorbing_states = absorbing_states
        absorbing_fluorophore_ids = {
            (fluorophore_comb, transition.get_identity()): (
                get_absorbing_fluorophore_ids(transition, absorbing_states)
            )
            for fluorophore_comb, transition_collection in self.transitions.items()
            for transition in transition_collection
        }
        self.transition_df["absorbing_fluorophore_ids"] = pd.Series(
            absorbing_fluorophore_ids,
            dtype=object,
        )
        self.transition_df["absorbing"] = self.transition_df[
            "absorbing_fluorophore_ids"
        ].map(bool)

        self._combined_state_transitions_df = None
        self._row_sums = None
        self._transition_matrix = None
        self._terminal_state_combinations = None

    @property
    def combined_state_transitions_df(self) -> pd.DataFrame:
        """
        Combined-state transition table, finalized on first access.
        """
        if self._combined_state_transitions_df is None:
            self.finalize()
        result = self._combined_state_transitions_df
        if result is None:
            raise RuntimeError(
                "transition set finalization did not create a DataFrame."
            )

        return result

    @property
    def row_sums(self) -> npt.NDArray[np.float64]:
        """
        Transition rates by combined state, finalized on first access.
        """
        if self._row_sums is None:
            self.finalize()
        result = self._row_sums
        if result is None:
            raise RuntimeError("transition set finalization did not create row sums.")

        return result

    @property
    def transition_matrix(self) -> npt.NDArray[np.float64]:
        """
        Transition-probability matrix, finalized on first access.
        """
        if self._transition_matrix is None:
            self.finalize()
        result = self._transition_matrix
        if result is None:
            raise RuntimeError(
                "transition set finalization did not create a transition matrix."
            )

        return result

    @property
    def absorbing_states(self) -> Mapping[int, npt.NDArray[np.int64]]:
        """Individually absorbing state values indexed by fluorophore identity."""
        return MappingProxyType(self._absorbing_states)

    @property
    def terminal_state_combinations(self) -> frozenset[StateCombination]:
        """
        Return combined states without a positive-rate outgoing transition.

        This reflects the autonomous rate matrix. Externally driven transitions, such
        as pulsed excitation in TCSPC, are not represented by this collection while
        their retained zero-rate transition definitions may remain in the structural
        topology.
        """
        if self._terminal_state_combinations is None:
            self.finalize()
        result = self._terminal_state_combinations
        if result is None:
            raise RuntimeError(
                "transition set finalization did not identify terminal states."
            )
        return result

    def filter_by_identity(
        self, remove_list: Collection[int] | None = None, keep_zero_rates: bool = False
    ) -> TransitionSet:
        """
        Returns another TransitionSet with transitions removed by their identity.

        Parameters
        ----------
        remove_list
            Contains identities of type int.
        keep_zero_rates
            Whether to retain transitions with rate 0 in the structural topology.

        Returns
        -------
        TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        transitions = copy.deepcopy(self.transitions)

        if remove_list is None:
            remove_list = []

        filtered_transitions: dict[str, list[Transition]] = {}

        for fluorophore, f_transitions in transitions.items():
            for transition in f_transitions:
                if transition.get_identity() in remove_list:
                    continue
                filtered_transitions.setdefault(fluorophore, []).append(transition)

        filtered = TransitionSet(
            transitions=filtered_transitions,
            fluorophore_system=self.fluorophore_system,
            keep_zero_rates=keep_zero_rates,
        )

        return filtered

    def adjust_rates(
        self,
        change_dict: Mapping[int, float] | None = None,
        keep_zero_rates: bool = False,
    ) -> TransitionSet:
        """
        Returns another TransitionSet with transition rates modified.

        Parameters
        ----------
        change_dict
            Contains identities of transitions as key and rates as values.
        keep_zero_rates
            Whether to retain transitions with rate 0 in the structural topology.

        Returns
        -------
        TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        transitions = copy.deepcopy(self.transitions)

        if change_dict is None:
            change_dict = {}
        for _, f_transitions in transitions.items():
            for transition in f_transitions:
                identity = transition.get_identity()
                if identity not in change_dict:
                    continue
                rate = change_dict[identity]

                if not isinstance(rate, Real):
                    raise ValueError("rate must be a finite, non-negative scalar.")

                float_rate = float(rate)
                if not np.isfinite(float_rate) or float_rate < 0:
                    raise ValueError("rate must be a finite, non-negative scalar.")

                transition.rate = float_rate

        adjusted = TransitionSet(
            transitions=transitions,
            fluorophore_system=self.fluorophore_system,
            keep_zero_rates=keep_zero_rates,
        )

        return adjusted

    def remove_zero_rates(self) -> TransitionSet:
        """
        Returns another TransitionSet with all transitions removed that have a rate
        constant of zero.

        Returns
        -------
        TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        transitions = copy.deepcopy(self.transitions)

        new_transition_set = TransitionSet(
            transitions=transitions,
            fluorophore_system=self.fluorophore_system,
            keep_zero_rates=False,
        )
        return new_transition_set

    def remove_absorbing_states(self, keep_zero_rates: bool = False) -> TransitionSet:
        """
        Returns another TransitionSet that contains no Markovian absorbing states.

        Parameters
        ----------
        keep_zero_rates
            Whether to retain transitions with rate 0 in the structural topology.

        Returns
        -------
        TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        keep_transitions: dict[str, list[Transition]] = {}
        absorbing_states = self.absorbing_states
        for fluorophore, f_transitions in self.transitions.items():
            for transition in f_transitions:
                initial_state = transition.initial_state
                final_state = transition.final_state
                if isinstance(initial_state, PairedState):
                    if not isinstance(final_state, PairedState):
                        raise TypeError(
                            "a paired transition must have a PairedState final state."
                        )
                    fluorophore_ids: Sequence[int] | Sequence[tuple[int, int]] = [
                        (donor_id, acceptor_id)
                        for donor_id, acceptor_id in transition.get_fluorophore_pairs()
                        if not (
                            initial_state.donor != final_state.donor
                            and final_state.donor.value in absorbing_states[donor_id]
                        )
                        and not (
                            initial_state.acceptor != final_state.acceptor
                            and final_state.acceptor.value
                            in absorbing_states[acceptor_id]
                        )
                    ]
                else:
                    if not isinstance(final_state, SingleState):
                        raise TypeError(
                            "a non-paired transition must have a SingleState final "
                            "state."
                        )
                    fluorophore_ids = [
                        identity
                        for identity in transition.get_single_fluorophore_ids()
                        if not (
                            initial_state != final_state
                            and final_state.value in absorbing_states[identity]
                        )
                    ]

                if fluorophore_ids:
                    keep_transitions.setdefault(fluorophore, []).append(
                        Transition(
                            transition_type=transition.transition_type,
                            rate=transition.rate,
                            fluorophore_ids=fluorophore_ids,
                        )
                    )

        no_abs = TransitionSet(
            transitions=keep_transitions,
            fluorophore_system=self.fluorophore_system,
            keep_zero_rates=keep_zero_rates,
        )

        return no_abs

    def remove_paired_transitions(self, keep_zero_rates: bool = False) -> TransitionSet:
        """
        Return a TransitionSet without transitions that specify a paired mechanism.

        Parameters
        ----------
        keep_zero_rates
            Whether to retain transitions with rate 0 in the structural topology.

        Returns
        -------
        TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        transitions = copy.deepcopy(self.transitions)

        keep_transitions: dict[str, list[Transition]] = {}
        for fluorophore, f_transitions in transitions.items():
            retained = [
                transition
                for transition in f_transitions
                if transition.transition_type.mechanism is None
            ]
            if retained:
                keep_transitions[fluorophore] = retained

        no_ets = TransitionSet(
            transitions=keep_transitions,
            fluorophore_system=self.fluorophore_system,
            keep_zero_rates=keep_zero_rates,
        )

        return no_ets

    def finalize(self) -> Self:
        """
        Construct combined_state_transitions_df, transition_matrix and row_sums.

        Returns
        -------
        Self
        """
        if self._combined_state_transitions_df is not None:
            return self

        state_combinations = get_state_combinations(
            single_states=self.single_states,
            fluorophores=self.fluorophore_system.fluorophores,
        )
        combined_state_transitions_with_rates = construct_transition_rate_list(
            transition_df=self.transition_df,
            state_combinations=state_combinations,
        )

        self._combined_state_transitions_df = pd.DataFrame(
            combined_state_transitions_with_rates,
            columns=[
                "initial_state",
                "final_state",
                "fluorophore_ids",
                "abbreviation",
                "transition_id",
                "rate",
                "photon",
                "mechanism",
            ],
        )
        self._combined_state_transitions_df.index.name = "id"

        self._transition_matrix, self._row_sums = construct_transition_matrix(
            combined_state_transitions_df=self._combined_state_transitions_df
        )
        terminal_indices = np.flatnonzero(self._row_sums == 0)
        self._terminal_state_combinations = frozenset(
            self._combined_state_transitions_df.iloc[terminal_indices]["final_state"]
        )

        return self

    def plot(
        self,
        graph_type: str = "shell",
        colors: Sequence[str] | None = None,
        scale: float = 1,
        axes: Iterable[mplAxes] | None = None,
    ) -> list[mplAxes]:
        """
        Plot photophysical system as network/graph.

        Parameters
        ----------
        graph_type
            Specifies network layout. One of 'shell', 'circular', 'planar' or 'kamada'.
        colors
            Contains two colors as Hex values of type str.
        scale
            Factor to scale the figure.
        axes
            Axes elements to plot graphs on.

        Returns
        -------
        list[matplotlib.axes.Axes]
            Axes objects with the plots.
        """
        graphs = net.construct_state_graphs(transition_df=self.transition_df)

        if axes is None:
            plot_axes: Iterable[mplAxes | None] = [None] * len(graphs)
        else:
            plot_axes = axes

        return_axes: list[mplAxes] = []

        try:
            for graph, ax in zip(graphs, plot_axes, strict=True):
                result_ax = net.plot_graph(
                    G=graph,
                    graph_type=graph_type,
                    colors=colors,
                    scale=scale,
                    ax=ax,
                )
                return_axes.append(result_ax)

        except ValueError as exception:
            raise ValueError(
                f"The number of axes elements must be {len(graphs)} or None"
            ) from exception
        return return_axes


def get_single_states(
    transitions: Mapping[str, Collection[Transition]],
    fluorophore_system: FluorophoreSystem,
) -> dict[str, npt.NDArray[np.int64]]:
    """
    Get the values of SingleStates occurring in transitions.

    PairedState components are assigned to their corresponding donor and acceptor
    fluorophore types.

    Parameters
    ----------
    transitions
        Contains retained transitions of type Transition. May include zero-rate
        transitions that define structural pathways.
    fluorophore_system
        Container for attributes of multiple, interrelated fluorophores.

    Returns
    -------
    dict[str, npt.NDArray[np.int64]]
        Contains the values of all relevant SingleStates as values. Name of
        fluorophores as keys.
    """
    single_states: dict[str, list[int]] = {}

    for fluorophore_comb, f_transitions in transitions.items():
        first_transition = next(iter(f_transitions), None)
        if first_transition is None:
            continue
        if not isinstance(first_transition.initial_state, PairedState):
            single_states_: list[int] = []
            for transition in f_transitions:
                initial_state = transition.initial_state
                final_state = transition.final_state
                if not isinstance(initial_state, SingleState):
                    raise TypeError(
                        "a non-paired transition must have a SingleState initial state."
                    )
                if not isinstance(final_state, SingleState):
                    raise TypeError(
                        "a non-paired transition must have a SingleState final state."
                    )
                if initial_state.value not in single_states_:
                    single_states_.append(initial_state.value)
                if final_state.value not in single_states_:
                    single_states_.append(final_state.value)

            single_states[fluorophore_comb] = single_states_
    for f_transitions in transitions.values():
        for transition in f_transitions:
            initial_state = transition.initial_state
            final_state = transition.final_state
            if not isinstance(initial_state, PairedState):
                continue
            if not isinstance(final_state, PairedState):
                raise TypeError(
                    "a paired transition must have a PairedState final state."
                )

            for donor_id, acceptor_id in transition.get_fluorophore_pairs():
                donor_name = fluorophore_system.fluorophores[donor_id].name
                acceptor_name = fluorophore_system.fluorophores[acceptor_id].name

                paired_states = (
                    (
                        donor_name,
                        (initial_state.donor.value, final_state.donor.value),
                    ),
                    (
                        acceptor_name,
                        (initial_state.acceptor.value, final_state.acceptor.value),
                    ),
                )

                for fluorophore_name, states in paired_states:
                    single_states.setdefault(fluorophore_name, [])
                    for state in states:
                        if state not in single_states[fluorophore_name]:
                            single_states[fluorophore_name].append(state)
    single_state_arrays = {
        fluorophore: np.asarray(states, dtype=np.int64)
        for fluorophore, states in single_states.items()
    }

    return single_state_arrays


def get_absorbing_states(
    transitions: Mapping[str, Collection[Transition]],
    single_states: Mapping[str, npt.NDArray[np.int64]],
    fluorophore_system: FluorophoreSystem,
) -> dict[int, npt.NDArray[np.int64]]:
    """
    Get individually absorbing states for every fluorophore identity.

    A state is individually absorbing if it occurs as a final state and no transition
    changes that fluorophore from the state. Paired transitions are evaluated
    separately for their donor and acceptor components. All supplied transitions define
    this structural topology regardless of rate. TransitionSet omits zero-rate
    transitions by default, but includes them here when keep_zero_rates is True so that
    externally driven or temporarily inactive pathways remain represented.

    Parameters
    ----------
    transitions
        Contains retained transitions of type Transition. May include zero-rate
        transitions that define structural pathways.
    single_states
        Contains relevant state values indexed by fluorophore name.
    fluorophore_system
        Container for attributes of multiple, interrelated fluorophores.

    Returns
    -------
    dict[int, npt.NDArray[np.int64]]
        Individually absorbing state values indexed by fluorophore identity.
    """
    final_states: dict[int, set[int]] = {
        identity: set() for identity in range(fluorophore_system.count)
    }
    changing_initial_states: dict[int, set[int]] = {
        identity: set() for identity in range(fluorophore_system.count)
    }

    for transition_collection in transitions.values():
        for transition in transition_collection:
            initial_state = transition.initial_state
            final_state = transition.final_state
            if isinstance(initial_state, PairedState):
                if not isinstance(final_state, PairedState):
                    raise TypeError(
                        "a paired transition must have a PairedState final state."
                    )
                for donor_id, acceptor_id in transition.get_fluorophore_pairs():
                    components = (
                        (donor_id, initial_state.donor, final_state.donor),
                        (acceptor_id, initial_state.acceptor, final_state.acceptor),
                    )
                    for identity, initial_component, final_component in components:
                        final_states[identity].add(final_component.value)
                        if initial_component != final_component:
                            changing_initial_states[identity].add(
                                initial_component.value
                            )
            else:
                if not isinstance(final_state, SingleState):
                    raise TypeError(
                        "a non-paired transition must have a SingleState final state."
                    )
                for identity in transition.get_single_fluorophore_ids():
                    final_states[identity].add(final_state.value)
                    if initial_state != final_state:
                        changing_initial_states[identity].add(initial_state.value)

    absorbing_states: dict[int, npt.NDArray[np.int64]] = {}
    for identity, fluorophore in enumerate(fluorophore_system.fluorophores):
        states = single_states.get(fluorophore.name, np.array([], dtype=np.int64))
        values = np.asarray(
            [
                state
                for state in states
                if state in final_states[identity]
                and state not in changing_initial_states[identity]
            ],
            dtype=np.int64,
        )
        values.setflags(write=False)
        absorbing_states[identity] = values

    return absorbing_states


def get_absorbing_fluorophore_ids(
    transition: Transition,
    absorbing_states: Mapping[int, npt.NDArray[np.int64]],
) -> tuple[int, ...]:
    """
    Return fluorophore identities entering an individually absorbing state.

    The result describes the retained transition topology regardless of rate. A
    retained zero-rate transition can therefore be labeled structurally absorbing even
    though it cannot occur in the autonomous rate matrix.

    Parameters
    ----------
    transition
        Transition whose component-state changes are inspected.
    absorbing_states
        Individually absorbing state values indexed by fluorophore identity.

    Returns
    -------
    tuple[int, ...]
        Identities of fluorophores that change into an individually absorbing state.
    """
    initial_state = transition.initial_state
    final_state = transition.final_state
    absorbing_identities: list[int] = []
    if isinstance(initial_state, PairedState):
        if not isinstance(final_state, PairedState):
            raise TypeError("a paired transition must have a PairedState final state.")
        for donor_id, acceptor_id in transition.get_fluorophore_pairs():
            components = (
                (donor_id, initial_state.donor, final_state.donor),
                (acceptor_id, initial_state.acceptor, final_state.acceptor),
            )
            for identity, initial_component, final_component in components:
                if (
                    initial_component != final_component
                    and final_component.value in absorbing_states[identity]
                    and identity not in absorbing_identities
                ):
                    absorbing_identities.append(identity)
    else:
        if not isinstance(final_state, SingleState):
            raise TypeError(
                "a non-paired transition must have a SingleState final state."
            )
        for identity in transition.get_single_fluorophore_ids():
            if (
                initial_state != final_state
                and final_state.value in absorbing_states[identity]
            ):
                absorbing_identities.append(identity)

    return tuple(absorbing_identities)


def get_state_combinations(
    single_states: Mapping[str, Collection[int]],
    fluorophores: Collection[Fluorophore],
) -> list[tuple[int, ...]]:
    """
    Combines all given states with each other according to the amount and order of the
    respective fluorophore. Cartesian product, see itertools.product().

    Parameters
    ----------
    single_states
        Contains the values of all relevant SingleStates as values. Name of
        fluorophores as keys.
    fluorophores
        Contains all given fluorophores of type Fluorophore.

    Returns
    -------
    list
        Contains state combinations of type tuple.
    """
    single_states_fluorophores = [
        single_states[fluorophore.name] for fluorophore in fluorophores
    ]
    return list(product(*single_states_fluorophores))


def construct_transition_rate_list(
    transition_df: pd.DataFrame,
    state_combinations: Collection[StateCombination],
) -> list[TransitionRateRecord]:
    """
    Construct realizable combined-state transitions.

    Parameters
    ----------
    transition_df
        Dataframe of all retained transitions containing their id as second level index
        and their other attributes as columns. Name of fluorophores as first level
        index.
    state_combinations
        Contains the possible combined states.

    Returns
    -------
    list[TransitionRateRecord]
        Contains lists of each realizable combined-state transition.
    """
    valid_states = set(state_combinations)
    transition_rate_list: list[TransitionRateRecord] = []

    for index, transition in transition_df.iterrows():
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("transition DataFrame must have a two-level index.")

        identity = index[1]
        if not isinstance(identity, int):
            raise TypeError("transition identity must be an integer.")

        initial_state = transition["initial_state"]
        final_state = transition["final_state"]

        if isinstance(initial_state, SingleState):
            source = initial_state.value
            destination = final_state.value
            for current_state in state_combinations:
                for fluorophore_id in transition["fluorophore_ids"]:
                    if current_state[fluorophore_id] != source:
                        continue

                    future_state_values = list(current_state)
                    future_state_values[fluorophore_id] = destination
                    future_state = tuple(future_state_values)
                    if future_state not in valid_states:
                        continue

                    transition_rate_list.append(
                        [
                            current_state,
                            future_state,
                            [fluorophore_id],
                            transition["abbreviation"],
                            identity,
                            transition["rate"],
                            transition["photon"],
                            transition["transition_type"].mechanism,
                        ]
                    )
        else:
            source_donor, source_acceptor = initial_state.single_state_values
            destination_donor, destination_acceptor = final_state.single_state_values
            for current_state in state_combinations:
                for donor, acceptor in transition["fluorophore_ids"]:
                    if (
                        current_state[donor] != source_donor
                        or current_state[acceptor] != source_acceptor
                    ):
                        continue

                    future_state_values = list(current_state)
                    future_state_values[donor] = destination_donor
                    future_state_values[acceptor] = destination_acceptor
                    future_state = tuple(future_state_values)
                    if future_state not in valid_states:
                        continue

                    transition_rate_list.append(
                        [
                            current_state,
                            future_state,
                            [donor, acceptor],
                            transition["abbreviation"],
                            identity,
                            transition["rate"],
                            transition["photon"],
                            transition["transition_type"].mechanism,
                        ]
                    )

    return transition_rate_list


def construct_transition_matrix(
    combined_state_transitions_df: pd.DataFrame,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """
    Constructs a matrix of shape (combined_state_transitions_df.index.size,
    combined_state_transitions_df.index.size). The matrix is non-zero at a position, if
    the first index is a final_state in combined_state_transition_df which is the
    initial_state of the second index. In other words, the matrix is non-zero, if a
    transition (first index or row) can be followed by another transition (second index
    or column).

    Parameters
    ----------
    combined_state_transitions_df
        Contains realizable combined_state_transitions with their id as index and their
        other attributes as columns.

    Returns
    -------
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each
        possible combined_state_transition at the corresponding index pair.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum
        of rates of all possbile combined_state_transitions.
    """
    transition_count = combined_state_transitions_df.index.size
    transition_rate_matrix = np.zeros(
        shape=(transition_count, transition_count), dtype=np.float64
    )

    for index, row in combined_state_transitions_df.iterrows():
        if not isinstance(index, int):
            raise TypeError("combined transition index must be an integer.")

        final_state = row["final_state"]
        matching_rows = combined_state_transitions_df[
            combined_state_transitions_df["initial_state"] == final_state
        ]
        indices = matching_rows.index.to_numpy(dtype=np.intp)
        rates = matching_rows["rate"].to_numpy(dtype=np.float64)

        transition_rate_matrix[index, indices] = rates

    row_sums = transition_rate_matrix.sum(axis=1, dtype=np.float64)
    row_sums_exp = np.tile(np.expand_dims(row_sums, axis=1), reps=row_sums.size)
    mask = np.ma.make_mask(row_sums_exp)
    transition_matrix = np.divide(
        transition_rate_matrix,
        row_sums_exp,
        out=np.zeros_like(transition_rate_matrix),
        where=mask,
    )

    return transition_matrix, row_sums


def derive_energy_transfer_rate(
    donor_data: FluorophoreData,
    acceptor_absorption: Spectrum,
    distance: float,
    dipole_orientation_factor: float = 2 / 3,
    refractive_index: float = 1.33,
) -> float:
    """
    Derive an energy-transfer rate from donor and acceptor spectra.

    Parameters
    ----------
    donor_data
        Contains the donor emission spectrum, quantum yield, and fluorescence
        lifetime.
    acceptor_absorption
        Acceptor absorption spectrum containing molar extinction coefficients.
    distance
        Distance between donor and acceptor in nm.
    dipole_orientation_factor
        Dipole orientation factor of the fluorophore pair.
    refractive_index
        Refractive index of the medium.

    Returns
    -------
    float
        Energy-transfer rate in 1/s.
    """
    donor_emission = donor_data.emission_spectrum
    if donor_emission is None:
        raise ValueError(
            "cannot derive an energy-transfer rate without a donor "
            "emission spectrum."
        )
    if donor_data.FLUORESCENCE_LIFETIME <= 0:
        raise ValueError(
            "donor FLUORESCENCE_LIFETIME must be greater than zero to derive "
            "an energy-transfer rate."
        )
    donor_area = donor_emission.integral()
    if donor_area <= 0:
        raise ValueError("donor emission spectrum must have positive area.")

    minimum = max(
        donor_emission.wavelengths[0],
        acceptor_absorption.wavelengths[0],
    )
    maximum = min(
        donor_emission.wavelengths[-1],
        acceptor_absorption.wavelengths[-1],
    )

    if minimum >= maximum:
        spectral_overlap_integral = 0.0
    else:
        donor_inside = donor_emission.wavelengths[
            (donor_emission.wavelengths > minimum)
            & (donor_emission.wavelengths < maximum)
        ]
        acceptor_inside = acceptor_absorption.wavelengths[
            (acceptor_absorption.wavelengths > minimum)
            & (acceptor_absorption.wavelengths < maximum)
        ]
        wavelengths = np.unique(
            np.concatenate(
                (
                    [minimum],
                    donor_inside,
                    acceptor_inside,
                    [maximum],
                )
            )
        )

        donor_values = np.interp(
            wavelengths,
            donor_emission.wavelengths,
            donor_emission.values,
        )
        acceptor_values = np.interp(
            wavelengths,
            acceptor_absorption.wavelengths,
            acceptor_absorption.values,
        )

        spectral_overlap_integral = fo.calculate_spectral_overlap_integral(
            donor=donor_values,
            acceptor=acceptor_values,
            wavelengths=wavelengths,
            donor_area=donor_area,
        )

    emission_rate = float(
        fo.calculate_emission_rate(
            quantum_yield=donor_data.QUANTUM_YIELD,
            fluorescence_lifetime=donor_data.FLUORESCENCE_LIFETIME,
        )
    )

    rate = fo.calculate_fret_rate(
        distance=distance,
        emission_rate=emission_rate,
        spectral_overlap_integral=spectral_overlap_integral,
        dipole_orientation_factor=dipole_orientation_factor,
        refractive_index=refractive_index,
    )

    rate = float(rate)

    return rate


def derive_energy_transfer_transitions(
    donor_data: FluorophoreData,
    acceptor_data: FluorophoreData,
    fluorophore_ids: list[tuple[int, int]],
    dipole_orientation_factor: float,
    distance: float,
    refractive_index: float,
    overwrite: Mapping[str, Sequence[float]] | None = None,
    exclude: list[str] | None = None,
    include: dict[str, list[tuple[TransitionType, float]]] | None = None,
) -> list[Transition]:
    """
    Derive energy transfer transitions based on the experimental conditions and the
    fluorophore-combinations to be mimicked. The type of energy transfer is determined
    by the state names in acceptor_data.absorption_spectra.

    Parameters
    ----------
    donor_data
        Contains all constant photophysical attributes of the donor.
    acceptor_data
        Contains all constant photophysical attributes of the acceptor.
    fluorophore_ids
        Contains the identities of all fluorophore pairs the transitions apply to as
        tuples.
    dipole_orientation_factor
        The dipole orientation factor of the fluorophore pair.
    distance
        The distance between the fluorophores of the fluorophore pair.
    refractive_index
        The refractive index of the medium.
    overwrite
        Contains the type of acceptor state as key and a list with a factor for the rate
        as well as an efficiency (of not recycling acceptor state) as value.
    exclude
        Contains the type of acceptor state (lowercase) to be excluded.
    include
        Contains the type of acceptor state as key and a list of tuples as values. The
        tuples contain the transition type and an efficiency. If the summed efficiencies
        is e.g., 0.5, all other energy transfers affecting the acceptor state are
        multiplied by 1-0.5.

    Returns
    -------
    list[Transition]
        Contains energy transfer transitions of type Transition.
    """
    acceptor_absorptions = acceptor_data.absorption_spectra
    if not acceptor_absorptions:
        raise ValueError(
            "cannot derive energy-transfer transitions without acceptor "
            "absorption spectra."
        )

    supported_acceptor_states = {"s0", "t1", "s1", "cis", "off"}
    if overwrite is not None:
        for acceptor_state, values in overwrite.items():
            if acceptor_state not in supported_acceptor_states:
                raise ValueError(
                    f"overwrite contains unsupported acceptor state "
                    f"{acceptor_state!r}."
                )
            if len(values) != 2:
                raise ValueError(
                    "overwrite values must contain a rate multiplier and an efficiency."
                )

            rate_multiplier, efficiency = values

            if not np.isfinite(rate_multiplier) or rate_multiplier < 0:
                raise ValueError(
                    "overwrite rate multipliers must be finite and non-negative."
                )
            if not np.isfinite(efficiency) or not 0 <= efficiency <= 1:
                raise ValueError(
                    "overwrite efficiencies must be finite and between 0 and 1."
                )

    if exclude is not None:
        unsupported = set(exclude) - supported_acceptor_states
        if unsupported:
            raise ValueError(
                f"exclude contains unsupported acceptor states: "
                f"{sorted(unsupported)}."
            )

    if include is not None:
        for acceptor_state, included_transitions in include.items():
            if acceptor_state not in supported_acceptor_states:
                raise ValueError(
                    f"include contains unsupported acceptor state "
                    f"{acceptor_state!r}."
                )

            factors = np.asarray(
                [factor for _, factor in included_transitions],
                dtype=float,
            )

            if np.any(~np.isfinite(factors)) or np.any(factors < 0):
                raise ValueError("include factors must be finite and non-negative.")
            if factors.sum() > 1:
                raise ValueError(
                    "include factors for each acceptor state must sum to at most 1."
                )

    which_et: dict[str, list[tuple[TransitionType, float]]] = {
        "s0": [(TransitionType.FRET, 1)],
        "t1": [
            (
                TransitionType.S_T_ANNIHILATION,
                (
                    1 - acceptor_data.STA_EFFICIENCY
                    if overwrite is None or "t1" not in overwrite
                    else 1 - overwrite["t1"][1]
                ),
            ),
            (
                TransitionType.S_T_ANNI_RISC,
                (
                    acceptor_data.STA_EFFICIENCY
                    if overwrite is None or "t1" not in overwrite
                    else overwrite["t1"][1]
                ),
            ),
        ],
        "s1": [(TransitionType.S_S_ANNIHILATION, 1.0)],
        "cis": [
            (
                TransitionType.CIS_FRET_1,
                (
                    1 - acceptor_data.BISO_EFFICIENCY
                    if overwrite is None or "cis" not in overwrite
                    else 1 - overwrite["cis"][1]
                ),
            ),
            (
                TransitionType.CIS_FRET_2,
                (
                    acceptor_data.BISO_EFFICIENCY
                    if overwrite is None or "cis" not in overwrite
                    else overwrite["cis"][1]
                ),
            ),
        ],
        "off": [
            (
                TransitionType.OFF_FRET_1,
                (
                    1 - acceptor_data.OFRET_EFFICIENCY
                    if overwrite is None or "off" not in overwrite
                    else 1 - overwrite["off"][1]
                ),
            ),
            (
                TransitionType.OFF_FRET_2,
                (
                    acceptor_data.OFRET_EFFICIENCY
                    if overwrite is None or "off" not in overwrite
                    else overwrite["off"][1]
                ),
            ),
        ],
    }

    which_et_new: dict[str, list[tuple[TransitionType, float]]] = {
        acceptor_state: transitions.copy()
        for acceptor_state, transitions in which_et.items()
    }
    if include is not None:
        for acceptor_state in include:
            which_et_new[acceptor_state] = []
            total_factor = 0.0
            for transition_type, factor in include[acceptor_state]:
                total_factor += factor
                which_et_new[acceptor_state].append((transition_type, factor))
            for transition_type, factor in which_et[acceptor_state]:
                which_et_new[acceptor_state].append(
                    (transition_type, factor * (1 - total_factor))
                )

    transitions: list[Transition] = []
    for acceptor_state, acceptor_absorption in sorted(acceptor_absorptions.items()):
        rate = derive_energy_transfer_rate(
            donor_data=donor_data,
            acceptor_absorption=acceptor_absorption,
            distance=distance,
            dipole_orientation_factor=dipole_orientation_factor,
            refractive_index=refractive_index,
        )

        if acceptor_state not in which_et_new:
            raise ValueError(
                f"energy transfer to acceptor state {acceptor_state!r} "
                "is not supported."
            )

        if exclude is not None and acceptor_state in exclude:
            continue
        for transition_type, factor in which_et_new[acceptor_state]:
            if overwrite is not None and acceptor_state in overwrite:
                change_rate = overwrite[acceptor_state][0]
            else:
                change_rate = 1
            transition = Transition(
                rate=rate * factor * change_rate,
                transition_type=transition_type,
                fluorophore_ids=fluorophore_ids,
            )
            transitions.append(transition)

    return transitions


def derive_transitions(
    fluorophore_data: FluorophoreData,
    fluorophore_ids: list[int] | None = None,
    summarize: bool = False,
    irradiance: float = 2,
    wavelength: float = 640,
    bleaching: bool = False,
    dstorm: bool = True,
    **dstorm_parameters: Any,
) -> list[Transition]:
    """
    Derive non-energy transfer transitions based on the experimental conditions and the
    fluorophore to be mimicked.

    Parameters
    ----------
    fluorophore_data
        Contains all constant photophysical attributes of the fluorophore.
    fluorophore_ids
        All identities of a fluorophore within a FluorophoreSystem.
    summarize
        Whether to summarize some transitions into fewer.
    irradiance
        Irradiance in kW/cm².
    wavelength
        Wavelength in nm.
    bleaching
        Whether to incorporate bleaching as a possible transition.
    dstorm
        Whether to incorporate dstorm photoswitching as possible transitions.
    **dstorm_parameters
        Additional arguments for fluopy.photophysics.calculate_pet_rate, excluding
        k_pet.

    Returns
    -------
    list[Transition]
        Contains transitions of type Transition.
    """
    fd = fluorophore_data
    if fd.FLUORESCENCE_LIFETIME <= 0:
        raise ValueError(
            "FLUORESCENCE_LIFETIME must be greater than zero to derive transitions."
        )
    if fd.CROSS_SECTION_WAVELENGTH is not None:
        if wavelength != fd.CROSS_SECTION_WAVELENGTH:
            logger.warning(
                f"The excitation wavelength is set to {wavelength} nm, but the "
                f"cross sections of states other than S0 are defined at "
                f"{fd.CROSS_SECTION_WAVELENGTH} nm."
            )
    _, _, frequency = fo.convert_wavenumber_wavelength_frequency(wavelength=wavelength)
    photon_flux = fo.calculate_photon_flux(irradiance=irradiance, frequency=frequency)

    if "s0" not in fd.absorption_spectra:
        raise ValueError(
            "cannot derive excitation transition without an S0 absorption spectrum."
        )

    absorption_spectrum = fd.absorption_spectra["s0"]

    if fluorophore_ids is None:
        fluorophore_ids = [0]

    extinction_coefficient = absorption_spectrum.at(wavelength)

    excitation_rate = float(
        fo.calculate_excitation_rate(
            photon_flux=photon_flux, extinction_coefficient=extinction_coefficient
        )
    )
    excitation = Transition(
        rate=excitation_rate,
        transition_type=TransitionType.EXCITATION,
        fluorophore_ids=fluorophore_ids,
    )

    emission_rate = float(
        fo.calculate_emission_rate(
            quantum_yield=fd.QUANTUM_YIELD,
            fluorescence_lifetime=fd.FLUORESCENCE_LIFETIME,
        )
    )
    emission = Transition(
        rate=emission_rate,
        transition_type=TransitionType.FLUORESCENT_EMISSION,
        fluorophore_ids=fluorophore_ids,
    )

    isc_st = Transition(
        rate=fd.ISC_ST_RATE,
        transition_type=TransitionType.INTERSYSTEM_CROSSING_ST,
        fluorophore_ids=fluorophore_ids,
    )

    isc_ts = Transition(
        rate=fd.ISC_TS_RATE,
        transition_type=TransitionType.INTERSYSTEM_CROSSING_TS,
        fluorophore_ids=fluorophore_ids,
    )

    isomerization = Transition(
        rate=fd.ISO_RATE,
        transition_type=TransitionType.ISOMERIZATION,
        fluorophore_ids=fluorophore_ids,
    )

    biso_rate = float(
        fo.calculate_excitation_rate(
            photon_flux=photon_flux, absorption_cross_section=fd.BISO_CROSS_SECTION
        )
    )
    photo_bisomerization = Transition(
        rate=biso_rate,
        transition_type=TransitionType.PHOTO_BISO,
        fluorophore_ids=fluorophore_ids,
    )
    thermal_bisomerization = Transition(
        rate=fd.BISO_THERMAL_RATE,
        transition_type=TransitionType.THERM_BISO,
        fluorophore_ids=fluorophore_ids,
    )

    internal_conversion_rate = float(
        fo.calculate_internal_conversion_rate(
            quantum_yield=fd.QUANTUM_YIELD,
            emission_rate=emission_rate,
            iso_rate=fd.ISO_RATE,
            isc_st_rate=fd.ISC_ST_RATE,
        )
    )
    internal_conversion = Transition(
        rate=internal_conversion_rate,
        transition_type=TransitionType.INTERNAL_CONVERSION_S,
        fluorophore_ids=fluorophore_ids,
    )
    risc = Transition(
        rate=fd.RISC_RATE,
        transition_type=TransitionType.REVERSE_INTERSYSTEM_CROSSING,
        fluorophore_ids=fluorophore_ids,
    )

    dstorm_transitions = []
    if dstorm:
        dstorm_pet_t_rate = float(
            fo.calculate_pet_rate(k_pet=fd.DSTORM_PET_T_RATE_MOL, **dstorm_parameters)
        )
        dstorm_pet_s_rate = float(
            fo.calculate_pet_rate(k_pet=fd.DSTORM_PET_S_RATE_MOL, **dstorm_parameters)
        )
        dstorm_pet_t = Transition(
            rate=dstorm_pet_t_rate,
            transition_type=TransitionType.ET_CYCLE_T,
            fluorophore_ids=fluorophore_ids,
        )
        dstorm_pet_s = Transition(
            rate=dstorm_pet_s_rate,
            transition_type=TransitionType.ET_CYCLE_S,
            fluorophore_ids=fluorophore_ids,
        )
        dstorm_add_t_rate = dstorm_pet_t_rate * fd.DSTORM_PET_SUCCESS_RATE
        dstorm_add_s_rate = dstorm_pet_s_rate * fd.DSTORM_PET_SUCCESS_RATE
        dstorm_adduct_t = Transition(
            rate=dstorm_add_t_rate,
            transition_type=TransitionType.ADDUCT_T,
            fluorophore_ids=fluorophore_ids,
        )
        dstorm_adduct_s = Transition(
            rate=dstorm_add_s_rate,
            transition_type=TransitionType.ADDUCT_S,
            fluorophore_ids=fluorophore_ids,
        )
        photo_uncage = float(
            fo.calculate_excitation_rate(
                photon_flux=photon_flux,
                absorption_cross_section=fd.DSTORM_P_EL_CROSS_SECTION,
            )
        )
        photo_uncaging = Transition(
            rate=photo_uncage,
            transition_type=TransitionType.PHOTO_UNCAGING,
            fluorophore_ids=fluorophore_ids,
        )
        thermal_elimination = Transition(
            rate=fd.DSTORM_TH_EL_RATE_1,
            transition_type=TransitionType.THERM_ELIMINATION,
            fluorophore_ids=fluorophore_ids,
        )
        rad_escape = Transition(
            rate=dstorm_pet_t_rate * fd.RAD_ESCAPE_EFFICIENCY,
            transition_type=TransitionType.RAD_ESCAPE,
            fluorophore_ids=fluorophore_ids,
        )
        rad_relax = Transition(
            rate=fd.RAD_RELAX_RATE,
            transition_type=TransitionType.RAD_RELAX,
            fluorophore_ids=fluorophore_ids,
        )
        dstorm_transitions = [
            dstorm_pet_t,
            dstorm_pet_s,
            dstorm_adduct_t,
            dstorm_adduct_s,
            photo_uncaging,
            thermal_elimination,
            rad_escape,
            rad_relax,
        ]

    bleach = []
    if bleaching:
        bleach = [
            Transition(
                rate=fd.PHOTOBLEACH_T1_RATE,
                transition_type=TransitionType.PHOTOBLEACHING_1,
                fluorophore_ids=fluorophore_ids,
            )
        ]

    transitions = (
        [
            excitation,
            emission,
            isc_st,
            isc_ts,
            isomerization,
            photo_bisomerization,
            thermal_bisomerization,
            internal_conversion,
            risc,
        ]
        + dstorm_transitions
        + bleach
    )

    summarized_transitions = [
        TransitionType.S1_S0_TRANSITIONS,
        TransitionType.T1_S0_TRANSITIONS,
        TransitionType.CIS_S0_TRANSITIONS,
        TransitionType.OFF_S0_TRANSITIONS,
    ]

    transitions_copy = transitions[:]
    if summarize:
        for summarized_transition in summarized_transitions:
            rate = 0.0
            for transition in transitions_copy:
                if not transition.transition_type.photon:
                    if (
                        transition.transition_type.initial_state
                        == summarized_transition.initial_state
                        and transition.transition_type.final_state
                        == summarized_transition.final_state
                    ):
                        rate += transition.rate
                        transitions.remove(transition)
            sum_transition = Transition(
                rate=rate,
                transition_type=summarized_transition,
                fluorophore_ids=fluorophore_ids,
            )
            transitions.append(sum_transition)

    return transitions
