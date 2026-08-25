"""
Define and handle photophysical transitions.
"""

from __future__ import annotations

import copy
import logging
import re
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field, fields
from itertools import product
from typing import TYPE_CHECKING, ClassVar, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import interpolate as itp

from . import formulas as fo
from . import network as net
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


@dataclass(frozen=True, slots=True)
class SingleState:
    """
    Contains the name and numerical identifier of a photophysical state.

    Attributes
    ----------
    name : str
        Name of the state.
    value : int
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
    Contains the donor and acceptor states of a paired transition.

    Built-in paired states are available as class attributes and in
    BUILTIN_PAIRED_STATES. Additional paired states can be constructed directly.

    Attributes
    ----------
    name : str
        Name of the paired state.
    donor : SingleState
        State of the donor.
    acceptor : SingleState
        State of the acceptor.
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
    abbreviation : str
        Abbreviation of the transition.
    initial_state : SingleState, PairedState
        Initial state of the transition.
    final_state : SingleState, PairedState
        Final state of the transition.
    photon : bool
        Whether the transition emits a photon.
    """

    abbreviation: str
    initial_state: SingleState | PairedState
    final_state: SingleState | PairedState
    photon: bool


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

# energy transfers
TransitionType.FRET = TransitionType(
    "FRET", PairedState.S1_S0, PairedState.S0_S1, False
)
TransitionType.CIS_FRET_1 = TransitionType(
    "CET_1", PairedState.S1_Cis, PairedState.S0_Cis, False
)
TransitionType.CIS_FRET_2 = TransitionType(
    "CET_2", PairedState.S1_Cis, PairedState.S0_S0, False
)
TransitionType.OFF_FRET_1 = TransitionType(
    "OET_1", PairedState.S1_OFF, PairedState.S0_OFF, False
)
TransitionType.OFF_FRET_2 = TransitionType(
    "OET_2", PairedState.S1_OFF, PairedState.S0_S0, False
)
TransitionType.S_S_ANNIHILATION = TransitionType(
    "SSA", PairedState.S1_S1, PairedState.S0_S1, False
)
TransitionType.S_T_ANNIHILATION = TransitionType(
    "STA", PairedState.S1_T1, PairedState.S0_T1, False
)
TransitionType.S_T_ANNI_RISC = TransitionType(
    "STA_2", PairedState.S1_T1, PairedState.S0_S1, False
)
TransitionType.S_T_ANNI_BLEACH = TransitionType(
    "STA_B", PairedState.S1_T1, PairedState.S0_B, False
)
TransitionType.R_FRET_1 = TransitionType(
    "RET_1", PairedState.S1_R, PairedState.S0_R, False
)
TransitionType.R_FRET_2 = TransitionType(
    "RET_2", PairedState.S1_R, PairedState.S0_S0, False
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
    identity : int
        The id of the transition. Not None if transition is part of a TransitionSet.
    transition_type : TransitionType
        The photophysical type of the transitions with its constant attributes.
    abbreviation : str
        The abbreviation of the transition.
    initial_state : SingleState | PairedState
        The initial state of the transition.
    final_state : SingleState | PairedState
        The final state of the transition.
    rate : float
        The rate of the transition.
    photon : bool
        Whether the transition emits a photon.
    fluorophore_ids : list[int] | list[tuple[int, int]]
        Contains the identities of relevant fluorophores.
        If energy transfer, tuples of fluorophore pairs, where the first is the donor
        and the second is the acceptor.
    """

    identity: int = field(init=False)
    transition_type: TransitionType = field()
    abbreviation: str = field(init=False)
    initial_state: SingleState | PairedState = field(init=False)
    final_state: SingleState | PairedState = field(init=False)
    rate: float = field()
    photon: bool = field(init=False)
    fluorophore_ids: list[int] | list[tuple[int, int]] = field()

    def __post_init__(self) -> None:
        if not np.isscalar(self.rate) or not np.isfinite(self.rate) or self.rate < 0:
            raise ValueError("rate must be a finite, non-negative scalar.")

        self.rate = float(self.rate)
        self.abbreviation = self.transition_type.abbreviation
        self.initial_state = self.transition_type.initial_state
        self.final_state = self.transition_type.final_state
        self.photon = self.transition_type.photon
        self.identity = None
        for fluorophore_id in self.fluorophore_ids:
            if isinstance(self.initial_state, PairedState):
                if not isinstance(fluorophore_id, tuple) or len(fluorophore_id) != 2:
                    raise ValueError(
                        f"{self.abbreviation} is energy transfer, "
                        "fluorophore_ids have to be tuples of fluorophore "
                        "pairs."
                    )
            else:
                if not isinstance(fluorophore_id, int):
                    raise ValueError(
                        f"{self.abbreviation} is not an energy transfer, "
                        "fluorophore_ids has to be a list of ints."
                    )

    def to_dict(self) -> dict[str, object]:
        """
        Return the transition fields as a shallow dictionary.
        """
        return {item.name: getattr(self, item.name) for item in fields(self)}


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
    states_by_value : dict[int, SingleState]
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
        Contains lists of transitions of type Transition with non-zero rate as values
        and fluorophores or fluorophore-combinations as keys. Fluorophore-combination
        keys require the format 'D: {name of donor}, A: {name of acceptor}, dist:
        {distance between them in nm}'.
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
    transition_df : pd.DataFrame
        Dataframe of all given transitions with non-zero rate containing their id as
        second level index and their other attributes as columns. Name of fluorophores
        as first level index.
    transition_matrix : np.ndarray
        Contains the normalized rate constants (i.e., point probabilities) for each
        possible combined_state_transition at the corresponding index pair.
    """

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
            Whether to keep transitions with rate 0.
        """
        transitions = {
            key: [copy.copy(transition) for transition in transition_collection]
            for key, transition_collection in transitions.items()
        }
        self.transitions = transitions
        self.fluorophore_system = fluorophore_system

        self.transition_df = pd.DataFrame()
        i = 0
        for fluorophore_comb, f_transitions in transitions.items():
            keep_transitions = []
            df_constructor = []
            for transition in f_transitions:
                if isinstance(transition.initial_state, PairedState):
                    pattern = (
                        r"D:\s*([^,]+),\s*A:\s*([^,]+),\s*dist:\s*(\d+(?:\.\d+)?)\s*"
                    )
                    match = re.fullmatch(pattern=pattern, string=fluorophore_comb)
                    if match is None:
                        raise ValueError(
                            "energy transfers have to be defined in transitions with "
                            "the key 'D: {name of donor}, A: {name of acceptor}, dist: "
                            "{distance between them in nm}'."
                        )
                    d, a, dist = match.groups()
                    for d_t, a_t in transition.fluorophore_ids:
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
                    for j in transition.fluorophore_ids:
                        if (
                            self.fluorophore_system.fluorophores[j].name
                            != fluorophore_comb
                        ):
                            raise ValueError(
                                f"{fluorophore_comb} indicated to be at identity {j}, "
                                f"{self.fluorophore_system.fluorophores[j].name} found."
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
                transitions[fluorophore_comb] = keep_transitions
                transition_df = pd.DataFrame(df_constructor)
                transition_df = transition_df.set_index("identity")
                transition_df = pd.concat(
                    {fluorophore_comb: transition_df}, names=["Fluorophore"]
                )
                self.transition_df = pd.concat([self.transition_df, transition_df])
        self.states_by_value = get_states_by_value(self.transitions)
        self.single_states = get_single_states(
            self.transitions,
            self.transition_df,
            self.fluorophore_system,
        )
        # also assigns whether a transition leads to a Markovian absorbing state

        self._combined_state_transitions_df = None
        self._row_sums = None
        self._transition_matrix = None

    @property
    def combined_state_transitions_df(self) -> pd.DataFrame:
        if self._combined_state_transitions_df is None:
            self.finalize()
        return self._combined_state_transitions_df

    @property
    def row_sums(self) -> npt.NDArray[np.float64]:
        if self._row_sums is None:
            self.finalize()
        return self._row_sums

    @property
    def transition_matrix(self) -> npt.NDArray[np.float64]:
        if self._transition_matrix is None:
            self.finalize()
        return self._transition_matrix

    def filter_by_identity(
        self, remove_list: Collection = None, keep_zero_rates: bool = False
    ) -> TransitionSet:
        """
        Returns another TransitionSet with transitions removed by their identity.

        Parameters
        ----------
        remove_list
            Contains identities of type int.
        keep_zero_rates
            Whether to keep transitions with rate 0.

        Returns
        -------
        filtered : TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        transitions = copy.deepcopy(self.transitions)
        if remove_list is None:
            remove_list = []
        filtered_transitions = {}
        for fluorophore, f_transitions in transitions.items():
            for transition in f_transitions:
                if transition.identity not in remove_list:
                    if fluorophore in filtered_transitions:
                        filtered_transitions[fluorophore] += [transition]
                    else:
                        filtered_transitions[fluorophore] = [transition]
        filtered = TransitionSet(
            transitions=filtered_transitions,
            fluorophore_system=self.fluorophore_system,
            keep_zero_rates=keep_zero_rates,
        )

        return filtered

    def adjust_rates(
        self, change_dict: dict[int, float] = None, keep_zero_rates: bool = False
    ) -> TransitionSet:
        """
        Returns another TransitionSet with transition rates modified.

        Parameters
        ----------
        change_dict
            Contains identities of transitions as key and rates as values.
        keep_zero_rates
            Whether to keep transitions with rate 0.

        Returns
        -------
        adjusted : TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        transitions = copy.deepcopy(self.transitions)

        if change_dict is None:
            change_dict = {}
        for _, f_transitions in transitions.items():
            for transition in f_transitions:
                if transition.identity in change_dict:
                    rate = change_dict[transition.identity]
                    if not np.isscalar(rate) or not np.isfinite(rate) or rate < 0:
                        raise ValueError("rate must be a finite, non-negative scalar.")
                    transition.rate = float(rate)
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
            Whether to keep transitions with rate 0.

        Returns
        -------
        no_abs : TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        transitions = copy.deepcopy(self.transitions)  # transitions are objects
        # if no deep copy, the transition objects of the new TransitionSet are the
        # very same objects as in the old one
        keep_transitions = {}
        for fluorophore, f_transitions in transitions.items():
            for transition in f_transitions:
                if not self.transition_df.loc[
                    (fluorophore, transition.identity), "absorbing"
                ]:
                    if fluorophore in keep_transitions:
                        keep_transitions[fluorophore] += [transition]
                    else:
                        keep_transitions[fluorophore] = [transition]

        no_abs = TransitionSet(
            transitions=keep_transitions,
            fluorophore_system=self.fluorophore_system,
            keep_zero_rates=keep_zero_rates,
        )

        return no_abs

    def remove_energy_transfers(self, keep_zero_rates: bool = False) -> TransitionSet:
        """
        Return another TransitionSet that contains no transitions that are energy
        transfers.

        Parameters
        ----------
        keep_zero_rates
            Whether to keep transitions with rate 0.

        Returns
        -------
        no_ets : TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        transitions = copy.deepcopy(self.transitions)

        keep_transitions = {}
        for fluorophore, f_transition in transitions.items():
            if not isinstance(f_transition[0].initial_state, PairedState):
                keep_transitions[fluorophore] = f_transition

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
        self
        """
        if self._combined_state_transitions_df is not None:
            return self

        state_combinations = get_state_combinations(
            single_states=self.single_states,
            fluorophores=self.fluorophore_system.fluorophores,
        )
        combined_state_transitions = get_combined_state_transitions(
            state_combinations=state_combinations
        )
        combined_state_transitions_with_rates = construct_transition_rate_list(
            transition_df=self.transition_df,
            combined_state_transitions=combined_state_transitions,
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
            ],
        )
        self._combined_state_transitions_df.index.name = "id"

        self._transition_matrix, self._row_sums = construct_transition_matrix(
            combined_state_transitions_df=self._combined_state_transitions_df
        )

        return self

    def plot(
        self,
        graph_type: str = "shell",
        colors: Collection | None = None,
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
            axes = [None] * len(graphs)

        return_axes = []
        try:
            for graph, ax in zip(graphs, axes, strict=True):
                ax = net.plot_graph(
                    G=graph, graph_type=graph_type, colors=colors, scale=scale, ax=ax
                )
                return_axes.append(ax)
        except ValueError as exception:
            raise ValueError(
                f"The number of axes elements must be {len(graphs)} or None"
            ) from exception
        return return_axes


def get_single_states(
    transitions: dict[str, Collection[Transition]],
    transition_df: pd.DataFrame,
    fluorophore_system: FluorophoreSystem,
) -> dict[str, npt.NDArray[int]]:
    """
    Get the values of SingleStates occurring in transitions.

    PairedState components are assigned to their corresponding donor and acceptor
    fluorophore types. Transitions leading to individually Markovian absorbing states
    are identified using only non-energy-transfer transitions.

    Parameters
    ----------
    transitions
        Contains transitions of type Transition with non-zero rate.
    transition_df
        Dataframe of all given transitions with non-zero rate containing their id as
        index and their other attributes as columns.
    fluorophore_system
        Container for attributes of multiple, interrelated fluorophores.

    Returns
    -------
    single_states : dict
        Contains the values of all relevant SingleStates as values. Name of
        fluorophores as keys.
    """
    transition_df["absorbing"] = False
    single_states = {}
    for fluorophore_comb, f_transitions in transitions.items():
        if not isinstance(f_transitions[0].initial_state, PairedState):
            single_states_ = []
            for transition in f_transitions:
                initial_state = transition.initial_state
                final_state = transition.final_state
                if initial_state.value not in single_states_:
                    single_states_.append(initial_state.value)
                if final_state.value not in single_states_:
                    single_states_.append(final_state.value)

            single_states[fluorophore_comb] = single_states_
            single_state_df = pd.DataFrame(single_states_, columns=["single_states"])
            single_state_df["absorbing"] = False

            initial_states = (
                transition_df.loc[fluorophore_comb, "initial_state"]
                .apply(lambda x: x.value)
                .values
            )
            for i, single_state in single_state_df["single_states"].items():
                if single_state not in initial_states:
                    single_state_df.at[i, "absorbing"] = True

            final_states = (
                transition_df.loc[fluorophore_comb, "final_state"]
                .apply(lambda x: x.value)
                .values
            )
            absorbing_states = single_state_df.loc[
                single_state_df["absorbing"],
                "single_states",
            ]
            if not absorbing_states.empty:
                indices = np.where(np.isin(final_states, absorbing_states.values))[0]
                index_values = transition_df.loc[fluorophore_comb].iloc[indices].index
                transition_df.loc[(fluorophore_comb, index_values), "absorbing"] = True
    for f_transitions in transitions.values():
        for transition in f_transitions:
            if not isinstance(transition.initial_state, PairedState):
                continue

            initial_state = transition.initial_state
            final_state = transition.final_state

            for donor_id, acceptor_id in transition.fluorophore_ids:
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
    single_states = {
        fluorophore: np.asarray(states, dtype=np.int64)
        for fluorophore, states in single_states.items()
    }

    return single_states


def get_state_combinations(
    single_states: dict[str, Collection[int]],
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


def get_combined_state_transitions(
    state_combinations: Collection[tuple[int, ...]],
) -> list[tuple[tuple[int, ...], ...]]:
    """
    Combines all given state_combinations with themselves 2 times. Cartesian product,
    see itertools.product(). Each combination resembles a combined_state_transition.

    Parameters
    ----------
    state_combinations
        state_combinations to be combined.

    Returns
    -------
    list
        Contains combinations of state_combinations of type tuple.
    """
    return list(product(state_combinations, repeat=2))


def rate_assignment_standard(
    transition: pd.Series,
    transition_id: int,
    transition_rate_list: list[float],
    combined_state_transitions: Collection[tuple[int, ...]],
) -> list[float]:
    """
    Adds a realizable combined_state_transition that is no energy transfer as a list to
    the transition_rate_list. Here, a combined_state_transition is realizable, if its
    first state_combination (i.e., current_state) and its second state_combination
    (i.e., future_state) have and only have a change in the state of one fluorophore
    that can also be found in the photophysical transition.

    Parameters
    ----------
    transition
        The transition to be assigned to combined_state_transitions.
    transition_id
        The identity of Transition.
    transition_rate_list
        Destination of realizable combined_state_transitions.
    combined_state_transitions
        Contains combinations of state_combinations of type tuple.

    Returns
    -------
    transition_rate_list : list
        The altered input parameter.
    """
    source = transition["initial_state"].value
    destination = transition["final_state"].value

    for current_state, future_state in combined_state_transitions:
        for index in transition["fluorophore_ids"]:
            if source == current_state[index]:
                if destination == future_state[index]:
                    future_state_part = future_state[:index] + future_state[index + 1 :]
                    current_state_part = (
                        current_state[:index] + current_state[index + 1 :]
                    )
                    if not future_state_part == current_state_part:
                        break
                    else:
                        transition_rate_list.append(
                            [
                                current_state,
                                future_state,
                                [index],
                                transition["abbreviation"],
                                transition_id,
                                transition["rate"],
                                transition["photon"],
                            ]
                        )

    return transition_rate_list


def rate_assignment_energy_transfer(
    transition: pd.Series,
    transition_id: int,
    transition_rate_list: list[float],
    combined_state_transitions: Collection[tuple[int, ...]],
) -> list[float]:
    """
    Adds a realizable combined_state_transition that is also an energy transfer as a
    list to the transition_rate_list. Here, a combined_state_transition is realizable,
    if its first state_combination (i.e., current_state) and its second
    state_combination (i.e., future_state) have and only have a change in the state of
    two fluorophores that can also be found in the photophysical transition of the
    corresponding distance.

    Parameters
    ----------
    transition
        The transition to be assigned to combined_state_transitions.
    transition_id
        The identity of Transition.
    transition_rate_list
        Destination of realizable combined_state_transitions.
    combined_state_transitions
        Contains combinations of state_combinations of type tuple.

    Returns
    -------
    transition_rate_list : list
        The altered input parameter.
    """
    source_donor, source_acceptor = transition["initial_state"].single_state_values
    destination_donor, destination_acceptor = transition[
        "final_state"
    ].single_state_values

    for current_state, future_state in combined_state_transitions:
        for donor, acceptor in transition["fluorophore_ids"]:
            if (
                source_donor == current_state[donor]
                and source_acceptor == current_state[acceptor]
                and destination_donor == future_state[donor]
                and destination_acceptor == future_state[acceptor]
            ):
                i, j = min(donor, acceptor), max(donor, acceptor)
                future_state_part = (
                    future_state[:i] + future_state[i + 1 : j] + future_state[j + 1 :]
                )
                current_state_part = (
                    current_state[:i]
                    + current_state[i + 1 : j]
                    + current_state[j + 1 :]
                )
                if not future_state_part == current_state_part:
                    break
                else:
                    transition_rate_list.append(
                        [
                            current_state,
                            future_state,
                            [donor, acceptor],
                            transition["abbreviation"],
                            transition_id,
                            transition["rate"],
                            transition["photon"],
                        ]
                    )

    return transition_rate_list


def construct_transition_rate_list(
    transition_df: pd.DataFrame, combined_state_transitions: Collection[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    """
    Constructs a list that contains lists of each realizable combined_state_transition.
    The inner lists contain initial state_combination, final state_combination,
    the index of involved fluorophores, abbreviation, transition id, rate and whether a
    photon is emitted.

    Parameters
    ----------
    transition_df
        Dataframe of all given transitions with non-zero rate containing their id as
        second level index and their other attributes as columns. Name of fluorophores
        as first level index.
    combined_state_transitions
        Contains combinations of state_combinations of type tuple.

    Returns
    -------
    transition_rate_list : list
        Contains lists of each realizable combined_state_transition.
    """
    transition_rate_list = list()
    for (_, identity), transition in transition_df.iterrows():
        if isinstance(transition["initial_state"], SingleState):
            transition_rate_list = rate_assignment_standard(
                transition=transition,
                transition_id=identity,
                transition_rate_list=transition_rate_list,
                combined_state_transitions=combined_state_transitions,
            )
        else:
            transition_rate_list = rate_assignment_energy_transfer(
                transition=transition,
                transition_id=identity,
                transition_rate_list=transition_rate_list,
                combined_state_transitions=combined_state_transitions,
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

    for i, row in combined_state_transitions_df.iterrows():
        final_state = row["final_state"]
        indices = combined_state_transitions_df[
            combined_state_transitions_df["initial_state"] == final_state
        ].index
        transition_rate_matrix[i][indices] = combined_state_transitions_df["rate"][
            indices
        ]

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
    rate : float
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

    emission_rate = fo.calculate_emission_rate(
        quantum_yield=donor_data.QUANTUM_YIELD,
        fluorescence_lifetime=donor_data.FLUORESCENCE_LIFETIME,
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
    overwrite: dict[str, Collection[float]] | None = None,
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
    transitions : list[Transition]
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

    which_et = {
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
        "s1": [(TransitionType.S_S_ANNIHILATION, 1)],
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

    which_et_new = which_et.copy()
    if include is not None:
        for acceptor_state in include:
            which_et_new[acceptor_state] = []
            total_factor = 0
            for transition_type, factor in include[acceptor_state]:
                total_factor += factor
                which_et_new[acceptor_state].append((transition_type, factor))
            for transition_type, factor in which_et[acceptor_state]:
                which_et_new[acceptor_state].append(
                    (transition_type, factor * (1 - total_factor))
                )

    transitions = []
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
    summarize: bool = False,
    fluorophore_data: FluorophoreData | None = None,
    fluorophore_ids: list[int] | None = None,
    irradiance: float = 2,
    wavelength: float = 640,
    bleaching: bool = False,
    dstorm: bool = True,
    **dstorm_parameters,
) -> list[Transition]:
    """
    Derive non-energy transfer transitions based on the experimental conditions and the
    fluorophore to be mimicked.

    Parameters
    ----------
    summarize
        Whether to summarize some transitions into fewer.
    fluorophore_data
        Contains all constant photophysical attributes of the fluorophore.
    fluorophore_ids
        All identities of a fluorophore within a FluorophoreSystem.
    irradiance
        Irradiance in kW/cm².
    wavelength
        Wavelength in nm.
    bleaching
        Whether to incorporate bleaching as a possible transition.
    dstorm
        Whether to incorporate dstorm photoswitching as possible transitions.
    dstorm_parameters : fo.calculate_pet_rate arguments (except k_pet)

    Returns
    -------
    transitions : list[Transition]
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

    excitation_rate = fo.calculate_excitation_rate(
        photon_flux=photon_flux, extinction_coefficient=extinction_coefficient
    )
    excitation = Transition(
        rate=excitation_rate,
        transition_type=TransitionType.EXCITATION,
        fluorophore_ids=fluorophore_ids,
    )

    emission_rate = fo.calculate_emission_rate(
        quantum_yield=fd.QUANTUM_YIELD, fluorescence_lifetime=fd.FLUORESCENCE_LIFETIME
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

    biso_rate = fo.calculate_excitation_rate(
        photon_flux=photon_flux, absorption_cross_section=fd.BISO_CROSS_SECTION
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

    internal_conversion_rate = fo.calculate_internal_conversion_rate(
        quantum_yield=fd.QUANTUM_YIELD,
        emission_rate=emission_rate,
        iso_rate=fd.ISO_RATE,
        isc_st_rate=fd.ISC_ST_RATE,
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
        dstorm_pet_t_rate = fo.calculate_pet_rate(
            k_pet=fd.DSTORM_PET_T_RATE_MOL, **dstorm_parameters
        )
        dstorm_pet_s_rate = fo.calculate_pet_rate(
            k_pet=fd.DSTORM_PET_S_RATE_MOL, **dstorm_parameters
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
        photo_uncage = fo.calculate_excitation_rate(
            photon_flux=photon_flux,
            absorption_cross_section=fd.DSTORM_P_EL_CROSS_SECTION,
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
            rate = 0
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


def interpolate_data(
    minimum_wavelength: int, maximum_wavelength: int, data: pd.DataFrame
) -> npt.NDArray[np.float64]:
    """
    Interpolate missing data points from data.

    Parameters
    ----------
    minimum_wavelength
        The minimum wavelength the interpolated data should cover.
    maximum_wavelength
        The maximum wavelength the interpolated data should cover.
    data
        Contains emission or absorption data with columns 'Wavelengths' and 'y'.

    Returns
    -------
    interpolated : 1-D array_like
        The data corresponding to each integer wavelength within the defined margins.
    """
    data["Wavelengths"] = data["Wavelengths"].astype(int)
    data = data.drop_duplicates(subset=["Wavelengths"])
    wavelengths = data["Wavelengths"]
    add_lower = np.zeros(shape=min(wavelengths) - minimum_wavelength)
    add_upper = np.zeros(shape=maximum_wavelength - max(wavelengths))
    wavelengths_stepwise = np.arange(min(wavelengths), max(wavelengths) + 1)
    interpolated = itp.CubicSpline(wavelengths, data["y"])(wavelengths_stepwise)
    interpolated = np.where(interpolated < 0, 0, interpolated)
    interpolated = np.concatenate((add_lower, interpolated, add_upper))

    return interpolated
