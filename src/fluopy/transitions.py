"""
Module transitions
"""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Collection
from dataclasses import asdict, dataclass, field
from enum import Enum
from itertools import product
from pathlib import Path
from typing import TYPE_CHECKING, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import interpolate as itp

from . import formulas as fo
from . import network as net

if TYPE_CHECKING:
    from fluopy.fluo_data import FluorophoreData
    from fluopy.fluorophores import Fluorophore, FluorophoreSystem


class SingleState(Enum):
    """
    Assigns a unique identifier (value) to each possible photophysical state.
    """

    S0 = 0
    S1 = 1
    S2 = 2
    T1 = 3
    T2 = 4
    B = 5
    cis = 6
    OFF = 7
    OFF2 = 8
    R = 9


class PairedState(Enum):
    """
    Assigns a combination of SingleState to each energy transfer related paired state.
    E.g., the classical Förster resonance energy transfer needs one fluorophore to be
    in S1 and another fluorophore closeby to be in S0. After the transition, the first
    fluorophore will be in S0 and the other in S1.
    """

    S1_S0 = [SingleState.S1, SingleState.S0]
    S0_S1 = [SingleState.S0, SingleState.S1]
    S1_T1 = [SingleState.S1, SingleState.T1]
    S1_Cis = [SingleState.S1, SingleState.cis]
    S0_Cis = [SingleState.S0, SingleState.cis]
    S1_OFF = [SingleState.S1, SingleState.OFF]
    S0_S0 = [SingleState.S0, SingleState.S0]
    S0_T2 = [SingleState.S0, SingleState.T2]
    S1_S1 = [SingleState.S1, SingleState.S1]
    S0_T1 = [SingleState.S0, SingleState.T1]
    S0_OFF2 = [SingleState.S0, SingleState.OFF2]
    S0_OFF = [SingleState.S0, SingleState.OFF]
    S0_B = [SingleState.S0, SingleState.B]

    @property
    def single_state_values(self) -> tuple[int, int]:
        """
        Returns a tuple of SingleState values.
        """
        return self.value[0].value, self.value[1].value

    @property
    def acceptor(self) -> SingleState:
        """
        Returns the acceptor (second value).
        """
        return self.value[1]

    @property
    def donor(self) -> SingleState:
        """
        Returns the donor (first value).
        """
        return self.value[0]


@dataclass
class TransitionAttributes:
    """
    Contains constant attributes of photophysical transitions.

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


class TransitionType(Enum):
    """
    Assigns constant attributes to each possible photophysical transition.
    """

    # general
    EXCITATION = TransitionAttributes("EXC", SingleState.S0, SingleState.S1, False)
    FLUORESCENT_EMISSION = TransitionAttributes(
        "FLU", SingleState.S1, SingleState.S0, True
    )
    SINGLET_QUENCHING = TransitionAttributes(
        "SQ", SingleState.S1, SingleState.S0, False
    )
    INTERSYSTEM_CROSSING_ST = TransitionAttributes(
        "ISC_ST", SingleState.S1, SingleState.T1, False
    )
    INTERSYSTEM_CROSSING_TS = TransitionAttributes(
        "ISC_TS", SingleState.T1, SingleState.S0, False
    )
    INTERNAL_CONVERSION_S = TransitionAttributes(
        "IC", SingleState.S1, SingleState.S0, False
    )
    REVERSE_INTERSYSTEM_CROSSING = TransitionAttributes(
        "RISC", SingleState.T2, SingleState.S1, False
    )
    PHOTOBLEACHING_1 = TransitionAttributes("BLE", SingleState.T1, SingleState.B, False)
    PHOTOBLEACHING_2 = TransitionAttributes(
        "BLE2", SingleState.T2, SingleState.B, False
    )

    # dstorm
    ET_CYCLE_T = TransitionAttributes("PET_TS", SingleState.T1, SingleState.S0, False)
    ET_CYCLE_S = TransitionAttributes("PET_SS", SingleState.S1, SingleState.S0, False)
    ADDUCT_T = TransitionAttributes("PET_TO", SingleState.T1, SingleState.OFF, False)
    ADDUCT_S = TransitionAttributes("PET_SO", SingleState.S1, SingleState.OFF, False)
    THERM_ELIMINATION = TransitionAttributes(
        "TE", SingleState.OFF, SingleState.S0, False
    )
    PHOTO_UNCAGING = TransitionAttributes("PU", SingleState.OFF, SingleState.S0, False)
    RAD_ESCAPE = TransitionAttributes("PET_TR", SingleState.T1, SingleState.R, False)
    RAD_RELAX = TransitionAttributes("OXI", SingleState.R, SingleState.S0, False)

    # cis trans isomerization
    ISOMERIZATION = TransitionAttributes("ISO", SingleState.S1, SingleState.cis, False)
    PHOTO_BISO = TransitionAttributes("PBISO", SingleState.cis, SingleState.S0, False)
    THERM_BISO = TransitionAttributes("TBISO", SingleState.cis, SingleState.S0, False)

    # energy transfers
    FRET = TransitionAttributes("FRET", PairedState.S1_S0, PairedState.S0_S1, False)
    CIS_FRET_1 = TransitionAttributes(
        "CET_1", PairedState.S1_Cis, PairedState.S0_Cis, False
    )
    CIS_FRET_2 = TransitionAttributes(
        "CET_2", PairedState.S1_Cis, PairedState.S0_S0, False
    )
    OFF_FRET_1 = TransitionAttributes(
        "OET_1", PairedState.S1_OFF, PairedState.S0_OFF, False
    )
    OFF_FRET_2 = TransitionAttributes(
        "OET_2", PairedState.S1_OFF, PairedState.S0_S0, False
    )
    S_S_ANNIHILATION = TransitionAttributes(
        "SSA", PairedState.S1_S1, PairedState.S0_S1, False
    )
    S_T_ANNIHILATION = TransitionAttributes(
        "STA", PairedState.S1_T1, PairedState.S0_T1, False
    )
    S_T_ANNI_RISC = TransitionAttributes(
        "STA_2", PairedState.S1_T1, PairedState.S0_S1, False
    )
    S_T_ANNI_BLEACH = TransitionAttributes(
        "STA_B", PairedState.S1_T1, PairedState.S0_B, False
    )

    # rhodamines
    H2O_ATTACK_S = TransitionAttributes("H2OS", SingleState.S1, SingleState.OFF, False)
    H2O_ATTACK_T = TransitionAttributes("H2OT", SingleState.T1, SingleState.OFF, False)
    BACK_REACTION = TransitionAttributes("BR", SingleState.OFF, SingleState.S0, False)

    # summarize
    S1_S0_TRANSITIONS = TransitionAttributes(
        "S1S0SUM", SingleState.S1, SingleState.S0, False
    )
    CIS_S0_TRANSITIONS = TransitionAttributes(
        "cisS0SUM", SingleState.cis, SingleState.S0, False
    )
    T1_S0_TRANSITIONS = TransitionAttributes(
        "T1S0SUM", SingleState.T1, SingleState.S0, False
    )
    OFF_S0_TRANSITIONS = TransitionAttributes(
        "OFFS0SUM", SingleState.OFF, SingleState.S0, False
    )

    @property
    def abbreviation(self) -> str:
        """
        Returns the abbreviation of type str.
        """
        return self.value.abbreviation

    @property
    def initial_state(self) -> SingleState | PairedState:
        """
        Returns the initial state of type SingleState or PairedState.
        """
        return self.value.initial_state

    @property
    def final_state(self) -> SingleState | PairedState:
        """
        Returns the final state of type SingleState or PairedState.
        """
        return self.value.final_state

    @property
    def photon(self) -> bool:
        """
        Returns bool indicating whether the transition emits a photon.
        """
        return self.value.photon


@dataclass(
    slots=True
)  # frozen=True if code will not be modified (autoreload complications otherwise)
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
        # __setattr__ needed if frozen=True
        object.__setattr__(self, "abbreviation", self.transition_type.abbreviation)
        object.__setattr__(self, "initial_state", self.transition_type.initial_state)
        object.__setattr__(self, "final_state", self.transition_type.final_state)
        object.__setattr__(self, "photon", self.transition_type.photon)
        object.__setattr__(self, "identity", None)
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


class TransitionSet:
    """
    Collection of all relevant transitions and related attributes. Allows optional
    post-init-modification and (subsequent) finalization.

    Attributes
    ----------
    transitions : dict[str, list[Transition]]
        Contains lists of transitions of type Transition with non-zero rate as values
        and fluorophores or fluorophore-combinations as keys.
    fluorophore_system : fluopy.fluorophores.FluorophoreSystem
        Container for attributes of multiple, interrelated fluorophores.
    combined_state_transitions_df : pd.DataFrame
        Contains realizable combined_state_transitions with their id as index and their
        other attributes as columns.
    row_sums : np.ndarray
        Contains the sum of each row of non-normalized transition rates, i.e., the sum
        of rates of all possbile combined_state_transitions.
    single_states : dict[str, list[SingleState]]
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
            or fluorophore-combinations as keys.
        fluorophore_system
            Container for attributes of multiple, interrelated fluorophores.
        keep_zero_rates
            Whether to keep transitions with rate 0.
        """
        self.transitions = transitions
        self.fluorophore_system = fluorophore_system

        self.transition_df = pd.DataFrame()
        i = 0
        for fluorophore_comb, f_transitions in transitions.items():
            keep_transitions = []
            df_constructor = []
            for transition in f_transitions:
                if "dist" not in fluorophore_comb and isinstance(
                    transition.initial_state, PairedState
                ):
                    raise ValueError(
                        "energy transfers have to be defined in transitions with the "
                        "key 'D: {name of donor}, A: {name of acceptor}, dist: "
                        "{distance between them}'."
                    )
                if "dist" in fluorophore_comb:
                    pattern = r"D:\s*([^,]+),\s*A:\s*([^,]+),\s*dist:\s*([\d.]+)"
                    match = re.match(pattern, fluorophore_comb)
                    d, a, dist = match.group(1), match.group(2), match.group(3)
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
                        if str(self.fluorophore_system.distances[(d_t, a_t)]) != dist:
                            raise ValueError(
                                f"{dist} nm indicated, "
                                f"{self.fluorophore_system.distances[(d_t, a_t)]} nm "
                                "found."
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
                        df_constructor.append(asdict(transition))
                else:
                    transition.identity = i
                    i += 1
                    keep_transitions.append(transition)
                    df_constructor.append(asdict(transition))
            if keep_transitions:
                transitions[fluorophore_comb] = keep_transitions
                transition_df = pd.DataFrame(df_constructor)
                transition_df = transition_df.set_index("identity")
                transition_df = pd.concat(
                    {fluorophore_comb: transition_df}, names=["Fluorophore"]
                )
                self.transition_df = pd.concat([self.transition_df, transition_df])

        self.single_states = get_single_states(self.transitions, self.transition_df)
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

    def filter_by_identity(self, remove_list: Collection = None) -> TransitionSet:
        """
        Returns another TransitionSet with transitions removed by their identity.

        Parameters
        ----------
        remove_list
            Contains identities of type int.

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
            transitions=filtered_transitions, fluorophore_system=self.fluorophore_system
        )

        return filtered

    def adjust_rates(
        self, change_dict: dict[str, float] = None, keep_zero_rates: bool = False
    ) -> TransitionSet:
        """
        Returns another TransitionSet with transition rates modified. Should be used
        as last modification step since other modifiers lack the keep_zero_rates
        argument.

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
                    transition.rate = change_dict[transition.identity]
        adjusted = TransitionSet(
            transitions=transitions,
            fluorophore_system=self.fluorophore_system,
            keep_zero_rates=keep_zero_rates,
        )

        return adjusted

    def remove_absorbing_states(self) -> TransitionSet:
        """
        Returns another TransitionSet that contains no Markovian absorbing states.

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
            transitions=keep_transitions, fluorophore_system=self.fluorophore_system
        )

        return no_abs

    def remove_energy_transfers(self) -> TransitionSet:
        """
        Return another TransitionSet that contains no transitions that are energy
        transfers.

        Returns
        -------
        no_ets : TransitionSet
            Re-initialization of the object with the modified transition collection.
        """
        transitions = copy.deepcopy(self.transitions)

        keep_transitions = {}
        for fluorophore, f_transition in transitions.items():
            if "dist" not in fluorophore:
                keep_transitions[fluorophore] = f_transition

        no_ets = TransitionSet(
            transitions=keep_transitions, fluorophore_system=self.fluorophore_system
        )

        return no_ets

    def finalize(self) -> Self:
        """
        Construct combined_state_transitions_df, transition_matrix and row_sums.

        Returns
        -------
        self
        """
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
    ) -> None:
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

        Returns
        -------
        None
        """
        graphs = net.construct_state_graphs(transition_df=self.transition_df)
        for graph in graphs:
            net.plot_graph(G=graph, graph_type=graph_type, colors=colors, scale=scale)


def get_single_states(
    transitions: Collection[Transition], transition_df: pd.DataFrame
) -> dict[str, npt.NDArray[SingleState]]:
    """
    Gets the values of SingleStates that occur in non-energy transfer transitions.
    Also assigns whether a transition leads to a Markovian absorbing state (note that
    hypothetically, an energy transfer onto that state which yields another or the same
    state could still happen).

    Parameters
    ----------
    transitions
        Contains transitions of type Transition with non-zero rate.
    transition_df
        Dataframe of all given transitions with non-zero rate containing their id as
        index and their other attributes as columns.

    Returns
    -------
    single_states : dict
        Contains the values of all relevant SingleStates as values. Name of
        fluorophores as keys.
    """
    transition_df["absorbing"] = False
    single_states = {}
    for fluorophore_comb, f_transitions in transitions.items():
        if "dist" not in fluorophore_comb:
            single_states_ = []
            for transition in f_transitions:
                initial_state = transition.initial_state
                final_state = transition.final_state
                if initial_state.value not in single_states_:
                    single_states_.append(initial_state.value)
                if final_state.value not in single_states_:
                    single_states_.append(final_state.value)
            single_states_ = np.array(single_states_)
            single_states[fluorophore_comb] = single_states_
            single_state_df = pd.DataFrame(single_states_, columns=["single_states"])
            single_state_df["absorbing"] = False
            for i, single_state in single_state_df["single_states"].items():
                if (
                    single_state
                    not in transition_df.loc[fluorophore_comb, "initial_state"]
                    .apply(lambda x: x.value)
                    .values
                ):
                    single_state_df.at[i, "absorbing"] = True
            final_states = (
                transition_df.loc[fluorophore_comb, "final_state"]
                .apply(lambda x: x.value)
                .values
            )
            absorbing_states = single_state_df["single_states"][
                single_state_df["absorbing"]
            ]
            if not absorbing_states.empty:
                indices = np.where(np.isin(final_states, absorbing_states.values))[0]
                index_values = transition_df.loc[fluorophore_comb].iloc[indices].index
                transition_df.loc[(fluorophore_comb, index_values), "absorbing"] = True

    return single_states


def get_state_combinations(
    single_states: dict[str, Collection[SingleState | int]],
    fluorophores: Collection[Fluorophore],
) -> list[tuple[int, ...]]:
    """
    Combines all given states with each other according to the amount and order of the
    respective fluorophore. Cartesian product, see itertools.product().

    Parameters
    ----------
    states
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
    state_combinations: Collection[tuple[int, int]],
) -> list[tuple[tuple[int, int], ...]]:
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
    transition_rate_matrix = np.zeros(shape=(transition_count, transition_count))

    for i, row in combined_state_transitions_df.iterrows():
        final_state = row["final_state"]
        indices = combined_state_transitions_df[
            combined_state_transitions_df["initial_state"] == final_state
        ].index
        transition_rate_matrix[i][indices] = combined_state_transitions_df["rate"][
            indices
        ]

    row_sums = transition_rate_matrix.sum(axis=1)
    row_sums_exp = np.tile(np.expand_dims(row_sums, axis=1), row_sums.size)
    mask = np.ma.make_mask(row_sums_exp)
    transition_matrix = np.divide(
        transition_rate_matrix,
        row_sums_exp,
        out=np.zeros_like(transition_rate_matrix),
        where=mask,
    )

    return transition_matrix, row_sums


def derive_energy_transfer_transitions(
    donor_data: FluorophoreData,
    acceptor_data: FluorophoreData,
    fluorophore_ids: list[int],
    dipole_orientation_factor: float,
    distance: float,
    refractive_index: float,
    overwrite: dict[str, list[float]] | None = None,
    exclude: list[str] | None = None,
    include: dict[str, list[float]] | None = None,
) -> list[Transition]:
    """
    Derive energy transfer transitions based on the experimental conditions and the
    fluorophore-combinations to be mimicked. The type of energy transfer is determined
    via the data file names.

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
    data_dir = os.path.join(Path(__file__).parent, "fluorophore_spectra")
    donor_emission = pd.read_csv(
        os.path.join(data_dir, donor_data.data_files, "emission.csv")
    )
    acceptor_files = os.listdir(os.path.join(data_dir, acceptor_data.data_files))
    acceptor_abs_files = [
        data_file for data_file in acceptor_files if data_file.startswith("absorption")
    ]

    emission_rate = fo.calculate_emission_rate(
        quantum_yield=donor_data.QUANTUM_YIELD,
        fluorescence_lifetime=donor_data.FLUORESCENCE_LIFETIME,
    )
    minimum, maximum = 200, 1000
    wavelengths_of_interest = np.arange(minimum, maximum + 1, 1, dtype=float)

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
    for acceptor_abs_file in acceptor_abs_files:
        acceptor_abs = pd.read_csv(
            os.path.join(data_dir, acceptor_data.data_files, acceptor_abs_file)
        )

        J = fo.calculate_spectral_overlap_integral(
            donor=donor_emission["y"],
            acceptor=acceptor_abs["y"],
            wavelengths=wavelengths_of_interest,
        )
        rate = fo.calculate_fret_rate(
            distance=distance,
            emission_rate=emission_rate,
            spectral_overlap_integral=J,
            dipole_orientation_factor=dipole_orientation_factor,
            refractive_index=refractive_index,
        )
        acceptor_state = acceptor_abs_file.split("_")[1].split(".")[0]
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
        Whether to incooperate bleaching as a possible transition.
    dstorm
        Whether to incooperate dstorm photoswitching as possible transitions.
    dstorm_parameters : fo.calculate_pet_rate arguments (except k_pet)

    Returns
    -------
    transitions : list[Transition]
        Contains transitions of type Transition.
    """
    fd = fluorophore_data
    _, _, frequency = fo.convert_wavenumber_wavelength_frequency(wavelength=wavelength)
    photon_flux = fo.calculate_photon_flux(irradiance=irradiance, frequency=frequency)
    path_absorption = os.path.join(
        Path(__file__).parent,
        "fluorophore_spectra",
        fd.data_files,
        "absorption_s0.csv",
    )
    if fluorophore_ids is None:
        fluorophore_ids = [0]

    dataframe_absorption = pd.read_csv(filepath_or_buffer=path_absorption, index_col=0)

    extinction_coefficient = dataframe_absorption.loc[int(wavelength), "y"]

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
        Contains emission or absorption data with columns 'Wavelength' and 'y'.

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
