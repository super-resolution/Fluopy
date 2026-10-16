"""
Compute a prediction for a photophysical system.
"""

from __future__ import annotations

import logging
import warnings
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from scipy import sparse
from scipy.sparse.linalg import MatrixRankWarning, spsolve
from scipy.stats import expon

from . import plotting
from ._statistics import (
    normalize_transition_frequencies,
    parse_paired_transition_label,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes

    from .transitions import TransitionSet

__all__: list[str] = ["Prediction"]

logger = logging.getLogger(__name__)


class Prediction:
    """
    Container of mathematically derived statistical attributes and methods.

    Attributes
    ----------
    paired_transitions
        Whether the prediction includes paired transitions.
    absorbing_chain
        Whether the subchain reachable from start_at contains a terminal
        combined state and the prediction was carried out on an absorbing Markov chain.
        Absorbing states have a lifetime of inf and a frequency / occupation of 0.
        Absorbing transitions have a frequency of 0.
    transition_set
        Collection of all relevant transitions and related attributes.
    start_at
        Initial combined state as a tuple of SingleState values, ordered by
        fluorophore identity. Defaults to S0 for every fluorophore.
    frequency_transitions
        Relative number of expected transition occurrences, normalized separately for
        each fluorophore. Paired-transition occurrences are assigned to the donor's
        transition group.
    frequency_states
        Relative expected number of visits to each state, normalized separately for each
        fluorophore.
    transition_time_distributions
        Expected distributions of time until transition.
        Contains objects of type scipy.stats.*.rv_frozen for each transition, or np.inf
        when its source state has no positive-rate outgoing transitions.
        None if paired_transitions is True.
    lifetime_distributions
        Name of fluorophores as keys and their state's expected lifetime distributions
        (objects of type scipy.stats.*.rv_frozen) (array) as values. States without
        positive-rate state-changing transitions have np.inf as their distribution.
        None if paired_transitions is True.
    mean_transition_times
        Expected means of time until transition.
        None if paired_transitions is True.
    mean_lifetimes
        Name of fluorophores as keys and their state's expected lifetime means (array)
        as values.
        None if paired_transitions is True.
    state_occupations
        Relative time spent in each state, normalized separately for each fluorophore.
        None if paired_transitions is True.

    Notes
    -----
    Transition statistics are indexed by transition identities from the second index
    level of transition_set.transition_df. State statistics are grouped by fluorophore
    name, with entries following the order of transition_set.single_states[name].

    Predictions are available for systems containing at most two fluorophores.

    Predicted lifetimes and state occupations are not available for systems containing
    paired transitions.

    For non-absorbing systems, transition frequencies are calculated from the
    stationary distribution of the subchain reachable from start_at. The
    reachable transition matrix must have a unique stationary distribution. The
    initial state therefore selects the relevant component of a disconnected system,
    but states with the same reachable subchain produce the same stationary
    distribution.

    A reachable subchain is treated as an absorbing Markov chain if it contains a
    terminal combined state and every reachable transition row can reach a terminal
    row. If a terminal row is reachable but absorption is not certain, prediction is
    rejected.

    The Markov-chain nodes are rows of
    transition_set.combined_state_transitions_df rather than combined states. An
    incoming row is used to represent start_at internally. It does not count as an
    event at time zero. The initial state's residence time is included in occupations.
    """

    def __init__(
        self,
        transition_set: TransitionSet,
        start_at: tuple[int, ...] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        transition_set
            Collection of all relevant transitions and related attributes.
        start_at
            If None, tuple of as many zeros as number of fluorophores.
            Can be any combination (size of number of fluorophores) of possible
            SingleState values. See transition_set.single_states. The selected state
            determines the reachable subchain, as in Simulation.run. Only events
            occurring after the start are counted in transition frequencies.
        """
        self.paired_transitions: bool = False
        self.absorbing_chain: bool = False
        if transition_set.fluorophore_system.count > 2:
            raise ValueError("prediction not available for more than 2 fluorophores.")
        self.transition_set = transition_set
        start_index = self._get_start_index(start_at)
        self.start_at = tuple(
            int(state) for state in self._combined_states[start_index]
        )
        self._terminal_state_combinations = self._get_terminal_state_combinations()
        transition_matrix = self._transition_matrix
        if sparse.issparse(transition_matrix):
            self._reachable_transition_indices = _get_reachable_indices_sparse(
                transition_matrix, start_index
            )
        else:
            self._reachable_transition_indices = _get_reachable_indices(
                transition_matrix, start_index
            )
        terminal_indices = np.flatnonzero(self._row_sums == 0)
        self._reachable_terminal_indices = np.intersect1d(
            self._reachable_transition_indices,
            terminal_indices,
            assume_unique=True,
        )
        if self._reachable_terminal_indices.size:
            if sparse.issparse(transition_matrix):
                can_reach_terminal = _get_reachable_indices_sparse(
                    transition_matrix.T, self._reachable_terminal_indices
                )
            else:
                can_reach_terminal = _get_reachable_indices(
                    transition_matrix.T, self._reachable_terminal_indices
                )
            if not np.all(
                np.isin(self._reachable_transition_indices, can_reach_terminal)
            ):
                raise ValueError(
                    "the reachable subchain contains a closed nonterminal class; "
                    "absorption is not certain."
                )
            self.absorbing_chain = True
        if any(
            transition_type.mechanism is not None
            for transition_type in transition_set.transition_df["transition_type"]
        ):
            logger.warning(
                "Only frequencies are available for systems with paired transitions; "
                "lifetimes and occupations are not available.",
                stacklevel=2,
            )
            self.paired_transitions = True
        if self.absorbing_chain:
            logger.warning(
                "absorbing states have a lifetime of inf and a frequency / occupation "
                "of 0. Absorbing transitions have a frequency of 0.",
                stacklevel=2,
            )
            if len(self._terminal_state_combinations) > 1:
                logger.warning(
                    "multiple terminal state combinations are available; predicted "
                    "transition frequencies depend on start_at.",
                    stacklevel=2,
                )

        self.frequency_transitions: npt.NDArray[np.float64]
        self.frequency_states: dict[str, npt.NDArray[np.float64]]
        self.transition_time_distributions: npt.NDArray[Any] | None
        self.lifetime_distributions: dict[str, npt.NDArray[Any]] | None
        self.mean_transition_times: npt.NDArray[np.float64] | None
        self.mean_lifetimes: dict[str, npt.NDArray[np.float64]] | None
        self.state_occupations: dict[str, npt.NDArray[np.float64]] | None
        self._stationary_distribution: npt.NDArray[np.float64] | None = None
        self._transient_visits: npt.NDArray[np.float64] | None = None
        if self.absorbing_chain:
            if sparse.issparse(transition_matrix):
                self.frequency_transitions = (
                    self.predict_transition_occurrences_absorbing_sparse(
                        start_at=self.start_at
                    )
                )
            else:
                self.frequency_transitions = (
                    self.predict_transition_occurrences_absorbing(
                        start_at=self.start_at
                    )
                )
        else:
            if sparse.issparse(transition_matrix):
                self.frequency_transitions = (
                    self.predict_transition_occurrences_sparse()
                )
            else:
                self.frequency_transitions = self.predict_transition_occurrences()
        self.frequency_states = self.predict_state_occurrences()
        if not self.paired_transitions:
            (
                self.transition_time_distributions,
                self.lifetime_distributions,
            ) = self.predict_lifetimes()
            self.mean_transition_times = np.array(
                [
                    distr.mean() if distr != np.inf else np.inf
                    for distr in self.transition_time_distributions
                ]
            )
            self.mean_lifetimes, self.state_occupations = self.infer_stats()
        else:
            (
                self.transition_time_distributions,
                self.lifetime_distributions,
                self.mean_transition_times,
                self.mean_lifetimes,
                self.state_occupations,
            ) = (None, None, None, None, None)

    def _get_start_index(self, start_at: tuple[int, ...] | None) -> int:
        """Resolve the initial state using the same row convention as Simulation.run."""
        start_at, start_index, self._transition_matrix, self._row_sums = (
            self.transition_set._get_start_configuration(start_at)
        )
        df = self.transition_set.combined_state_transitions_df
        final_states = df["final_state"].tolist()
        if start_index == len(df):
            final_states.append(start_at)
        self._combined_states = np.asarray(final_states, dtype=np.int64)
        return start_index

    def _get_terminal_state_combinations(self) -> list[tuple[int, ...]]:
        """
        Return terminal state combinations in deterministic order.

        Returns
        -------
        list[tuple[int, ...]]
            Terminal state combinations sorted by their state values.
        """
        return sorted(self.transition_set.terminal_state_combinations)

    def predict_transition_occurrences(self) -> npt.NDArray[np.float64]:
        """
        Predict transition frequencies in the reachable non-absorbing subchain.

        The calculation is restricted to transition rows reachable from
        start_at. Each different type of fluorophore's transition
        frequencies sum to 1.

        Each paired transition event is counted as one transition occurrence, including
        events that change both the donor and acceptor states. For normalization, a
        paired transition occurrence is assigned only to the donor's transition group;
        ordinary transitions are assigned to their respective fluorophore groups.

        Returns
        -------
        npt.NDArray[np.float64]
            Expected relative frequencies of each transition. Frequencies remain 0 for
            a fluorophore with no expected transitions.

        Notes
        -----
        The stationary distribution is calculated by solving pi P = pi together with
        the constraint that the entries of pi sum to 1. The reachable transition
        matrix must have a unique stationary distribution. start_at selects
        the relevant component of a disconnected transition matrix. Starting states with
        the same reachable subchain produce the same stationary distribution. An
        all-zero transition matrix returns zero frequencies.
        """
        full_transition_matrix = self._transition_matrix
        frequency_transitions = np.zeros(self.transition_set.transition_df.shape[0])
        self._transition_occurrences = frequency_transitions
        self._stationary_distribution = np.zeros(full_transition_matrix.shape[0])
        if not np.any(full_transition_matrix):
            return frequency_transitions

        reachable_indices = getattr(
            self,
            "_reachable_transition_indices",
            np.arange(full_transition_matrix.shape[0]),
        )
        transition_matrix = full_transition_matrix[
            np.ix_(reachable_indices, reachable_indices)
        ]
        stationary_system = transition_matrix.T - np.identity(
            transition_matrix.shape[0]
        )
        stationary_system[-1] = 1
        normalization = np.zeros(transition_matrix.shape[0])
        normalization[-1] = 1
        try:
            reachable_stationary_distribution = np.linalg.solve(
                stationary_system, normalization
            )
        except np.linalg.LinAlgError as exception:
            raise ValueError(
                "the reachable non-absorbing subchain does not have a unique "
                "stationary distribution."
            ) from exception
        stationary_distribution_combined_state_transitions = (
            self._stationary_distribution
        )
        stationary_distribution_combined_state_transitions[reachable_indices] = (
            reachable_stationary_distribution
        )

        df = self.transition_set.combined_state_transitions_df
        for _, i in self.transition_set.transition_df.index:
            indices = df.index[df["transition_id"] == i].tolist()
            frequency_transitions[i] = (
                stationary_distribution_combined_state_transitions[indices].sum()
            )

        return normalize_transition_frequencies(
            frequency_transitions, self.transition_set.transition_df
        )

    def predict_transition_occurrences_sparse(self) -> npt.NDArray[np.float64]:
        """
        Predict transition frequencies in the reachable non-absorbing subchain using
        a sparse transition matrix.

        The calculation is restricted to transition rows reachable from
        start_at. Each different type of fluorophore's transition
        frequencies sum to 1.

        Each paired transition event is counted as one transition occurrence, including
        events that change both the donor and acceptor states. For normalization, a
        paired transition occurrence is assigned only to the donor's transition group;
        ordinary transitions are assigned to their respective fluorophore groups.

        Returns
        -------
        npt.NDArray[np.float64]
            Expected relative frequencies of each transition. Frequencies remain 0 for
            a fluorophore with no expected transitions.

        Notes
        -----
        The stationary distribution is calculated by solving pi P = pi together with
        the constraint that the entries of pi sum to 1. The reachable transition
        matrix must have a unique stationary distribution. start_at selects
        the relevant component of a disconnected transition matrix. Starting states with
        the same reachable subchain produce the same stationary distribution. An
        all-zero transition matrix returns zero frequencies.
        """
        full_transition_matrix = sparse.csr_array(
            self._transition_matrix, dtype=np.float64
        )
        frequency_transitions = np.zeros(self.transition_set.transition_df.shape[0])
        self._transition_occurrences = frequency_transitions
        self._stationary_distribution = np.zeros(full_transition_matrix.shape[0])
        if full_transition_matrix.nnz == 0:
            return frequency_transitions

        reachable_indices = getattr(
            self,
            "_reachable_transition_indices",
            np.arange(full_transition_matrix.shape[0]),
        )
        transition_matrix = full_transition_matrix[reachable_indices][
            :, reachable_indices
        ]
        size = transition_matrix.shape[0]
        stationary_system = (
            transition_matrix.T - sparse.eye(size, format="csr")
        ).tolil()
        stationary_system[-1, :] = np.ones(size)
        normalization = np.zeros(size)
        normalization[-1] = 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                reachable_stationary_distribution = spsolve(
                    stationary_system.tocsr(), normalization
                )
        except (MatrixRankWarning, RuntimeError) as exception:
            raise ValueError(
                "the reachable non-absorbing subchain does not have a unique "
                "stationary distribution."
            ) from exception
        if not np.all(np.isfinite(reachable_stationary_distribution)):
            raise ValueError(
                "the reachable non-absorbing subchain does not have a unique "
                "stationary distribution."
            )
        stationary_distribution_combined_state_transitions = (
            self._stationary_distribution
        )
        stationary_distribution_combined_state_transitions[reachable_indices] = (
            reachable_stationary_distribution
        )

        df = self.transition_set.combined_state_transitions_df
        for _, i in self.transition_set.transition_df.index:
            indices = df.index[df["transition_id"] == i].tolist()
            frequency_transitions[i] = (
                stationary_distribution_combined_state_transitions[indices].sum()
            )

        return normalize_transition_frequencies(
            frequency_transitions, self.transition_set.transition_df
        )

    def predict_transition_occurrences_absorbing(
        self, start_at: tuple[int, ...] | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Predict transition frequencies before absorption from a selected state.

        The calculation includes only transition rows reachable from
        start_at. A reachable row without positive-rate outgoing
        transitions is terminal. Every reachable row must be able to reach a terminal
        row, so that absorption is certain. Absorbing transitions have the value 0.

        Each paired transition event is counted as one transition occurrence, including
        events that change both the donor and acceptor states. For normalization, a
        paired transition occurrence is assigned only to the donor's transition group;
        ordinary transitions are assigned to their respective fluorophore groups.

        Parameters
        ----------
        start_at
            Initial combined state as a tuple of SingleState values, ordered by
            fluorophore identity. If None, all fluorophores start in S0. The incoming
            row representing this state is excluded from transition occurrences at
            time zero, while its residence time is included in state occupations.

        Returns
        -------
        npt.NDArray[np.float64]
            Expected relative frequencies of each transition. Frequencies remain 0 for
            a fluorophore with no expected transitions.
        """
        start_index = self._get_start_index(start_at)
        transition_abs = self.transition_set.transition_df["absorbing"]
        abs_indices = transition_abs[transition_abs].index.get_level_values(1)
        df = self.transition_set.combined_state_transitions_df
        abs_indices_combined = df[df["transition_id"].isin(abs_indices)].index
        transition_matrix = self._transition_matrix
        reachable_indices = _get_reachable_indices(transition_matrix, start_index)
        terminal_indices = np.flatnonzero(self._row_sums == 0)
        drop_transitions = np.intersect1d(
            reachable_indices, terminal_indices, assume_unique=True
        )
        frequency_transitions = np.zeros(transition_abs.size)
        self._transition_occurrences = frequency_transitions
        self._transient_visits = np.zeros(transition_matrix.shape[0])
        if start_index in drop_transitions:
            return frequency_transitions
        if not drop_transitions.size:
            raise ValueError("no terminal state is reachable from start_at.")
        can_reach_terminal = _get_reachable_indices(
            transition_matrix.T, drop_transitions
        )
        if not np.all(np.isin(reachable_indices, can_reach_terminal)):
            raise ValueError(
                "the reachable subchain contains a closed nonterminal class; "
                "absorption is not certain."
            )
        drop_diff = abs_indices_combined[
            ~np.isin(abs_indices_combined, drop_transitions)
        ]
        transient_indices = reachable_indices[
            ~np.isin(reachable_indices, drop_transitions)
        ]
        Q = transition_matrix[np.ix_(transient_indices, transient_indices)]
        initial_transient_index = int(
            np.flatnonzero(transient_indices == start_index)[0]
        )
        initial_visit = np.zeros(transient_indices.size)
        initial_visit[initial_transient_index] = 1
        try:
            expected_transient_visits = np.linalg.solve(
                (np.eye(Q.shape[0]) - Q).T, initial_visit
            )
        except np.linalg.LinAlgError as exception:
            raise ValueError("the transient subchain is singular.") from exception
        if not np.all(np.isfinite(expected_transient_visits)):
            raise ValueError("the transient subchain is singular.")
        self._transient_visits[transient_indices] = expected_transient_visits
        expected_visits = self._transient_visits.copy()
        expected_visits[start_index] -= 1
        expected_visits[drop_diff] = 0
        for _, i in self.transition_set.transition_df.index:
            indices = df.index[df["transition_id"] == i].tolist()
            frequency_transitions[i] = expected_visits[indices].sum()

        return normalize_transition_frequencies(
            frequency_transitions, self.transition_set.transition_df
        )

    def predict_transition_occurrences_absorbing_sparse(
        self, start_at: tuple[int, ...] | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Predict transition frequencies before absorption from a selected state using
        a sparse transition matrix.

        The calculation includes only transition rows reachable from
        start_at. A reachable row without positive-rate outgoing
        transitions is terminal. Every reachable row must be able to reach a terminal
        row, so that absorption is certain. Absorbing transitions have the value 0.

        Each paired transition event is counted as one transition occurrence, including
        events that change both the donor and acceptor states. For normalization, a
        paired transition occurrence is assigned only to the donor's transition group;
        ordinary transitions are assigned to their respective fluorophore groups.

        Parameters
        ----------
        start_at
            Initial combined state as a tuple of SingleState values, ordered by
            fluorophore identity. If None, all fluorophores start in S0. The incoming
            row representing this state is excluded from transition occurrences at
            time zero, while its residence time is included in state occupations.

        Returns
        -------
        npt.NDArray[np.float64]
            Expected relative frequencies of each transition. Frequencies remain 0 for
            a fluorophore with no expected transitions.
        """
        start_index = self._get_start_index(start_at)
        transition_abs = self.transition_set.transition_df["absorbing"]
        abs_indices = transition_abs[transition_abs].index.get_level_values(1)
        df = self.transition_set.combined_state_transitions_df
        abs_indices_combined = df[df["transition_id"].isin(abs_indices)].index
        transition_matrix = sparse.csr_array(self._transition_matrix, dtype=np.float64)
        reachable_indices = _get_reachable_indices_sparse(
            transition_matrix, start_index
        )
        terminal_indices = np.flatnonzero(self._row_sums == 0)
        drop_transitions = np.intersect1d(
            reachable_indices, terminal_indices, assume_unique=True
        )
        frequency_transitions = np.zeros(transition_abs.size)
        self._transition_occurrences = frequency_transitions
        self._transient_visits = np.zeros(transition_matrix.shape[0])
        if start_index in drop_transitions:
            return frequency_transitions
        if not drop_transitions.size:
            raise ValueError("no terminal state is reachable from start_at.")
        can_reach_terminal = _get_reachable_indices_sparse(
            transition_matrix.T, drop_transitions
        )
        if not np.all(np.isin(reachable_indices, can_reach_terminal)):
            raise ValueError(
                "the reachable subchain contains a closed nonterminal class; "
                "absorption is not certain."
            )
        drop_diff = abs_indices_combined[
            ~np.isin(abs_indices_combined, drop_transitions)
        ]
        transient_indices = reachable_indices[
            ~np.isin(reachable_indices, drop_transitions)
        ]
        Q = transition_matrix[transient_indices][:, transient_indices]
        initial_transient_index = int(
            np.flatnonzero(transient_indices == start_index)[0]
        )
        initial_visit = np.zeros(transient_indices.size)
        initial_visit[initial_transient_index] = 1
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", MatrixRankWarning)
                expected_transient_visits = spsolve(
                    (sparse.eye(Q.shape[0], format="csr") - Q).T,
                    initial_visit,
                )
        except (MatrixRankWarning, RuntimeError) as exception:
            raise ValueError("the transient subchain is singular.") from exception
        if not np.all(np.isfinite(expected_transient_visits)):
            raise ValueError("the transient subchain is singular.")
        self._transient_visits[transient_indices] = expected_transient_visits
        expected_visits = self._transient_visits.copy()
        expected_visits[start_index] -= 1
        expected_visits[drop_diff] = 0
        for _, i in self.transition_set.transition_df.index:
            indices = df.index[df["transition_id"] == i].tolist()
            frequency_transitions[i] = expected_visits[indices].sum()

        return normalize_transition_frequencies(
            frequency_transitions, self.transition_set.transition_df
        )

    def predict_state_occurrences(self) -> dict[str, npt.NDArray[np.float64]]:
        """
        Predict the relative frequencies of states. Each different type of fluorophore's
        states frequencies sum up to 1.

        State visits are counted separately for each physical fluorophore. A
        paired transition event therefore contributes a visit for both donor and
        acceptor if both states change, while still representing one transition
        occurrence.

        Returns
        -------
        dict[str, npt.NDArray[np.float64]]
            Name of fluorophores as keys and their state's expected relative
            frequencies (array) as values. Frequencies remain 0 if no state visits are
            expected.
        """
        single_states = self.transition_set.single_states
        frequency_states = {
            key: np.zeros(len(value)) for key, value in single_states.items()
        }
        self_frequency_states = {
            key: np.zeros(len(value)) for key, value in single_states.items()
        }
        transition_occurrences = self._transition_occurrences
        grouped = self.transition_set.transition_df.groupby(level=0)
        for fluorophore_comb_raw, f_transitions in grouped:
            fluorophore_comb = cast(str, fluorophore_comb_raw)
            paired_transition = parse_paired_transition_label(fluorophore_comb)
            if paired_transition is not None:
                d, a, _ = paired_transition
                single_states_a = single_states[a]
                single_states_d = single_states[d]
                for row_index, transition in f_transitions.iterrows():
                    identity = int(cast(tuple[Any, int], row_index)[1])
                    donor_i, acceptor_i = transition[
                        "initial_state"
                    ].single_state_values
                    donor_f, acceptor_f = transition["final_state"].single_state_values
                    index_1 = np.where(single_states_d == donor_f)[0][0]
                    donor_frequencies = (
                        frequency_states
                        if donor_i != donor_f
                        else self_frequency_states
                    )
                    donor_frequencies[d][index_1] += transition_occurrences[identity]
                    if acceptor_i != acceptor_f:
                        index_2 = np.where(single_states_a == acceptor_f)[0][0]
                        frequency_states[a][index_2] += transition_occurrences[identity]
                    else:
                        index_2 = np.where(single_states_a == acceptor_f)[0][0]
                        self_frequency_states[a][index_2] += transition_occurrences[
                            identity
                        ]

            else:
                single_states_f = single_states[fluorophore_comb]
                for row_index, transition in f_transitions.iterrows():
                    identity = int(cast(tuple[Any, int], row_index)[1])
                    index = np.where(
                        single_states_f == transition["final_state"].value
                    )[0][0]
                    target = (
                        frequency_states
                        if transition["initial_state"] != transition["final_state"]
                        else self_frequency_states
                    )
                    target[fluorophore_comb][index] += transition_occurrences[identity]
        for fluorophore, state_frequencies in frequency_states.items():
            total = state_frequencies.sum()
            if total == 0:
                stationary_distribution = self._stationary_distribution
                if (
                    stationary_distribution is not None
                    and stationary_distribution.sum() > 0
                ):
                    fluorophore_ids = (
                        self.transition_set.fluorophore_system.get_identities(
                            fluorophore
                        )
                    )
                    combined_states = self._combined_states[:, fluorophore_ids]
                    state_counts = np.count_nonzero(
                        combined_states[:, :, None] == single_states[fluorophore],
                        axis=1,
                    )
                    state_frequencies = stationary_distribution @ state_counts
                else:
                    state_frequencies = self_frequency_states[fluorophore]
                total = state_frequencies.sum()
            if total > 0:
                frequency_states[fluorophore] = state_frequencies / total

        return frequency_states

    def predict_lifetimes(
        self,
    ) -> tuple[npt.NDArray[Any], dict[str, npt.NDArray[Any]]]:
        """
        Predict the lifetime distributions of states and the time until occurrence
        distributions of transitions.

        Returns
        -------
        transition_time_distributions : npt.NDArray[object]
            Expected distributions of time until transition.
            Contains objects of type scipy.stats.*.rv_frozen for each transition, or
            np.inf when its source state has no positive-rate outgoing transitions.
        lifetime_distributions : dict[str, npt.NDArray[object]]
            Name of fluorophores as keys and their state's expected lifetime
            distributions (objects of type scipy.stats.*.rv_frozen) (array) as values.
            States without positive-rate state-changing transitions have np.inf as
            their distribution.
        """
        lifetime_distributions = {
            key: np.empty(len(value), dtype=object)
            for key, value in self.transition_set.single_states.items()
        }
        transition_time_distributions = np.empty(
            self.transition_set.transition_df.shape[0], dtype=object
        )

        for fluorophore, states in self.transition_set.single_states.items():
            for i, state in enumerate(states):
                event_rate = 0.0
                exit_rate = 0.0
                associated_transitions: list[int] = []
                for j, transition in self.transition_set.transition_df.loc[
                    fluorophore
                ].iterrows():
                    source = transition.initial_state.value
                    if source == state:
                        event_rate += transition.rate
                        if transition.initial_state != transition.final_state:
                            exit_rate += transition.rate
                        associated_transitions.append(cast(int, j))
                if event_rate == 0:
                    transition_pdf: Any = np.inf
                else:
                    transition_pdf = expon(scale=1 / event_rate)
                if exit_rate == 0:
                    lifetime_pdf: Any = np.inf
                else:
                    lifetime_pdf = expon(scale=1 / exit_rate)
                lifetime_distributions[fluorophore][i] = lifetime_pdf
                transition_time_distributions[associated_transitions] = transition_pdf

        return transition_time_distributions, lifetime_distributions

    def infer_stats(
        self,
    ) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]:
        """
        Calculate mean lifetimes and relative state occupations.

        Mean lifetimes are calculated from lifetime_distributions. State occupations
        use stationary or expected transient visits to transition rows, weighted by
        their mean holding times. In absorbing chains, only residence time before
        reaching a terminal combined state is included.

        Returns
        -------
        mean_lifetimes : dict[str, npt.NDArray[np.float64]]
            Name of fluorophores as keys and their state's expected lifetime means
            (array) as values.
        state_occupations : dict[str, npt.NDArray[np.float64]]
            Fluorophore names as keys and arrays of expected relative residence times
            as values, normalized separately for each fluorophore name.
        """
        lifetime_distributions = self.lifetime_distributions
        if lifetime_distributions is None:
            raise ValueError(
                "lifetime statistics are unavailable for paired transitions."
            )
        mean_lifetimes: dict[str, npt.NDArray[np.float64]] = {}
        for fluorophore, distributions in lifetime_distributions.items():
            mean_lifetimes[fluorophore] = np.array(
                [distr.mean() if distr != np.inf else np.inf for distr in distributions]
            )
        visits = self._stationary_distribution
        if visits is None:
            visits = cast(npt.NDArray[np.float64], self._transient_visits)
        row_sums = self._row_sums
        time_weights = np.divide(
            visits,
            row_sums,
            out=np.zeros_like(row_sums),
            where=row_sums > 0,
        )
        combined_states = self._combined_states
        state_occupations = {}
        for fluorophore, states in self.transition_set.single_states.items():
            identities = self.transition_set.fluorophore_system.get_identities(
                fluorophore
            )
            state_counts = np.count_nonzero(
                combined_states[:, identities, None] == states, axis=1
            )
            occupations = time_weights @ state_counts
            total = occupations.sum()
            if total > 0:
                occupations /= total
            state_occupations[fluorophore] = occupations

        return mean_lifetimes, state_occupations

    def plot_frequency_transitions(self, **kwargs: Any) -> mplAxes:
        """
        Plot frequencies of transitions.

        Parameters
        ----------
        kwargs
            kwargs for fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        return plotting._plot_transition_bars(
            transition_df=self.transition_set.transition_df,
            values=self.frequency_transitions,
            default_ylabel="Prob. occurrence",
            **kwargs,
        )

    def plot_frequency_states(self, **kwargs: Any) -> mplAxes:
        """
        Plot frequencies of states.

        Parameters
        ----------
        kwargs
            kwargs for fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """

        values = plotting._flatten_state_values(
            self.transition_set, self.frequency_states
        )
        return plotting._plot_state_bars(
            transition_set=self.transition_set,
            values=values,
            default_ylabel="Prob. occurrence",
            **kwargs,
        )

    def plot_mean_transition_times(self, **kwargs: Any) -> mplAxes:
        """
        Plot mean times until transitions occur.

        Parameters
        ----------
        kwargs
            kwargs for fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        if self.paired_transitions:
            raise ValueError(
                "mean_transition_times not available if paired transitions possible."
            )
        mean_transition_times = self.mean_transition_times
        if mean_transition_times is None:
            raise ValueError("mean transition times are unavailable.")
        return plotting._plot_transition_bars(
            transition_df=self.transition_set.transition_df,
            values=mean_transition_times,
            default_ylabel=r"$\tau$ (s)",
            **kwargs,
        )

    def plot_mean_lifetimes(self, **kwargs: Any) -> mplAxes:
        """
        Plot mean lifetimes of states.

        Parameters
        ----------
        kwargs
            kwargs for fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        if self.paired_transitions:
            raise ValueError(
                "mean_lifetimes not available if paired transitions possible."
            )
        mean_lifetimes = self.mean_lifetimes
        if mean_lifetimes is None:
            raise ValueError("mean lifetimes are unavailable.")

        values = plotting._flatten_state_values(self.transition_set, mean_lifetimes)
        return plotting._plot_state_bars(
            transition_set=self.transition_set,
            values=values,
            default_ylabel=r"$\tau$ (s)",
            full_xlim=True,
            **kwargs,
        )

    def plot_state_occupations(self, **kwargs: Any) -> mplAxes:
        """
        Plot state occupation times (relative total time spent in state).

        Parameters
        ----------
        kwargs
            kwargs for fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        if self.paired_transitions:
            raise ValueError(
                "state_occupations not available if paired transitions possible."
            )
        state_occupations = self.state_occupations
        if state_occupations is None:
            raise ValueError("state occupations are unavailable.")

        values = plotting._flatten_state_values(self.transition_set, state_occupations)
        return plotting._plot_state_bars(
            transition_set=self.transition_set,
            values=values,
            default_ylabel="Prob. occupation",
            **kwargs,
        )

    def plot_lifetime_distributions(
        self,
        fluorophore: str,
        state_identity: int,
        x: npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> mplAxes:
        """
        Plot lifetime distributions of states.

        Parameters
        ----------
        fluorophore
            The name of the fluorophore whose state's distribution is to be shown.
        state_identity
            The identity of the state whose distribution is to be shown.
        x
            The x values for which the distribution is to be shown.
        kwargs
            kwargs for fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        if self.paired_transitions:
            raise ValueError(
                "lifetime_distributions not available if paired transitions possible."
            )
        lifetime_distributions = self.lifetime_distributions
        mean_lifetimes = self.mean_lifetimes
        if lifetime_distributions is None or mean_lifetimes is None:
            raise ValueError("lifetime distributions are unavailable.")

        kwargs.setdefault("type_", "line")
        kwargs.setdefault("ylabel", "PD")
        kwargs.setdefault(
            "title",
            rf"$\tau$ of {fluorophore} {self.transition_set.states_by_value[state_identity].name}",
        )
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("xlabel", "lifetime [s]")
        index = np.where(
            self.transition_set.single_states[fluorophore] == state_identity
        )[0][0]
        distribution = lifetime_distributions[fluorophore][index]
        if isinstance(distribution, float):
            raise ValueError(f"The lifetimes are all equal to {distribution}")

        if x is None:
            x = np.linspace(0, mean_lifetimes[fluorophore][index] * 10, 1000)
        data = [x, distribution.pdf(x)]
        ax = plotting.plot_data(data=data, **kwargs)

        return ax

    def plot_transition_time_distributions(
        self,
        fluorophore: str,
        transition_id: int,
        x: npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> mplAxes:
        """
        Plot distributions of time until transition occurs.

        Parameters
        ----------
        fluorophore
            The name of the fluorophore whose transition's distribution is to be shown.
        transition_id
            The identity of the transition whose distribution is to be shown.
        x
            The x values for which the distribution is to be shown.
        kwargs
            kwargs for fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        if self.paired_transitions:
            raise ValueError(
                "transition_time_distributions not available if paired transitions "
                "possible."
            )
        transition_distributions = self.transition_time_distributions
        mean_transition_times = self.mean_transition_times
        if transition_distributions is None or mean_transition_times is None:
            raise ValueError("transition-time distributions are unavailable.")
        kwargs.setdefault("type_", "line")
        kwargs.setdefault("ylabel", "PD")
        kwargs.setdefault(
            "title",
            rf"""$\tau$ of {fluorophore}
            {self.transition_set.transition_df.loc[(fluorophore, transition_id),
                                                   "abbreviation"]}""",
        )
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("xlabel", "time to transition [s]")
        if x is None:
            x = np.linspace(0, mean_transition_times[transition_id] * 10, 1000)
        data = [x, transition_distributions[transition_id].pdf(x)]

        ax = plotting.plot_data(data=data, **kwargs)

        return ax


def _get_reachable_indices_sparse(
    transition_matrix: sparse.spmatrix | sparse.sparray,
    start_indices: int | npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    """
    Return rows reachable through positive-probability matrix entries.

    A positive entry at row i and column j defines a directed edge from i to j. The
    search includes every supplied starting row and repeatedly follows these edges
    until no unvisited rows remain. Passing a transposed transition matrix therefore
    finds rows that can reach the supplied starting rows in the original matrix.

    Parameters
    ----------
    transition_matrix
        Sparse square matrix whose positive entries define directed edges between rows.
    start_indices
        Row or rows from which to start the search.

    Returns
    -------
    npt.NDArray[np.int64]
        Sorted reachable row indices, including start_indices.
    """
    matrix = sparse.csr_array(transition_matrix, dtype=np.float64)
    matrix.eliminate_zeros()
    reachable = np.zeros(matrix.shape[0], dtype=bool)
    frontier = np.atleast_1d(start_indices).astype(np.int64)
    reachable[frontier] = True

    while frontier.size:
        neighbor_parts = [
            matrix.indices[matrix.indptr[row] : matrix.indptr[row + 1]]
            for row in frontier
        ]
        next_indices = np.unique(np.concatenate(neighbor_parts))
        frontier = next_indices[~reachable[next_indices]]
        reachable[frontier] = True

    return np.flatnonzero(reachable)


def _get_reachable_indices(
    transition_matrix: npt.ArrayLike,
    start_indices: int | npt.ArrayLike,
) -> npt.NDArray[np.int64]:
    """
    Return rows reachable through positive-probability matrix entries.

    A positive entry at row i and column j defines a directed edge from i to j. The
    search includes every supplied starting row and repeatedly follows these edges
    until no unvisited rows remain. Passing a transposed transition matrix therefore
    finds rows that can reach the supplied starting rows in the original matrix.

    Parameters
    ----------
    transition_matrix
        Square matrix whose positive entries define directed edges between rows.
    start_indices
        Row or rows from which to start the search.

    Returns
    -------
    npt.NDArray[np.int64]
        Sorted reachable row indices, including start_indices.
    """
    matrix = np.asarray(transition_matrix, dtype=np.float64)
    reachable = np.zeros(matrix.shape[0], dtype=bool)
    frontier = np.atleast_1d(start_indices).astype(np.int64)
    reachable[frontier] = True

    while frontier.size:
        next_indices = np.flatnonzero(np.any(matrix[frontier] > 0, axis=0))
        frontier = next_indices[~reachable[next_indices]]
        reachable[frontier] = True

    return np.flatnonzero(reachable)
