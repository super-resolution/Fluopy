"""
Compute a prediction for a photophysical system.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
from scipy.stats import expon

from . import plotting
from ._statistics import (
    calculate_state_occupations,
    normalize_transition_frequencies,
    parse_energy_transfer_label,
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
    energy_transfer : bool
        Whether the prediction was carried out on energy transfer systems.
    absorbing_chain : bool
        Whether the system has at least one terminal combined state and the prediction
        was carried out on an absorbing Markov chain.
        Absorbing states have a lifetime of inf and a frequency / occupation of 0.
        Absorbing transitions have a frequency of 0.
    transition_set : fluopy.transitions.TransitionSet
        Collection of all relevant transitions and related attributes.
    initial_state_index : int
        Row of transition_set.combined_state_transitions_df whose final state defines
        the initial combined state used for absorbing predictions. This has no effect
        on non-absorbing predictions.
    frequency_transitions : npt.NDArray[np.float64]
        Relative number of expected transition occurrences, normalized separately for
        each fluorophore. Energy-transfer occurrences are assigned to the donor's
        transition group.
    frequency_states : dict[str, npt.NDArray[np.float64]]
        Relative expected number of visits to each state, normalized separately for each
        fluorophore.
    transition_time_distributions : npt.NDArray[object] | None
        Expected distributions of time until transition.
        Contains objects of type scipy.stats.*.rv_frozen for each transition.
        None if energy transfer is True.
    lifetime_distributions : dict[str, npt.NDArray[object]] | None
        Name of fluorophores as keys and their state's expected lifetime distributions
        (objects of type scipy.stats.*.rv_frozen) (array) as values.
        None if energy transfer is True.
    mean_transition_times : npt.NDArray[np.float64] | None
        Expected means of time until transition.
        None if energy transfer is True.
    mean_lifetimes : dict[str, npt.NDArray[np.float64]] | None
        Name of fluorophores as keys and their state's expected lifetime means (array)
        as values.
        None if energy transfer is True.
    state_occupations : dict[str, npt.NDArray[np.float64]] | None
        Relative time spent in each state, normalized separately for each fluorophore.
        None if energy transfer is True.

    Notes
    -----
    Predictions are available for systems containing at most two fluorophores.

    Predicted lifetimes and state occupations are not available for systems containing
    energy transfer.

    For non-absorbing systems, transition frequencies are calculated from the
    stationary distribution of the transition matrix. The transition matrix must have
    a unique stationary distribution.

    Systems containing terminal state combinations are treated separately as absorbing
    Markov chains.
    """

    def __init__(
        self,
        transition_set: TransitionSet,
        initial_state_index: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        transition_set
            Collection of all relevant transitions and related attributes.
        initial_state_index
            Row of transition_set.combined_state_transitions_df whose final state
            defines the initial combined state for absorbing predictions. This has no
            effect on non-absorbing predictions.
        """
        self.energy_transfer = False
        self.absorbing_chain = False
        if transition_set.fluorophore_system.count > 2:
            raise ValueError("prediction not available for more than 2 fluorophores.")
        self.transition_set = transition_set
        if not isinstance(initial_state_index, (int, np.integer)):
            raise ValueError("initial_state_index must be an integer.")
        if not 0 <= initial_state_index < transition_set.transition_matrix.shape[0]:
            raise ValueError(
                "initial_state_index must identify a row of the transition matrix."
            )
        self.initial_state_index = int(initial_state_index)
        self._terminal_state_combinations = self._get_terminal_state_combinations()
        if any(
            parse_energy_transfer_label(fluorophore_comb) is not None
            for fluorophore_comb in transition_set.transition_df.index.get_level_values(
                0
            )
        ):
            logger.warning(
                "Only frequencies are available for systems with energy transfer; "
                "lifetimes and occupations are not available.",
                stacklevel=2,
            )
            self.energy_transfer = True
        if self._terminal_state_combinations:
            logger.warning(
                "absorbing states have a lifetime of inf and a frequency / occupation "
                "of 0. Absorbing transitions have a frequency of 0.",
                stacklevel=2,
            )
            self.absorbing_chain = True
            if len(self._terminal_state_combinations) > 1:
                logger.warning(
                    "multiple terminal state combinations are available; predicted "
                    "transition frequencies depend on initial_state_index.",
                    stacklevel=2,
                )

        self.transition_time_distributions: npt.NDArray[Any] | None
        self.lifetime_distributions: dict[str, npt.NDArray[Any]] | None
        self.mean_transition_times: npt.NDArray[np.float64] | None
        self.mean_lifetimes: dict[str, npt.NDArray[np.float64]] | None
        self.state_occupations: dict[str, npt.NDArray[np.float64]] | None
        if self.absorbing_chain:
            self.frequency_transitions = self.predict_transition_occurrences_absorbing(
                initial_state_index=self.initial_state_index
            )
        else:
            self.frequency_transitions = self.predict_transition_occurrences()
        self.frequency_states = self.predict_state_occurrences()
        if not self.energy_transfer:
            (
                self.transition_time_distributions,
                self.lifetime_distributions,
            ) = self.predict_lifetimes()
            self.mean_transition_times = np.array(
                [distr.mean() for distr in self.transition_time_distributions]
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

    def _get_terminal_state_combinations(self) -> list[tuple[int, ...]]:
        """Return terminal state combinations in deterministic order."""
        return sorted(self.transition_set.terminal_state_combinations)

    def predict_transition_occurrences(self) -> npt.NDArray[np.float64]:
        """
        Predict the relative frequencies of transitions. Each different type of
        fluorophore's transitions frequencies sum up to 1.

        Each energy-transfer event is counted as one transition occurrence, including
        events that change both the donor and acceptor states. For normalization, an
        energy-transfer occurrence is assigned only to the donor's transition group;
        ordinary transitions are assigned to their respective fluorophore groups.

        Returns
        -------
        npt.NDArray[np.float64]
            Expected relative frequencies of each transition. Frequencies remain 0 for
            a fluorophore with no expected transitions.

        Notes
        -----
        The stationary distribution is calculated by solving pi P = pi together with
        the constraint that the entries of pi sum to 1. The transition matrix must
        have a unique stationary distribution. An all-zero transition matrix returns
        zero frequencies.
        """
        transition_matrix = self.transition_set.transition_matrix
        frequency_transitions = np.zeros(self.transition_set.transition_df.shape[0])
        if not np.any(transition_matrix):
            return frequency_transitions

        stationary_system = transition_matrix.T - np.identity(
            transition_matrix.shape[0]
        )
        stationary_system[-1] = 1
        normalization = np.zeros(transition_matrix.shape[0])
        normalization[-1] = 1
        stationary_distribution_combined_state_transitions = np.linalg.solve(
            stationary_system, normalization
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
        self, initial_state_index: int = 0
    ) -> npt.NDArray[np.float64]:
        """
        Predict the relative frequencies of transitions. Absorbing transitions will
        have the value 0. Combined states without positive-rate outgoing transitions
        are treated as absorbing.

        Each energy-transfer event is counted as one transition occurrence, including
        events that change both the donor and acceptor states. For normalization, an
        energy-transfer occurrence is assigned only to the donor's transition group;
        ordinary transitions are assigned to their respective fluorophore groups.

        Parameters
        ----------
        initial_state_index
            Row of the transition matrix used as the initial combined state.

        Returns
        -------
        npt.NDArray[np.float64]
            Expected relative frequencies of each transition. Frequencies remain 0 for
            a fluorophore with no expected transitions.
        """
        transition_abs = self.transition_set.transition_df["absorbing"]
        abs_indices = transition_abs[transition_abs].index.get_level_values(1)
        df = self.transition_set.combined_state_transitions_df
        abs_indices_combined = df[df["transition_id"].isin(abs_indices)].index
        terminal_state_combinations = set(self._terminal_state_combinations)
        drop_transitions = df.index[
            df["final_state"].map(lambda state: state in terminal_state_combinations)
        ]
        frequency_transitions = np.zeros(transition_abs.size)
        if initial_state_index in drop_transitions:
            return frequency_transitions
        drop_diff = abs_indices_combined[
            ~np.isin(abs_indices_combined, drop_transitions)
        ]
        Q = get_Q(
            P=self.transition_set.transition_matrix, drop_transitions=drop_transitions
        )
        I_t = get_I_t(Q=Q)
        N = get_N(I_t=I_t, Q=Q)
        initial_transient_index = initial_state_index - np.count_nonzero(
            drop_transitions < initial_state_index
        )
        expected_transient_visits = N[initial_transient_index]
        expected_visits = np.zeros(
            expected_transient_visits.size + drop_transitions.size,
            dtype=expected_transient_visits.dtype,
        )
        mask = np.ones(len(expected_visits), dtype=bool)
        mask[drop_transitions] = False
        expected_visits[mask] = expected_transient_visits
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

        State visits are counted separately for each physical fluorophore. An
        energy-transfer event therefore contributes a visit for both donor and
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
        grouped = self.transition_set.transition_df.groupby(level=0)
        for fluorophore_comb_raw, f_transitions in grouped:
            fluorophore_comb = cast(str, fluorophore_comb_raw)
            energy_transfer = parse_energy_transfer_label(fluorophore_comb)
            if energy_transfer is not None:
                d, a, _ = energy_transfer
                single_states_a = single_states[a]
                single_states_d = single_states[d]
                factor = 1.0
                for row_index, transition in f_transitions.iterrows():
                    identity = int(cast(tuple[Any, int], row_index)[1])
                    _, acceptor_i = transition["initial_state"].single_state_values
                    donor_f, acceptor_f = transition["final_state"].single_state_values
                    index_1 = np.where(single_states_d == donor_f)[0][0]
                    frequency_states[d][index_1] += (
                        self.frequency_transitions[identity] * factor
                    )
                    if acceptor_i != acceptor_f:
                        index_2 = np.where(single_states_a == acceptor_f)[0][0]
                        frequency_states[a][index_2] += (
                            self.frequency_transitions[identity] * factor
                        )

            else:
                single_states_f = single_states[fluorophore_comb]
                for row_index, transition in f_transitions.iterrows():
                    identity = int(cast(tuple[Any, int], row_index)[1])
                    index = np.where(
                        single_states_f == transition["final_state"].value
                    )[0][0]
                    frequency_states[fluorophore_comb][
                        index
                    ] += self.frequency_transitions[identity]
        for fluorophore, state_frequencies in frequency_states.items():
            total = state_frequencies.sum()
            if total > 0:
                frequency_states[fluorophore] /= total

        return frequency_states

    def predict_lifetimes(
        self,
    ) -> tuple[npt.NDArray[Any], dict[str, npt.NDArray[Any]]]:
        """
        Predict the lifetime distributions of states and the time until occurrence
        distributions of transitions.

        Returns
        -------
        transition_time_distributions : npt.NDArray[np.float64]
            Expected distributions of time until transition.
            Contains objects of type scipy.stats.*.rv_frozen for each transition.
        lifetime_distributions : dict[str, npt.NDArray[np.float64]]
            Name of fluorophores as keys and their state's expected lifetime
            distributions (objects of type scipy.stats.*.rv_frozen) (array) as values.
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
                total_rate = 0.0
                associated_transitions: list[int] = []
                for j, transition in self.transition_set.transition_df.loc[
                    fluorophore
                ].iterrows():
                    source = transition.initial_state.value
                    if source == state:
                        total_rate += transition.rate
                        associated_transitions.append(cast(int, j))
                if total_rate == 0:
                    lifetime_mean = np.inf
                    lifetime_pdf: Any = np.inf
                else:
                    lifetime_mean = 1 / total_rate
                    lifetime_pdf = expon(scale=lifetime_mean)
                lifetime_distributions[fluorophore][i] = lifetime_pdf
                transition_time_distributions[associated_transitions] = lifetime_pdf

        return transition_time_distributions, lifetime_distributions

    def infer_stats(
        self,
    ) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]:
        """
        Infers statistics of states based on lifetime distributions and frequencies.

        Returns
        -------
        mean_lifetimes : dict[str, npt.NDArray[np.float64]]
            Name of fluorophores as keys and their state's expected lifetime means
            (array) as values.
        state_occupations : dict[str, npt.NDArray[np.float64]]
            Name of fluorophores as keys and their state's expected probability of
            being occupied at any given point in time (array) as values.
        """
        lifetime_distributions = self.lifetime_distributions
        if lifetime_distributions is None:
            raise ValueError("lifetime statistics are unavailable for energy transfer.")
        mean_lifetimes: dict[str, npt.NDArray[np.float64]] = {}
        for fluorophore, distributions in lifetime_distributions.items():
            mean_lifetimes[fluorophore] = np.array(
                [distr.mean() if distr != np.inf else np.inf for distr in distributions]
            )
        return mean_lifetimes, calculate_state_occupations(
            self.frequency_states, mean_lifetimes
        )

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
        if self.energy_transfer:
            raise ValueError(
                "mean_transition_times not available if energy transfers possible."
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
        if self.energy_transfer:
            raise ValueError(
                "mean_lifetimes not available if energy transfers possible."
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
        if self.energy_transfer:
            raise ValueError(
                "state_occupations not available if energy transfers possible."
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
        if self.energy_transfer:
            raise ValueError(
                "lifetime_distributions not available if energy transfers possible."
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
        if self.energy_transfer:
            raise ValueError(
                "transition_time_distributions not available if energy transfers "
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


def get_Q(
    P: npt.ArrayLike, drop_transitions: int | npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """
    Q describes the probability of transitioning from some transient state to another.

    Parameters
    ----------
    P
        Transition matrix with transient states t and absorbing state r.
    drop_transitions
        Index of absorbing state (i.e., photophysical transition with no return).

    Returns
    -------
    npt.NDArray[np.float64]
        Transition matrix Q with transient states t.
    """
    # Q takes the original transition matrix into account, because within Q the state
    # that leads to the absorbing state has to take on the probability GIVEN the
    # possibility of the transition to the absorbing state.
    matrix = np.asarray(P, dtype=np.float64)
    indices = np.asarray(drop_transitions, dtype=np.int64)
    Q = np.delete(matrix, indices, axis=0)
    Q = np.delete(Q, indices, axis=1)

    return Q


def get_I_t(Q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    I_t is the identity matrix of Q.

    Parameters
    ----------
    Q
        Transition matrix with transient states t.

    Returns
    -------
    npt.NDArray[np.float64]
        Identity matrix I_t of Q.
    """
    matrix = np.asarray(Q, dtype=np.float64)
    I_t = np.identity(matrix.shape[0])

    return I_t


def get_N(I_t: npt.ArrayLike, Q: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """
    N is the fundamental matrix. At entry (i, j) it contains the expected number
    of visits to a transient state j starting from transient state i before being
    absorbed.

    Parameters
    ----------
    I_t
        Identity matrix of Q.
    Q
        Transition matrix with transient states t.

    Returns
    -------
    npt.NDArray[np.float64]
        Fundamental matrix N of absorbing Markov chain.
    """
    identity = np.asarray(I_t, dtype=np.float64)
    transition_matrix = np.asarray(Q, dtype=np.float64)
    N = np.linalg.inv(identity - transition_matrix)

    return np.asarray(N, dtype=np.float64)
