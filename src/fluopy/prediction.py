"""
Compute a prediction for a photophysical system.
"""

from __future__ import annotations

import logging
import re
from itertools import product
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
from scipy.stats import expon

from . import figure as fi
from .miscellaneous import format_electronic_state, format_transition

if TYPE_CHECKING:

    from .transitions import TransitionSet

__all__: list[str] = ["Prediction"]

logger = logging.getLogger(__name__)

_ENERGY_TRANSFER_LABEL = re.compile(
    r"D:\s*([^,]+),\s*A:\s*([^,]+),\s*dist:\s*(\d+(?:\.\d+)?)\s*"
)


class Prediction:
    """
    Container of mathematically derived statistical attributes and methods.

    Attributes
    ----------
    energy_transfer : bool
        Whether the prediction was carried out on energy transfer systems.
    absorbing_chain : bool
        Whether every fluorophore has at least one absorbing state and the prediction
        was carried out on an absorbing Markov chain.
        Absorbing states have a lifetime of inf and a frequency / occupation of 0.
        Absorbing transitions have a frequency of 0.
    transition_set : fluopy.transitions.TransitionSet
        Collection of all relevant transitions and related attributes.
    initial_state_index : int
        Row of transition_set.combined_state_transitions_df whose final state defines
        the initial combined state used for the prediction.
    frequency_transitions : npt.NDArray[np.float64]
        Relative number of expected transition occurrences, normalized separately for
        each fluorophore.
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

    For non-absorbing systems, transition frequencies are approximated from a large
    power of the transition matrix. This requires an irreducible, aperiodic Markov
    chain for convergence to a unique limiting distribution.

    Systems containing absorbing transitions are treated separately as absorbing
    Markov chains. Predicted lifetimes and state occupations are not available for
    systems containing energy transfer.
    """

    def __init__(
        self,
        transition_set: TransitionSet,
        matrix_power: float = 1e9,
        initial_state_index: int = 0,
    ) -> None:
        """
        Parameters
        ----------
        transition_set
            Collection of all relevant transitions and related attributes.
        matrix_power
            Exponent used to approximate the limiting distribution of a non-absorbing
            transition matrix. Must be a positive integer-valued number. Larger values
            allow more steps toward convergence but do not directly specify numerical
            accuracy.
        initial_state_index
            Row of transition_set.combined_state_transitions_df whose final state
            defines the initial combined state. This row is used in both absorbing and
            non-absorbing predictions.
        """
        if not np.isfinite(matrix_power) or matrix_power <= 0:
            raise ValueError("matrix_power must be a positive, finite integer.")
        if not float(matrix_power).is_integer():
            raise ValueError("matrix_power must be an integer-valued number.")

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
        self._absorbing_state_combinations = self._get_absorbing_state_combinations()
        # too large matrix in np.linalg.matrix_power
        if any(
            _ENERGY_TRANSFER_LABEL.fullmatch(fluorophore_comb) is not None
            for fluorophore_comb in transition_set.transition_df.index.get_level_values(
                0
            )
        ):
            logger.warning(
                "prediction accuracy of energy transfers more difficult to tune. Only "
                "frequencies available, lifetimes and occupations not available.",
                stacklevel=2,
            )
            self.energy_transfer = True
        has_absorbing_states = bool(transition_set.transition_df["absorbing"].any())
        if has_absorbing_states and not self._absorbing_state_combinations:
            raise ValueError(
                "absorbing states must be defined for every fluorophore or for none."
            )
        if self._absorbing_state_combinations:
            logger.warning(
                "absorbing states have a lifetime of inf and a frequency / occupation "
                "of 0. Absorbing transitions have a frequency of 0.",
                stacklevel=2,
            )
            self.absorbing_chain = True
            if len(self._absorbing_state_combinations) > 1:
                logger.warning(
                    "multiple absorbing combined states are available; predicted "
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
            self.frequency_transitions = self.predict_transition_occurrences(
                matrix_power=int(matrix_power),
                initial_state_index=self.initial_state_index,
            )
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

    def _get_absorbing_state_combinations(self) -> list[tuple[int, ...]]:
        absorbing = self.transition_set.transition_df["absorbing"]
        absorbing_transition_df = self.transition_set.transition_df[absorbing]
        absorbing_states_by_fluorophore: list[npt.NDArray[np.int64]] = []
        for fluorophore in self.transition_set.fluorophore_system.fluorophores:
            if fluorophore.name not in absorbing_transition_df.index.get_level_values(
                0
            ):
                return []
            final_states = absorbing_transition_df["final_state"].xs(
                fluorophore.name, level=0
            )
            absorbing_state_values = final_states.map(
                lambda state: state.value
            ).to_numpy(dtype=np.int64)
            absorbing_states_by_fluorophore.append(np.unique(absorbing_state_values))

        return [
            tuple(int(state) for state in state_combination)
            for state_combination in product(*absorbing_states_by_fluorophore)
        ]

    def predict_transition_occurrences(
        self, matrix_power: int, initial_state_index: int = 0
    ) -> npt.NDArray[np.float64]:
        """
        Predict the relative frequencies of transitions. Each different type of
        fluorophore's transitions frequencies sum up to 1.

        Parameters
        ----------
        matrix_power
            Exponent used to approximate the limiting distribution of the transition
            matrix.
        initial_state_index
            Row of the transition matrix used as the initial combined state.

        Returns
        -------
         npt.NDArray[np.float64]
            Expected relative frequencies of each transition. Frequencies remain 0 for
            a fluorophore with no expected transitions.

        Notes
        -----
        The transition matrix must describe an irreducible, aperiodic Markov chain for
        its powers to converge to a unique limiting distribution.
        """
        powered_transition_matrix = np.linalg.matrix_power(
            self.transition_set.transition_matrix, n=matrix_power
        )
        stationary_distribution_combined_state_transitions = powered_transition_matrix[
            initial_state_index
        ]
        # https://brilliant.org/wiki/stationary-distributions/
        frequency_transitions = np.zeros(self.transition_set.transition_df.shape[0])
        df = self.transition_set.combined_state_transitions_df
        for _, i in self.transition_set.transition_df.index:
            indices = df.index[df["transition_id"] == i].tolist()
            frequency_transitions[i] = (
                stationary_distribution_combined_state_transitions[indices].sum()
            )

        grouper: dict[str, list[int]] = {}
        for fluorophore_comb_raw, group in self.transition_set.transition_df.groupby(
            level=0, sort=False
        ):
            fluorophore_comb = cast(str, fluorophore_comb_raw)
            match = _ENERGY_TRANSFER_LABEL.fullmatch(fluorophore_comb)
            if match is not None:
                d, _, _ = match.group(1), match.group(2), match.group(3)
            else:
                d = fluorophore_comb
            if d in grouper:
                grouper[d] += group.index.get_level_values(1).tolist()
            else:
                grouper[d] = group.index.get_level_values(1).tolist()

        for _, indices in grouper.items():
            total = np.sum(frequency_transitions[indices])
            if total > 0:
                frequency_transitions[indices] /= total

        return frequency_transitions

    def predict_transition_occurrences_absorbing(
        self, initial_state_index: int = 0
    ) -> npt.NDArray[np.float64]:
        """
        Predict the relative frequencies of transitions. Absorbing transitions will
        have the value 0. Every combination of the fluorophores' absorbing states is
        treated as an absorbing combined state.

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
        absorbing_state_combinations = set(self._absorbing_state_combinations)
        drop_transitions = df.index[
            df["final_state"].map(lambda state: state in absorbing_state_combinations)
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

        grouper: dict[str, list[int]] = {}
        for fluorophore_comb_raw, group in self.transition_set.transition_df.groupby(
            level=0, sort=False
        ):
            fluorophore_comb = cast(str, fluorophore_comb_raw)
            match = _ENERGY_TRANSFER_LABEL.fullmatch(fluorophore_comb)
            if match is not None:
                d, _, _ = match.group(1), match.group(2), match.group(3)
            else:
                d = fluorophore_comb
            if d in grouper:
                grouper[d] += group.index.get_level_values(1).tolist()
            else:
                grouper[d] = group.index.get_level_values(1).tolist()

        for _, indices in grouper.items():
            total = np.sum(frequency_transitions[indices])
            if total > 0:
                frequency_transitions[indices] /= total

        return frequency_transitions

    def predict_state_occurrences(self) -> dict[str, npt.NDArray[np.float64]]:
        """
        Predict the relative frequencies of states. Each different type of fluorophore's
        states frequencies sum up to 1.

        Returns
        -------
        frequency_states : dict[str, npt.NDArray[np.float64]]
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
            match = _ENERGY_TRANSFER_LABEL.fullmatch(fluorophore_comb)
            if match is not None:
                d, a, _ = match.group(1), match.group(2), match.group(3)
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
                        if d == a:
                            factor = 0.5
                            # factor to adjust that this energy transfer effects two
                            # fluorophores of the same type, not only one
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
        state_occupations: dict[str, npt.NDArray[np.float64]] = {}
        for fluorophore, distributions in lifetime_distributions.items():
            mean_lifetimes[fluorophore] = np.array(
                [distr.mean() if distr != np.inf else np.inf for distr in distributions]
            )
            state_occupations[fluorophore] = np.multiply(
                self.frequency_states[fluorophore],
                mean_lifetimes[fluorophore],
                where=mean_lifetimes[fluorophore] != np.inf,
                out=np.zeros(self.frequency_states[fluorophore].size),
            )
            total_occupation = state_occupations[fluorophore].sum()
            if total_occupation > 0:
                state_occupations[fluorophore] /= total_occupation

        return mean_lifetimes, state_occupations

    def plot_frequency_transitions(self, **kwargs: Any) -> fi.AxesArray:
        """
        Plot frequencies of transitions.

        Parameters
        ----------
        kwargs
            kwargs for fluopy.figure.universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        df = self.transition_set.transition_df
        data = [np.arange(df.shape[0]), self.frequency_transitions]
        kwargs.setdefault("type_", "bar")
        kwargs.setdefault("xlabel", None)
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("xticks", range(df.shape[0]))
        kwargs.setdefault(
            "xticklabels",
            dict(labels=df["abbreviation"].apply(format_transition), rotation=70),
        )
        colormap = mpl.colors.ListedColormap(
            [
                mpl.colormaps["Spectral"](value)
                for value in np.linspace(0, 1, df.index.get_level_values(0).nunique())
            ]
        )
        kwargs.setdefault(
            "color",
            [
                colormap(i)
                for i, size in enumerate(df.groupby(level=0, sort=False).size())
                for _ in range(size)
            ],
        )
        kwargs.setdefault("ylabel", "Prob. occurrence")
        kwargs.setdefault("legend", True)
        kwargs.setdefault(
            "legendhandles",
            [
                mpl.patches.Patch(color=colormap(i), label=name)
                for i, name in enumerate(df.index.get_level_values(0).unique())
            ],
        )
        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_frequency_states(self, **kwargs: Any) -> fi.AxesArray:
        """
        Plot frequencies of states.

        Parameters
        ----------
        kwargs
            kwargs for fluopy.figure.universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """

        single_states = self.transition_set.single_states
        colormap = mpl.colors.ListedColormap(
            [
                mpl.colormaps["Spectral"](value)
                for value in np.linspace(0, 1, len(single_states))
            ]
        )
        colors: list[Any] = []
        patches: list[Any] = []
        xticks = 0
        data_parts: list[npt.NDArray[np.float64]] = []
        labels: list[str] = []
        for i, (fluorophore, states) in enumerate(single_states.items()):
            colors.extend([colormap(i) for _ in range(states.size)])
            patches.append(mpl.patches.Patch(color=colormap(i), label=fluorophore))
            xticks += states.size
            data_parts.append(self.frequency_states[fluorophore])
            labels.extend(
                [
                    format_electronic_state(
                        label=self.transition_set.states_by_value[identity].name
                    )
                    for identity in states
                ]
            )
        data_merged = np.concatenate(data_parts)
        data = [np.arange(xticks), data_merged]
        kwargs.setdefault("type_", "bar")
        kwargs.setdefault("xlabel", None)
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("xticks", range(xticks))
        kwargs.setdefault("xticklabels", dict(labels=labels, rotation=70))
        kwargs.setdefault("ylabel", "Prob. occurrence")
        kwargs.setdefault("color", colors)
        kwargs.setdefault("legend", True)
        kwargs.setdefault("legendhandles", patches)
        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_mean_transition_times(self, **kwargs: Any) -> fi.AxesArray:
        """
        Plot mean times until transitions occur.

        Parameters
        ----------
        kwargs
            kwargs for fluopy.figure.universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.energy_transfer:
            raise ValueError(
                "mean_transition_times not available if energy transfers possible."
            )
        mean_transition_times = self.mean_transition_times
        if mean_transition_times is None:
            raise ValueError("mean transition times are unavailable.")
        df = self.transition_set.transition_df
        data = [np.arange(df.shape[0]), mean_transition_times]
        kwargs.setdefault("type_", "bar")
        kwargs.setdefault("xlabel", None)
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("xticks", range(df.shape[0]))
        kwargs.setdefault(
            "xticklabels",
            dict(labels=df["abbreviation"].apply(format_transition), rotation=70),
        )
        colormap = mpl.colors.ListedColormap(
            [
                mpl.colormaps["Spectral"](value)
                for value in np.linspace(0, 1, df.index.get_level_values(0).nunique())
            ]
        )
        kwargs.setdefault(
            "color",
            [
                colormap(i)
                for i, size in enumerate(df.groupby(level=0, sort=False).size())
                for _ in range(size)
            ],
        )
        kwargs.setdefault("ylabel", r"$\tau$ (s)")
        kwargs.setdefault("legend", True)
        kwargs.setdefault(
            "legendhandles",
            [
                mpl.patches.Patch(color=colormap(i), label=name)
                for i, name in enumerate(df.index.get_level_values(0).unique())
            ],
        )
        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_mean_lifetimes(self, **kwargs: Any) -> fi.AxesArray:
        """
        Plot mean lifetimes of states.

        Parameters
        ----------
        kwargs
            kwargs for fluopy.figure.universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.energy_transfer:
            raise ValueError(
                "mean_lifetimes not available if energy transfers possible."
            )
        mean_lifetimes = self.mean_lifetimes
        if mean_lifetimes is None:
            raise ValueError("mean lifetimes are unavailable.")

        single_states = self.transition_set.single_states
        colormap = mpl.colors.ListedColormap(
            [
                mpl.colormaps["Spectral"](value)
                for value in np.linspace(0, 1, len(single_states))
            ]
        )
        colors: list[Any] = []
        patches: list[Any] = []
        xticks = 0
        data_parts: list[npt.NDArray[np.float64]] = []
        labels: list[str] = []
        for i, (fluorophore, states) in enumerate(single_states.items()):
            colors.extend([colormap(i) for _ in range(states.size)])
            patches.append(mpl.patches.Patch(color=colormap(i), label=fluorophore))
            xticks += states.size
            data_parts.append(mean_lifetimes[fluorophore])
            labels.extend(
                [
                    format_electronic_state(
                        label=self.transition_set.states_by_value[identity].name
                    )
                    for identity in states
                ]
            )
        data_merged = np.concatenate(data_parts)
        data = [np.arange(xticks), data_merged]
        kwargs.setdefault("type_", "bar")
        kwargs.setdefault("xlabel", None)
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("xticks", range(xticks))
        kwargs.setdefault("xlim", [-1, xticks])
        kwargs.setdefault("xticklabels", dict(labels=labels, rotation=70))
        kwargs.setdefault("color", colors)
        kwargs.setdefault("legend", True)
        kwargs.setdefault("legendhandles", patches)
        kwargs.setdefault("ylabel", r"$\tau$ (s)")
        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_state_occupations(self, **kwargs: Any) -> fi.AxesArray:
        """
        Plot state occupation times (relative total time spent in state).

        Parameters
        ----------
        kwargs
            kwargs for fluopy.figure.universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.energy_transfer:
            raise ValueError(
                "state_occupations not available if energy transfers possible."
            )
        state_occupations = self.state_occupations
        if state_occupations is None:
            raise ValueError("state occupations are unavailable.")

        single_states = self.transition_set.single_states
        colormap = mpl.colors.ListedColormap(
            [
                mpl.colormaps["Spectral"](value)
                for value in np.linspace(0, 1, len(single_states))
            ]
        )
        colors: list[Any] = []
        patches: list[Any] = []
        xticks = 0
        data_parts: list[npt.NDArray[np.float64]] = []
        labels: list[str] = []
        for i, (fluorophore, states) in enumerate(single_states.items()):
            colors.extend([colormap(i) for _ in range(states.size)])
            patches.append(mpl.patches.Patch(color=colormap(i), label=fluorophore))
            xticks += states.size
            data_parts.append(state_occupations[fluorophore])
            labels.extend(
                [
                    format_electronic_state(
                        label=self.transition_set.states_by_value[identity].name
                    )
                    for identity in states
                ]
            )
        data_merged = np.concatenate(data_parts)
        data = [np.arange(xticks), data_merged]
        kwargs.setdefault("type_", "bar")
        kwargs.setdefault("xlabel", None)
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("xticks", range(xticks))
        kwargs.setdefault("xticklabels", dict(labels=labels, rotation=70))
        kwargs.setdefault("ylabel", "Prob. occupation")
        kwargs.setdefault("color", colors)
        kwargs.setdefault("legend", True)
        kwargs.setdefault("legendhandles", patches)
        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_lifetime_distributions(
        self,
        fluorophore: str,
        state_identity: int,
        x: npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> fi.AxesArray:
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
            kwargs for fluopy.figure.universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
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
        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_transition_time_distributions(
        self,
        fluorophore: str,
        transition_id: int,
        x: npt.ArrayLike | None = None,
        **kwargs: Any,
    ) -> fi.AxesArray:
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
            kwargs for fluopy.figure.universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
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

        axes = fi.universal_figure(data=data, **kwargs)

        return axes


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
