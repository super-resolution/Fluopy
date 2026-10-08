"""
Analysis of a photophysical simulation.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import numpy.typing as npt
import pandas as pd

from . import plotting
from ._statistics import (
    calculate_state_occupations,
    normalize_transition_frequencies,
    parse_paired_transition_label,
)
from .plotting import format_electronic_state, format_transition

if TYPE_CHECKING:
    from matplotlib.axes import Axes as mplAxes

    from .prediction import Prediction
    from .simulation import Simulation


__all__: list[str] = ["Analysis"]

logger = logging.getLogger(__name__)


class Analysis:
    """
    Container of simulation-derived statistical attributes and methods.

    Attributes
    ----------
    simulation : fluopy.simulation.Simulation
        Container for simulation-associated attributes.
    frequency_transitions : npt.NDArray[np.float64]
        Relative number of simulated transition occurrences, normalized separately for
        each fluorophore. Paired-transition occurrences are assigned to the donor's
        transition group.
    frequency_states : dict[str, npt.NDArray[np.float64]]
        Relative simulated number of visits to each state, normalized separately for
        each fluorophore.
    transition_time_distributions : Collection
        Contains 1-D array_like for each transition (time until the transition).
    lifetime_distributions : dict
        Name of fluorophores as keys and collections of their state's simulated
        lifetimes (1-D array_like) as values. The final residence may be right-censored
        by the observation limit and is not included.
    mean_transition_times : 1-D array_like
        Simulated means of time until transition.
    mean_lifetimes : dict
        Name of fluorophores as keys and their state's simulated lifetime means (array)
        as values.
    state_occupations : dict[str, npt.NDArray[np.float64]]
        Relative time spent in each state, normalized separately for each fluorophore.
    """

    def __init__(self, simulation: Simulation) -> None:
        """
        Parameters
        ----------
        simulation
            Container for simulation-associated attributes.
        """
        if (
            simulation.transition_series is None
            or simulation.state_series is None
            or simulation.time_series is None
        ):
            raise ValueError("analysis not available if simulation has not been run.")

        self.simulation: Simulation = simulation
        self.transition_series: npt.NDArray[np.int64] = cast(
            npt.NDArray[np.int64], simulation.transition_series
        )
        self.state_series: npt.NDArray[np.int64] = cast(
            npt.NDArray[np.int64], simulation.state_series
        )
        self.time_series: npt.NDArray[np.float64] = simulation.time_series
        self.frequency_transitions: npt.NDArray[np.float64]
        self.frequency_states: dict[str, npt.NDArray[np.float64]]
        self.transition_time_distributions: list[npt.NDArray[np.float64]]
        self.lifetime_distributions: dict[str, list[npt.NDArray[np.float64]]]
        self.mean_transition_times: npt.NDArray[np.float64]
        self.mean_lifetimes: dict[str, npt.NDArray[np.float64]]
        self.state_occupations: dict[str, npt.NDArray[np.float64]]

        absorbing = self.is_absorbing()
        if absorbing:
            logger.warning(
                "if a fluorophore reaches its individual absorbing state, it has an "
                "absolute state and transition frequency of 1. Completed lifetime "
                "estimates may be nan.",
                stacklevel=2,
            )

        self.frequency_transitions = self.get_transition_occurrences()
        self.frequency_states = self.get_state_occurrences()
        self.transition_time_distributions, self.lifetime_distributions = (
            self.get_lifetimes()
        )
        self.mean_transition_times = np.array(
            [
                (
                    np.mean(transition_time_distribution)
                    if transition_time_distribution.size > 0
                    else np.nan
                )
                for transition_time_distribution in self.transition_time_distributions
            ]
        )
        self.mean_lifetimes, self.state_occupations = self.infer_stats()

    def is_absorbing(self) -> bool:
        """
        Check whether any fluorophore reached one of its individual absorbing states.

        Returns
        -------
        bool
            Whether at least one fluorophore reached one of its individual absorbing
            states.
        """
        absorbing_states = self.simulation.transition_set.absorbing_states

        reached_absorbing_state = False
        for i, state_series in enumerate(self.state_series):
            last_state = state_series[-1]
            if last_state in absorbing_states.get(i, np.array([], dtype=np.int64)):
                reached_absorbing_state = True
                logger.info(
                    "fluorophore %d has reached the Markovian absorbing state %s",
                    i,
                    self.simulation.transition_set.states_by_value[last_state].name,
                )

        return reached_absorbing_state

    def get_transition_occurrences(self) -> npt.NDArray[np.float64]:
        """
        Get the relative frequencies of simulated transition occurrences.

        Each paired-transition event is counted as one transition occurrence, including
        events that change both the donor and acceptor states. For normalization, a
        paired-transition occurrence is assigned only to the donor's transition group;
        ordinary transitions are assigned to their respective fluorophore groups.

        Returns
        -------
        npt.NDArray[np.float64]
            Relative number of simulated transition occurrences, normalized separately
            for each fluorophore. Frequencies remain 0 for a fluorophore with no
            observed transitions.
        """
        df = self.simulation.transition_set.combined_state_transitions_df
        transition_ids = df["transition_id"].to_numpy(dtype=np.int64)
        simulated_transition_ids = transition_ids[self.transition_series]
        frequency_transitions = np.bincount(
            simulated_transition_ids,
            minlength=self.simulation.transition_set.transition_df.shape[0],
        ).astype(np.float64)

        return normalize_transition_frequencies(
            frequency_transitions, self.simulation.transition_set.transition_df
        )

    def get_state_occurrences(self) -> dict[str, npt.NDArray[np.float64]]:
        """
        Get the relative frequencies of simulated state visits.

        State visits are counted separately for each physical fluorophore. A
        paired-transition event therefore contributes a visit for both donor and
        acceptor if both states change, while still representing one transition
        occurrence.

        Returns
        -------
        dict[str, npt.NDArray[np.float64]]
            Relative simulated number of visits to each state, normalized separately
            for each fluorophore. Frequencies remain 0 if no state visits were counted.
            A trajectory without state changes counts as one visit to its initial state.
        """
        single_states = self.simulation.transition_set.single_states
        occurrences_states = {
            key: np.zeros(len(value)) for key, value in single_states.items()
        }
        for i, state_series_fluorophore in enumerate(self.state_series):
            fluorophore = (
                self.simulation.transition_set.fluorophore_system.fluorophores[i].name
            )
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            if changes_at.size == 0:
                states = state_series_fluorophore[:1]
            else:
                last_state = changes_at[-1] + 1
                changes_at_and_last = np.append(changes_at, last_state)
                states = state_series_fluorophore[changes_at_and_last]
            state_ids, state_counts = np.unique(states, return_counts=True)
            _, corresponding_indices, _ = np.intersect1d(
                ar1=single_states[fluorophore],
                ar2=state_ids,
                assume_unique=True,
                return_indices=True,
            )

            occurrences_states[fluorophore][corresponding_indices] += state_counts

        frequency_states = {}
        for fluorophore, occurrences in occurrences_states.items():
            total = np.sum(occurrences)
            frequency_states[fluorophore] = (
                occurrences / total if total > 0 else occurrences
            )

        return frequency_states

    def get_lifetimes(
        self,
    ) -> tuple[
        list[npt.NDArray[np.float64]],
        dict[str, list[npt.NDArray[np.float64]]],
    ]:
        """
        Get the lifetime distributions of states and the time until occurrence
        distributions of transitions.

        A paired-transition event that does not change the acceptor state does not
        interrupt the acceptor's state lifetime. For paired transitions, time to
        transition is collected only from the donor's point of view. State lifetime
        distributions, including those of S1, do not distinguish intervals in which
        a paired transition was possible from intervals in which it was not.

        Only completed residence intervals are included; the final right-censored
        interval is excluded. A fluorophore without state changes therefore contributes
        no lifetime samples.

        Returns
        -------
        transition_time_distributions : list[npt.NDArray[np.float64]]
            Contains 1-D array_like for each transition (time until the transition).
        lifetime_distributions : dict[str, npt.NDArray[np.float64]]
            Name of fluorophores as keys and collections of their state's simulated
            lifetimes (1-D array_like) as values.
        """
        single_states = self.simulation.transition_set.single_states
        combined_transition_df = (
            self.simulation.transition_set.combined_state_transitions_df
        )
        lifetime_parts: dict[str, list[list[npt.NDArray[np.float64]]]] = {
            key: [[] for _ in range(len(value))] for key, value in single_states.items()
        }
        transition_time_parts: list[list[npt.NDArray[np.float64]]] = [
            [] for _ in range(self.simulation.transition_set.transition_df.shape[0])
        ]
        transition_ids = combined_transition_df["transition_id"].to_numpy(
            dtype=np.int64
        )
        combined_transition_indices = np.asarray(self.transition_series, dtype=np.int64)
        event_transition_ids = transition_ids[combined_transition_indices]
        involved_fluorophore_ids = combined_transition_df["fluorophore_ids"].to_numpy(
            dtype=object
        )[combined_transition_indices]
        initiating_fluorophore_ids = np.fromiter(
            (fluorophore_ids[0] for fluorophore_ids in involved_fluorophore_ids),
            dtype=np.int64,
            count=self.transition_series.size,
        )
        event_times = self.time_series[1 : self.transition_series.size + 1]
        state_changes = np.diff(self.state_series, axis=1) != 0
        for fluorophore_id in range(self.state_series.shape[0]):
            initiated = initiating_fluorophore_ids == fluorophore_id
            resets = initiated | state_changes[fluorophore_id]
            reset_indices = np.flatnonzero(resets)
            reset_times = event_times[reset_indices]
            reset_intervals = np.diff(np.insert(reset_times, 0, self.time_series[0]))
            initiated_resets = initiated[reset_indices]
            initiated_transition_ids = event_transition_ids[reset_indices][
                initiated_resets
            ]
            initiated_intervals = reset_intervals[initiated_resets]
            # Repeated masks outperform sorting for the typically small number of
            # transition definitions in Fluopy, including for long trajectories.
            for _, transition_id in self.simulation.transition_set.transition_df.index:
                transition_time_parts[transition_id].append(
                    initiated_intervals[initiated_transition_ids == transition_id]
                )

        for fluorophore_id, fluorophore_state_series in enumerate(self.state_series):
            fluorophore = (
                self.simulation.transition_set.fluorophore_system.fluorophores[
                    fluorophore_id
                ].name
            )
            change_indices = np.flatnonzero(state_changes[fluorophore_id])
            if change_indices.size == 0:
                continue
            changed_state_indices = change_indices + 1
            initial_states = fluorophore_state_series[change_indices]
            state_change_times = self.time_series[changed_state_indices]
            residence_times = np.diff(state_change_times)
            residence_times = np.insert(
                arr=residence_times,
                obj=0,
                values=state_change_times[0],
            )
            for state_index, state in enumerate(single_states[fluorophore]):
                state_residence_times = residence_times[
                    np.where(initial_states == state)
                ]
                lifetime_parts[fluorophore][state_index].append(state_residence_times)

        lifetime_distributions = {
            fluorophore: [
                np.concatenate(parts) if parts else np.array([], dtype=np.float64)
                for parts in state_parts
            ]
            for fluorophore, state_parts in lifetime_parts.items()
        }
        transition_time_distributions = [
            np.concatenate(parts) if parts else np.array([], dtype=np.float64)
            for parts in transition_time_parts
        ]

        return transition_time_distributions, lifetime_distributions

    def infer_stats(
        self,
    ) -> tuple[dict[str, npt.NDArray[np.float64]], dict[str, npt.NDArray[np.float64]]]:
        """
        Infer mean lifetimes and relative state occupations from lifetime distributions
        and state frequencies.

        Returns
        -------
        mean_lifetimes : dict[str, npt.NDArray[np.float64]]
            Name of fluorophores as keys and their state's simulated lifetime means
            (array) as values.
        state_occupations : dict[str, npt.NDArray[np.float64]]
            Relative time spent in each state, normalized separately for each
            fluorophore.
        """
        mean_lifetimes: dict[str, npt.NDArray[np.float64]] = {}
        for fluorophore, distributions in self.lifetime_distributions.items():
            mean_lifetimes[fluorophore] = np.array(
                [
                    np.mean(distr) if distr.size != 0 else np.nan
                    for distr in distributions
                ]
            )
        state_occupations = calculate_state_occupations(
            self.frequency_states, mean_lifetimes
        )
        if self.transition_series.size:
            combined_transition_df = (
                self.simulation.transition_set.combined_state_transitions_df
            )
            observed_transitions = combined_transition_df.iloc[self.transition_series]
            has_self_event = any(
                initial_state == final_state
                for initial_state, final_state in zip(
                    observed_transitions["initial_state"],
                    observed_transitions["final_state"],
                    strict=True,
                )
            )
            if has_self_event:
                for fluorophore, occupations in state_occupations.items():
                    if occupations.sum() == 0:
                        state_occupations[fluorophore] = self.frequency_states[
                            fluorophore
                        ].copy()

        return mean_lifetimes, state_occupations

    def get_fluorescence_lifetimes(
        self, fluorophore: str | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Get the fluorescence lifetime (i.e., S1 lifetime) of the specified fluorophore.
        Note that this does not consider whether the S1 state decays via photon
        emission.

        Parameters
        ----------
        fluorophore
            The name of the fluorophore whose fluorescence lifetime is to be returned.

        Returns
        -------
        npt.NDArray[np.float64]
            The fluorescence lifetimes of the specified fluorophore.
        """
        s1_value = 1  # hardcoded but covered by tests

        if fluorophore is not None:
            if fluorophore not in self.lifetime_distributions:
                raise ValueError(
                    f"fluorophore {fluorophore} not found in lifetime_distributions."
                )
        if len(self.lifetime_distributions) == 1:
            fluorophore = list(self.lifetime_distributions.keys())[0]
        else:
            if fluorophore is None:
                raise ValueError(
                    "if multiple fluorophores are present, fluorophore must be "
                    "specified."
                )
        s1_index = np.where(
            self.simulation.transition_set.single_states[fluorophore] == s1_value
        )[0][0]

        fluorescence_lifetimes = self.lifetime_distributions[fluorophore][s1_index]

        return np.asarray(fluorescence_lifetimes, dtype=np.float64)

    def get_emitting_transition_lifetimes(
        self, fluorophore: str | None = None
    ) -> npt.NDArray[np.float64]:
        """
        Get the lifetimes of the emitting transitions (i.e., S1 deexcitation via photon
        emission) of the specified fluorophore.

        Parameters
        ----------
        fluorophore
            The name of the fluorophore whose fluorescence lifetime is to be returned.

        Returns
        -------
        npt.NDArray[np.float64]
            The fluorescence lifetimes (photon emission) of the specified fluorophore.
        """
        fluorophores = []
        for key, _ in self.simulation.transition_set.single_states.items():
            fluorophores.append(key)
        if fluorophore is not None:
            if fluorophore not in fluorophores:
                raise ValueError(
                    f"fluorophore {fluorophore} not found in transition dataframe."
                )
        if len(fluorophores) == 1:
            fluorophore = fluorophores[0]
        else:
            if fluorophore is None:
                raise ValueError(
                    "if multiple fluorophores are present, fluorophore must be "
                    "specified."
                )
        sub_df = self.simulation.transition_set.transition_df.loc[fluorophore]
        emitting_transitions_f = sub_df[sub_df["photon"]].index.to_numpy()
        lifetime_parts = [
            self.transition_time_distributions[emitting_transition_f]
            for emitting_transition_f in emitting_transitions_f
        ]
        exp_fluorescence_lifetimes = np.concatenate(lifetime_parts)

        return exp_fluorescence_lifetimes

    def plot_frequency_transitions(
        self,
        prediction: Prediction | None = None,
        diff_dist: bool = True,
        **kwargs: Any,
    ) -> mplAxes:
        """
        Plot relative frequencies of simulated transition occurrences.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        diff_dist
            Whether to plot paired transitions distance-specific or not.
        kwargs
            kwargs for fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        transition_df = self.simulation.transition_set.transition_df
        frequencies = self.frequency_transitions
        if not diff_dist:
            transition_df, discarded_ids_by_retained_position, discarded_ids = (
                no_diff_dist(
                    transition_df=transition_df,
                    fluorophores=self.simulation.transition_set.single_states.keys(),
                )
            )
            collapsed_frequencies = np.delete(frequencies, discarded_ids)
            for (
                retained_position,
                discarded_transition_ids,
            ) in discarded_ids_by_retained_position.items():
                collapsed_frequencies[retained_position] += np.sum(
                    frequencies[discarded_transition_ids]
                )
            frequencies = collapsed_frequencies
        legend_labels = [
            (
                name.rsplit(", dist:", maxsplit=1)[0]
                if parse_paired_transition_label(name) is not None and not diff_dist
                else name
            )
            for name in transition_df.index.get_level_values(0).unique()
        ]

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction uses a different TransitionSet object; verify that "
                    "states and transition ordering are compatible.",
                    stacklevel=2,
                )
            predicted_frequencies = prediction.frequency_transitions
            # Prediction supports at most one distance for each fluorophore pair, so
            # distance-specific collapsing is only required for the simulation.
            if predicted_frequencies.shape != frequencies.shape:
                raise ValueError(
                    "prediction and simulation have incompatible transition dimensions."
                )
            draw_marker = [
                np.arange(transition_df.shape[0]),
                predicted_frequencies,
            ]

        return plotting._plot_transition_bars(
            transition_df=transition_df,
            values=frequencies,
            default_ylabel="Prob. occurrence",
            draw_marker=draw_marker,
            legend_labels=legend_labels,
            **kwargs,
        )

    def plot_frequency_states(
        self, prediction: Prediction | None = None, **kwargs: Any
    ) -> mplAxes:
        """
        Plot relative frequencies of simulated state visits.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """

        transition_set = self.simulation.transition_set
        data_merged = plotting._flatten_state_values(
            transition_set, self.frequency_states
        )

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction uses a different TransitionSet object; verify that "
                    "states and transition ordering are compatible.",
                    stacklevel=2,
                )
            predicted_frequencies = plotting._flatten_state_values(
                transition_set, prediction.frequency_states
            )
            if predicted_frequencies.shape != data_merged.shape:
                raise ValueError(
                    "prediction and simulation have incompatible state dimensions."
                )
            draw_marker = [
                np.arange(data_merged.size),
                predicted_frequencies,
            ]

        return plotting._plot_state_bars(
            transition_set=transition_set,
            values=data_merged,
            default_ylabel="Prob. occurrence",
            draw_marker=draw_marker,
            **kwargs,
        )

    def plot_mean_transition_times(
        self,
        prediction: Prediction | None = None,
        diff_dist: bool = True,
        **kwargs: Any,
    ) -> mplAxes:
        """
        Plot mean times until transitions occur.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        diff_dist
            Whether to plot paired transitions distance-specific or not.
        kwargs
            kwargs to fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        transition_df = self.simulation.transition_set.transition_df
        mean_transition_times = self.mean_transition_times
        if not diff_dist:
            transition_df, discarded_ids_by_retained_position, discarded_ids = (
                no_diff_dist(
                    transition_df=transition_df,
                    fluorophores=self.simulation.transition_set.single_states.keys(),
                )
            )
            collapsed_distributions = [
                distribution
                for i, distribution in enumerate(self.transition_time_distributions)
                if i not in discarded_ids
            ]
            for (
                retained_position,
                discarded_transition_ids,
            ) in discarded_ids_by_retained_position.items():
                for transition_id in discarded_transition_ids:
                    collapsed_distributions[retained_position] = np.concatenate(
                        (
                            collapsed_distributions[retained_position],
                            self.transition_time_distributions[transition_id],
                        )
                    )
            mean_transition_times = np.array(
                [
                    np.mean(distribution) if distribution.size > 0 else np.nan
                    for distribution in collapsed_distributions
                ]
            )
        legend_labels = [
            (
                name.rsplit(", dist:", maxsplit=1)[0]
                if parse_paired_transition_label(name) is not None and not diff_dist
                else name
            )
            for name in transition_df.index.get_level_values(0).unique()
        ]

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction uses a different TransitionSet object; verify that "
                    "states and transition ordering are compatible.",
                    stacklevel=2,
                )
            if prediction.paired_transitions:
                raise ValueError(
                    "predicted mean_transition_times not available if paired transitions "
                    "are possible."
                )
            predicted_means = prediction.mean_transition_times
            if predicted_means is None:
                raise ValueError("predicted mean transition times are unavailable.")
            # Prediction supports at most one distance for each fluorophore pair, so
            # distance-specific collapsing is only required for the simulation.
            if predicted_means.shape != mean_transition_times.shape:
                raise ValueError(
                    "prediction and simulation have incompatible transition dimensions."
                )
            draw_marker = [np.arange(transition_df.shape[0]), predicted_means]

        return plotting._plot_transition_bars(
            transition_df=transition_df,
            values=mean_transition_times,
            default_ylabel=r"$\tau$ (s)",
            draw_marker=draw_marker,
            legend_labels=legend_labels,
            **kwargs,
        )

    def plot_mean_lifetimes(
        self, prediction: Prediction | None = None, **kwargs: Any
    ) -> mplAxes:
        """
        Plot mean lifetimes of states.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """

        transition_set = self.simulation.transition_set
        data_merged = plotting._flatten_state_values(
            transition_set, self.mean_lifetimes
        )

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction uses a different TransitionSet object; verify that "
                    "states and transition ordering are compatible.",
                    stacklevel=2,
                )
            if prediction.paired_transitions:
                raise ValueError(
                    "predicted lifetime_distributions not available if paired transitions "
                    "are possible."
                )
            predicted_lifetimes = prediction.mean_lifetimes
            if predicted_lifetimes is None:
                raise ValueError("predicted mean lifetimes are unavailable.")
            predicted_lifetimes_merged = plotting._flatten_state_values(
                transition_set, predicted_lifetimes
            )
            if predicted_lifetimes_merged.shape != data_merged.shape:
                raise ValueError(
                    "prediction and simulation have incompatible state dimensions."
                )
            draw_marker = [
                np.arange(data_merged.size),
                predicted_lifetimes_merged,
            ]

        return plotting._plot_state_bars(
            transition_set=transition_set,
            values=data_merged,
            default_ylabel=r"$\tau$ (s)",
            draw_marker=draw_marker,
            full_xlim=True,
            **kwargs,
        )

    def plot_state_occupations(
        self, prediction: Prediction | None = None, **kwargs: Any
    ) -> mplAxes:
        """
        Plot the relative time spent in each state.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """

        transition_set = self.simulation.transition_set
        data_merged = plotting._flatten_state_values(
            transition_set, self.state_occupations
        )

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction uses a different TransitionSet object; verify that "
                    "states and transition ordering are compatible.",
                    stacklevel=2,
                )
            if prediction.paired_transitions:
                raise ValueError(
                    "predicted state_occupations not available if paired transitions "
                    "are possible."
                )
            predicted_occupations = prediction.state_occupations
            if predicted_occupations is None:
                raise ValueError("predicted state occupations are unavailable.")
            predicted_occupations_merged = plotting._flatten_state_values(
                transition_set, predicted_occupations
            )
            if predicted_occupations_merged.shape != data_merged.shape:
                raise ValueError(
                    "prediction and simulation have incompatible state dimensions."
                )
            draw_marker = [
                np.arange(data_merged.size),
                predicted_occupations_merged,
            ]

        return plotting._plot_state_bars(
            transition_set=transition_set,
            values=data_merged,
            default_ylabel="Prob. occupation",
            draw_marker=draw_marker,
            **kwargs,
        )

    def plot_lifetime_distributions(
        self,
        fluorophore: str,
        state_identity: int,
        prediction: Prediction | None = None,
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
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """

        kwargs.setdefault("type_", "hist")
        kwargs.setdefault("ylabel", "Prob. density")
        kwargs.setdefault("title", f"{fluorophore}")
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault(
            "xlabel",
            rf"{format_electronic_state(self.simulation.transition_set.states_by_value[state_identity].name)}"
            " duration (s)",
        )
        kwargs.setdefault("density", True)
        index = np.where(
            self.simulation.transition_set.single_states[fluorophore] == state_identity
        )[0][0]
        data = self.lifetime_distributions[fluorophore][index]
        plot_distribution = None
        plot_distribution_label = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction uses a different TransitionSet object; verify that "
                    "states and transition ordering are compatible.",
                    stacklevel=2,
                )
            if prediction.paired_transitions:
                raise ValueError(
                    "predicted lifetime_distributions not available if paired transitions "
                    "are possible."
                )
            predicted_distributions = prediction.lifetime_distributions
            if predicted_distributions is None:
                raise ValueError("predicted lifetime distributions are unavailable.")
            plot_distribution = predicted_distributions[fluorophore][index]
            plot_distribution_label = "Prediction"
            kwargs.setdefault("legend", True)

        ax = plotting.plot_data(
            data=data,
            plot_distribution=plot_distribution,
            plot_distribution_label=plot_distribution_label,
            **kwargs,
        )

        return ax

    def plot_transition_time_distributions(
        self,
        fluorophore: str,
        transition_id: int,
        prediction: Prediction | None = None,
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
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to fluopy.plotting.plot_data

        Returns
        -------
        matplotlib.axes.Axes
            The modified axis.
        """
        kwargs.setdefault("type_", "hist")
        kwargs.setdefault("ylabel", "PD")
        kwargs.setdefault(
            "title",
            rf"""$\tau$ of {fluorophore}
            {
                format_transition(
                    cast(
                        str,
                        self.simulation.transition_set.transition_df.loc[
                            (fluorophore, transition_id), "abbreviation"
                        ],
                    )
                )
            }""",
        )
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("xlabel", "time to transition [s]")
        kwargs.setdefault("density", True)
        data = self.transition_time_distributions[transition_id]
        plot_distribution = None
        plot_distribution_label = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction uses a different TransitionSet object; verify that "
                    "states and transition ordering are compatible.",
                    stacklevel=2,
                )
            if prediction.paired_transitions:
                raise ValueError(
                    "predicted transition_time_distributions not available if paired "
                    "transitions are possible."
                )
            predicted_distributions = prediction.transition_time_distributions
            if predicted_distributions is None:
                raise ValueError(
                    "predicted transition-time distributions are unavailable."
                )
            plot_distribution = predicted_distributions[transition_id]
            plot_distribution_label = "pred"
            kwargs.setdefault("label", "sim")
            kwargs.setdefault("legend", True)
        ax = plotting.plot_data(
            data=data,
            plot_distribution=plot_distribution,
            plot_distribution_label=plot_distribution_label,
            **kwargs,
        )

        return ax


def no_diff_dist(transition_df: pd.DataFrame, fluorophores: Iterable[str]) -> tuple[
    pd.DataFrame,
    dict[int, pd.Index[Any]],
    npt.NDArray[np.int64],
]:
    """
    Get a transition_df containing one distance for each paired-transition type.

    Distance-specific transitions are matched by donor, acceptor, abbreviation, state
    change, photon emission and mechanism.

    Parameters
    ----------
    transition_df
        Dataframe of all given transitions with non-zero rate containing their id as
        second level index and their other attributes as columns. Name of fluorophores
        as first level index.
    fluorophores
        Names of fluorophores.

    Returns
    -------
    collapsed_transition_df : pd.DataFrame
        Transition dataframe containing one distance for each paired-transition group.
    discarded_ids_by_retained_position : dict[int, pd.Index[Any]]
        Positions of retained transitions as keys and the corresponding discarded
        distance-specific transition IDs as values.
    discarded_ids : npt.NDArray[np.int64]
        Flattened array of discarded distance-specific transition IDs.
    """
    transition_ids = transition_df.index.get_level_values(1)
    fluorophore_names = set(fluorophores)
    retained_label_by_pair: dict[tuple[str, str], str] = {}
    retained_id_by_transition: dict[tuple[object, ...], int] = {}
    retained_label_by_id: dict[int, str] = {}
    discarded_ids_by_retained_id: dict[int, list[int]] = {}

    for index, transition in transition_df.iterrows():
        if not isinstance(index, tuple) or len(index) != 2:
            raise TypeError("transition DataFrame must have a two-level index.")
        group_label_raw, transition_id_raw = index
        if not isinstance(transition_id_raw, int):
            raise TypeError("transition identity must be an integer.")
        group_label = str(group_label_raw)
        transition_id = transition_id_raw
        paired_transition = parse_paired_transition_label(group_label)
        if paired_transition is None:
            retained_label_by_id[transition_id] = group_label
            continue
        donor, acceptor, _ = paired_transition
        if donor not in fluorophore_names:
            retained_label_by_id[transition_id] = group_label
            continue

        pair = donor, acceptor
        retained_label = retained_label_by_pair.setdefault(pair, group_label)
        transition_key = (
            donor,
            acceptor,
            transition["abbreviation"],
            transition["initial_state"],
            transition["final_state"],
            transition["photon"],
            transition["mechanism"],
        )
        if transition_key not in retained_id_by_transition:
            retained_id_by_transition[transition_key] = transition_id
            retained_label_by_id[transition_id] = retained_label
            continue

        retained_id = retained_id_by_transition[transition_key]
        discarded_ids_by_retained_id.setdefault(retained_id, []).append(transition_id)

    discarded_ids = np.asarray(
        [
            transition_id
            for ids in discarded_ids_by_retained_id.values()
            for transition_id in ids
        ],
        dtype=np.int64,
    )
    retained_mask = ~transition_ids.isin(discarded_ids)
    collapsed_transition_df = transition_df[retained_mask].copy()
    retained_transition_ids = collapsed_transition_df.index.get_level_values(1)
    retained_group_labels = [
        retained_label_by_id[cast(int, transition_id)]
        for transition_id in retained_transition_ids
    ]
    collapsed_transition_df.index = pd.MultiIndex.from_arrays(
        [
            retained_group_labels,
            range(len(collapsed_transition_df)),
        ]
    )
    retained_position_by_id = {
        cast(int, transition_id): position
        for position, transition_id in enumerate(retained_transition_ids)
    }
    discarded_ids_by_retained_position: dict[int, pd.Index[Any]] = {}
    for retained_id, discarded_transition_ids in discarded_ids_by_retained_id.items():
        retained_position = retained_position_by_id[retained_id]
        discarded_ids_by_retained_position[retained_position] = pd.Index(
            discarded_transition_ids,
            dtype="int64",
            name=transition_ids.name,
        )

    return (
        collapsed_transition_df,
        discarded_ids_by_retained_position,
        discarded_ids,
    )


def get_absorbing_state_times(simulation: Simulation) -> npt.NDArray[np.float64]:
    """
    Get the first time each fluorophore reaches an individually absorbing state.

    If a fluorophore does not reach an absorbing state, its entry is np.nan. The times
    are sorted, with np.nan entries at the end. Photobleaching is one possible process
    that can be represented by an absorbing state.

    Parameters
    ----------
    simulation
        Container for simulation-associated attributes

    Returns
    -------
    npt.NDArray[np.float64]
        First absorbing-state times of shape (n_fluorophores,).
    """
    state_series = simulation.state_series
    time_series = simulation.time_series
    if state_series is None or time_series is None:
        raise ValueError("absorbing-state times require a completed simulation.")
    absorbing_states = simulation.transition_set.absorbing_states
    absorbing_state_times: list[float] = []
    for identity, fluorophore_states in enumerate(state_series):
        absorbing = absorbing_states.get(identity, np.array([], dtype=np.int64))
        occurrences = np.flatnonzero(np.isin(fluorophore_states, absorbing))
        time = time_series[occurrences[0]] if occurrences.size else np.nan
        absorbing_state_times.append(time)

    return np.sort(np.asarray(absorbing_state_times, dtype=np.float64))


def get_delta_absorbing_state_times(
    absorbing_state_times: npt.ArrayLike,
) -> list[npt.NDArray[np.float64]]:
    """
    Get elapsed times between successive entries into absorbing states.

    Parameters
    ----------
    absorbing_state_times
        First absorbing-state times. Each run is a row and each fluorophore is a
        column. Each row is sorted, with np.nan entries at the end.

    Returns
    -------
    list[npt.NDArray[np.float64]]
        Elapsed times from the start to the first absorbing-state entry and between
        subsequent entries. Each array contains one event order across all runs.
    """
    absorbing_state_times_array = np.asarray(absorbing_state_times, dtype=np.float64)
    delta_absorbing_state_times_all: list[npt.NDArray[np.float64]] = []
    previous_times = np.zeros(absorbing_state_times_array.shape[0], dtype=np.float64)
    for fluorophore in range(absorbing_state_times_array.shape[1]):
        times_for_event_order = absorbing_state_times_array[:, fluorophore]
        delta_absorbing_state_times = times_for_event_order - previous_times
        delta_absorbing_state_times = delta_absorbing_state_times[
            ~np.isnan(delta_absorbing_state_times)
        ]
        delta_absorbing_state_times_all.append(delta_absorbing_state_times)
        previous_times = times_for_event_order

    return delta_absorbing_state_times_all
