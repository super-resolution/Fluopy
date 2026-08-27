"""
Analysis of a photophysical simulation.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

import matplotlib as mpl
import numpy as np
import numpy.typing as npt
import pandas as pd

from . import figure as fi
from .miscellaneous import format_electronic_state, format_transition

if TYPE_CHECKING:
    from .prediction import Prediction
    from .simulation import Simulation


__all__: list[str] = ["Analysis"]

logger = logging.getLogger(__name__)

_ENERGY_TRANSFER_LABEL = re.compile(
    r"D:\s*([^,]+),\s*A:\s*([^,]+),\s*dist:\s*(\d+(?:\.\d+)?)\s*"
)


class Analysis:
    """
    Container of simulation-derived statistical attributes and methods.

    Attributes
    ----------
    simulation : fluopy.simulation.Simulation
        Container for simulation-associated attributes.
    frequency_transitions : npt.NDArray[np.float64]
        Relative number of simulated transition occurrences, normalized separately for
        each fluorophore.
    frequency_states : dict[str, npt.NDArray[np.float64]]
        Relative simulated number of visits to each state, normalized separately for
        each fluorophore.
    transition_time_distributions : Collection
        Contains 1-D array_like for each transition (time until the transition).
    lifetime_distributions : dict
        Name of fluorophores as keys and collections of their state's simulated
        lifetimes (1-D array_like) as values.
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
                "absolute state and transition frequency of 1, but the lifetime is nan "
                "and the state occupation 0.",
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
        Check whether fluorophores reached Markovian absorbing states.

        Returns
        -------
        is_abs : bool
            Whether at least one of the fluorophores has reached a Markovian absorbing
            state.
        """
        from .transitions import SingleState

        initial_states = self.simulation.transition_set.transition_df[
            "initial_state"
        ].apply(lambda state: (state.value if isinstance(state, SingleState) else None))
        initial_state_values = initial_states.dropna().astype(int).to_numpy()
        absorbing_states: dict[str, list[int]] = {}
        is_abs = False
        for (
            fluorophore,
            single_states,
        ) in self.simulation.transition_set.single_states.items():
            for single_state in single_states:
                if single_state not in initial_state_values:
                    if fluorophore in absorbing_states:
                        absorbing_states[fluorophore] += [single_state]
                    else:
                        absorbing_states[fluorophore] = [single_state]
        for i, state_series in enumerate(self.state_series):
            fluorophore_obj = (
                self.simulation.transition_set.fluorophore_system.fluorophores[i]
            )
            last_state = state_series[-1]
            if fluorophore_obj.name in absorbing_states:
                if last_state in absorbing_states[fluorophore_obj.name]:
                    is_abs = True
                    print(
                        f"fluorophore {i} has reached the Markovian absorbing state "
                        f"{self.simulation.transition_set.states_by_value[last_state].name}"
                    )

        return is_abs

    def get_transition_occurrences(self) -> npt.NDArray[np.float64]:
        """
        Get the relative frequencies of simulated transition occurrences.

        Returns
        -------
        frequency_transitions : npt.NDArray[np.float64]
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

        grouper: dict[str, list[int]] = {}
        for (
            fluorophore_comb_raw,
            group,
        ) in self.simulation.transition_set.transition_df.groupby(level=0, sort=False):
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

    def get_state_occurrences(self) -> dict[str, npt.NDArray[np.float64]]:
        """
        Get the relative frequencies of simulated state visits.

        Returns
        -------
        frequency_states : dict[str, npt.NDArray[np.float64]]
            Relative simulated number of visits to each state, normalized separately
            for each fluorophore. Frequencies remain 0 if no state visits were counted.
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
        Note: if transition of interest is energy transfer, the time to transition is
        only collected from the donor's point of view.

        Returns
        -------
        transition_time_distributions : list[npt.NDArray[np.float64]]
            Contains 1-D array_like for each transition (time until the transition).
        lifetime_distributions : dict[str, npt.NDArray[np.float64]]
            Name of fluorophores as keys and collections of their state's simulated
            lifetimes (1-D array_like) as values.
        """
        single_states = self.simulation.transition_set.single_states
        df = self.simulation.transition_set.combined_state_transitions_df
        lifetime_parts: dict[str, list[list[npt.NDArray[np.float64]]]] = {
            key: [[] for _ in range(len(value))] for key, value in single_states.items()
        }
        transition_time_parts: list[list[npt.NDArray[np.float64]]] = [
            [] for _ in range(self.simulation.transition_set.transition_df.shape[0])
        ]
        transition_ids = df["transition_id"].to_numpy(dtype=np.int64)

        for i, state_series_fluorophore in enumerate(self.state_series):
            fluorophore = (
                self.simulation.transition_set.fluorophore_system.fluorophores[i].name
            )
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            changed = changes_at + 1
            initial_single_states = state_series_fluorophore[changes_at]
            total_times = self.time_series[changed]
            time_intervals = np.diff(total_times)
            time_intervals = np.insert(arr=time_intervals, obj=0, values=total_times[0])
            for j, state in enumerate(single_states[fluorophore]):
                time_intervals_state = time_intervals[
                    np.where(initial_single_states == state)
                ]
                lifetime_parts[fluorophore][j].append(time_intervals_state)

            transitions_fluorophore = self.transition_series[changes_at]
            transition_ids_fluorophore = transition_ids[transitions_fluorophore]
            for h, j in self.simulation.transition_set.transition_df.index:
                occurrence_mask = transition_ids_fluorophore == j
                if _ENERGY_TRANSFER_LABEL.fullmatch(h) is not None:
                    source_donor = self.simulation.transition_set.transition_df.loc[
                        (h, j), "initial_state"
                    ].donor.value
                    occurrence_mask &= initial_single_states == source_donor

                transition_time_parts[j].append(time_intervals[occurrence_mask])

        lifetime_distributions = {
            fluorophore: [np.concatenate(parts) for parts in state_parts]
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
        state_occupations: dict[str, npt.NDArray[np.float64]] = {}
        for fluorophore, distributions in self.lifetime_distributions.items():
            mean_lifetimes[fluorophore] = np.array(
                [
                    np.mean(distr) if distr.size != 0 else np.nan
                    for distr in distributions
                ]
            )
            state_occupations[fluorophore] = np.multiply(
                self.frequency_states[fluorophore],
                mean_lifetimes[fluorophore],
                where=~np.isnan(mean_lifetimes[fluorophore]),
                out=np.zeros(self.frequency_states[fluorophore].size),
            )
            total_occupation = state_occupations[fluorophore].sum()
            if total_occupation > 0:
                state_occupations[fluorophore] /= total_occupation

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
        fluorescence_lifetimes : npt.NDArray[np.float64]
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
        exp_fluorescence_lifetimes : npt.NDArray[np.float64]
            The fluorescence lifetimes (photon emssion) of the specified fluorophore.
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
    ) -> fi.AxesArray:
        """
        Plot relative frequencies of simulated transition occurrences.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        diff_dist
            Whether to plot energy transfers distance-specific or not.
        kwargs
            kwargs for fluopy.figure.universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        df = self.simulation.transition_set.transition_df
        dat = self.frequency_transitions
        if not diff_dist:
            df, dict2, dict_values = no_diff_dist(
                transition_df=df,
                fluorophores=self.simulation.transition_set.single_states.keys(),
            )
            data2 = np.delete(dat, dict_values)
            for key, values in dict2.items():
                data2[key] += np.sum(dat[values])
            dat = data2
        data = [np.arange(df.shape[0]), dat]
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
                mpl.patches.Patch(
                    color=colormap(i),
                    label=(
                        name.rsplit(", dist:", maxsplit=1)[0]
                        if _ENERGY_TRANSFER_LABEL.fullmatch(name) is not None
                        and not diff_dist
                        else name
                    ),
                )
                for i, name in enumerate(df.index.get_level_values(0).unique())
            ],
        )

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction is based on different TransitionSet than simulation.",
                    stacklevel=2,
                )
            draw_marker = [np.arange(df.shape[0]), prediction.frequency_transitions]

        axes = fi.universal_figure(data=data, draw_marker=draw_marker, **kwargs)

        return axes

    def plot_frequency_states(
        self, prediction: Prediction | None = None, **kwargs: Any
    ) -> fi.AxesArray:
        """
        Plot relative frequencies of simulated state visits.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """

        single_states = self.simulation.transition_set.single_states
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
                        self.simulation.transition_set.states_by_value[identity].name
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

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction is based on different TransitionSet than simulation.",
                    stacklevel=2,
                )
            draw_marker = [
                np.arange(xticks),
                np.concatenate(
                    [
                        prediction.frequency_states[fluorophore]
                        for fluorophore in single_states
                    ]
                ),
            ]

        axes = fi.universal_figure(data=data, draw_marker=draw_marker, **kwargs)

        return axes

    def plot_mean_transition_times(
        self,
        prediction: Prediction | None = None,
        diff_dist: bool = True,
        **kwargs: Any,
    ) -> fi.AxesArray:
        """
        Plot mean times until transitions occur.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        diff_dist
            Whether to plot energy transfers distance-specific or not.
        kwargs
            kwargs to universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        df = self.simulation.transition_set.transition_df
        dat = self.mean_transition_times
        if not diff_dist:
            df, dict2, dict_values = no_diff_dist(
                transition_df=df,
                fluorophores=self.simulation.transition_set.single_states.keys(),
            )
            data2 = [
                val
                for i, val in enumerate(self.transition_time_distributions)
                if i not in dict_values
            ]
            for key, values in dict2.items():
                for value in values:
                    data2[key] = np.concatenate(
                        (data2[key], self.transition_time_distributions[value])
                    )
            dat = np.array([(np.mean(d) if d.size > 0 else np.nan) for d in data2])
        data = [np.arange(df.shape[0]), dat]
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
                mpl.patches.Patch(
                    color=colormap(i),
                    label=(
                        name.rsplit(", dist:", maxsplit=1)[0]
                        if _ENERGY_TRANSFER_LABEL.fullmatch(name) is not None
                        and not diff_dist
                        else name
                    ),
                )
                for i, name in enumerate(df.index.get_level_values(0).unique())
            ],
        )

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction is based on different TransitionSet than simulation.",
                    stacklevel=2,
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted mean_transition_times not available if energy transfer "
                    "possible."
                )
            predicted_means = prediction.mean_transition_times
            if predicted_means is None:
                raise ValueError("predicted mean transition times are unavailable.")
            draw_marker = [np.arange(df.shape[0]), predicted_means]

        axes = fi.universal_figure(data=data, draw_marker=draw_marker, **kwargs)

        return axes

    def plot_mean_lifetimes(
        self, prediction: Prediction | None = None, **kwargs: Any
    ) -> fi.AxesArray:
        """
        Plot mean lifetimes of states.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """

        single_states = self.simulation.transition_set.single_states
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
            data_parts.append(self.mean_lifetimes[fluorophore])
            labels.extend(
                [
                    format_electronic_state(
                        self.simulation.transition_set.states_by_value[identity].name
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

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction is based on different TransitionSet than simulation.",
                    stacklevel=2,
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted lifetime_distributions not available if energy "
                    "transfers possible."
                )
            predicted_lifetimes = prediction.mean_lifetimes
            if predicted_lifetimes is None:
                raise ValueError("predicted mean lifetimes are unavailable.")
            draw_marker = [
                np.arange(xticks),
                np.concatenate(
                    [predicted_lifetimes[fluorophore] for fluorophore in single_states]
                ),
            ]

        axes = fi.universal_figure(data=data, draw_marker=draw_marker, **kwargs)

        return axes

    def plot_state_occupations(
        self, prediction: Prediction | None = None, **kwargs: Any
    ) -> fi.AxesArray:
        """
        Plot the relative time spent in each state.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """

        single_states = self.simulation.transition_set.single_states
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
            data_parts.append(self.state_occupations[fluorophore])
            labels.extend(
                [
                    format_electronic_state(
                        self.simulation.transition_set.states_by_value[identity].name
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

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                logger.warning(
                    "prediction is based on different TransitionSet than simulation.",
                    stacklevel=2,
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted state_occupations not available if energy transfers "
                    "possible."
                )
            predicted_occupations = prediction.state_occupations
            if predicted_occupations is None:
                raise ValueError("predicted state occupations are unavailable.")
            draw_marker = [
                np.arange(xticks),
                np.concatenate(
                    [
                        predicted_occupations[fluorophore]
                        for fluorophore in single_states
                    ]
                ),
            ]

        axes = fi.universal_figure(data=data, draw_marker=draw_marker, **kwargs)

        return axes

    def plot_lifetime_distributions(
        self,
        fluorophore: str,
        state_identity: int,
        prediction: Prediction | None = None,
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
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
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
                    "prediction is based on different TransitionSet than simulation.",
                    stacklevel=2,
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted lifetime_distributions not available if energy transfer "
                    "possible."
                )
            predicted_distributions = prediction.lifetime_distributions
            if predicted_distributions is None:
                raise ValueError("predicted lifetime distributions are unavailable.")
            plot_distribution = predicted_distributions[fluorophore][index]
            plot_distribution_label = "Prediction"
            kwargs.setdefault("legend", True)

        axes = fi.universal_figure(
            data=data,
            plot_distribution=plot_distribution,
            plot_distribution_label=plot_distribution_label,
            **kwargs,
        )

        return axes

    def plot_transition_time_distributions(
        self,
        fluorophore: str,
        transition_id: int,
        prediction: Prediction | None = None,
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
        prediction
            Container of mathematically derived statistical attributes and methods.
        kwargs
            kwargs to universal_figure

        Returns
        -------
        npt.NDArray[mplAxes]
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        kwargs.setdefault("type_", "hist")
        kwargs.setdefault("ylabel", "PD")
        kwargs.setdefault(
            "title",
            rf"""$\tau$ of {fluorophore}
            {format_transition(cast(str, self.simulation.transition_set.transition_df.loc[(fluorophore,
                                                               transition_id),
                                                               "abbreviation"]))}""",
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
                    "prediction is based on different TransitionSet than simulation.",
                    stacklevel=2,
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted transition_time_distributions not available if energy "
                    "transfer possible."
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
        axes = fi.universal_figure(
            data=data,
            plot_distribution=plot_distribution,
            plot_distribution_label=plot_distribution_label,
            **kwargs,
        )

        return axes


def no_diff_dist(transition_df: pd.DataFrame, fluorophores: Iterable[str]) -> tuple[
    pd.DataFrame,
    dict[int, pd.Index[Any]],
    npt.NDArray[np.int64],
]:
    """
    Get a transition_df which only contains one distance for each type of energy
    transfer.

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
    df2 : pd.DataFrame
        Altered transition_df.
    dict2 : dict[str, Any]
        New indices of type of energy transfer as keys and the old indices of distance-
        dependent duplices as values.
    dict2_vals : npt.NDArray[np.float64]
        Flattened array of the values of dict2.
    """
    df2 = transition_df.copy()
    level_0 = df2.index.get_level_values(0)
    level_1 = df2.index.get_level_values(1)
    level_0_unique = level_0.unique()
    fluorophore_names = set(fluorophores)
    labels_by_pair: dict[tuple[str, str], list[str]] = {}
    for name in level_0_unique:
        match = _ENERGY_TRANSFER_LABEL.fullmatch(name)
        if match is None:
            continue
        donor, acceptor, _ = match.groups()
        if donor in fluorophore_names:
            labels_by_pair.setdefault((donor, acceptor), []).append(name)

    dict1: dict[str, pd.Index[Any]] = {}
    discard = []
    for different_names in labels_by_pair.values():
        to_keep = different_names[0]
        to_discard = different_names[1:]
        if not to_discard:
            continue
        corresponding_level1 = level_1[level_0.isin(to_discard)]
        dict1[to_keep] = corresponding_level1
        discard.extend(to_discard)

    df2 = df2[~df2.index.get_level_values(0).isin(discard)]
    df2.index = pd.MultiIndex.from_arrays(
        [df2.index.get_level_values(0), range(len(df2))]
    )
    dict2: dict[int, pd.Index[Any]] = {}
    for key, values in dict1.items():
        df_subset = df2.loc[key]
        size = df_subset.shape[0]
        for i in range(size):
            new_index = cast(int, df_subset.index[i])
            dict2[new_index] = values[i::size]
    dict2_vals = np.concatenate(
        [values.to_numpy(dtype=np.int64) for values in dict2.values()]
    )

    return df2, dict2, dict2_vals
