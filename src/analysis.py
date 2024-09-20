"""
Module analysis
"""

import re
import warnings
import numpy as np
import matplotlib as mpl
import src.figure as fi


class Analysis:
    """
    Container of simulation-dervied statistical attributes and methods.

    Attributes
    ----------
    simulation : src.simulation.Simulation
        Container for simulation-associated attributes.
    frequency_transitions : 1-D array_like
        Simulated relative frequencies of each transition.
    frequency_states : dict
        Name of fluorophores as keys and their state's simulated relative frequencies
        (array) as values.
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
    state_occupations : dict
        Name of fluorophores as keys and their state's simulated probability of being
        occupied at any given point in time (array) as values.
    """

    def __init__(self, simulation):
        """
        Parameters
        ----------
        simulation : src.simulation.Simulation
            Container for simulation-associated attributes.
        """
        if simulation.transition_series is None:
            raise ValueError("analysis not available if simulation has not been run.")

        self.simulation = simulation

        absorbing = self.check_absorbing()
        if absorbing:
            warnings.warn(
                "if a fluorophore reaches its individual absorbing state, it has an "
                "absolute state and transition frequency of 1, but the lifetime is nan "
                "and the state occupation 0."
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

    def check_absorbing(self):
        """
        Check whether fluorophores reached Markovian absorbing states.

        Returns
        -------
        absorbing : bool
            Whether at least one of the fluorophores has reached a Markovian absorbing
            state.
        """
        from src.transitions import SingleState

        initial_states = self.simulation.transition_set.transition_df[
            "initial_state"
        ].apply(lambda x: x.value if not isinstance(x.value, list) else None)
        initial_states = initial_states.dropna().astype(int).values
        absorbing_states = {}
        absorbing = False
        for (
            fluorophore,
            single_states,
        ) in self.simulation.transition_set.single_states.items():
            for single_state in single_states:
                if single_state not in initial_states:
                    if fluorophore in absorbing_states:
                        absorbing_states[fluorophore] += [single_state]
                    else:
                        absorbing_states[fluorophore] = [single_state]
        for i, state_series in enumerate(self.simulation.state_series):
            fluorophore = (
                self.simulation.transition_set.fluorophore_system.fluorophores[i]
            )
            last_state = state_series[-1]
            if fluorophore.name in absorbing_states:
                if last_state in absorbing_states[fluorophore.name]:
                    absorbing = True
                    print(
                        f"fluorophore {i} has reached the Markovian absorbing state "
                        f"{SingleState(last_state)}"
                    )

        return absorbing

    def get_transition_occurrences(self):
        """
        Get the relative frequencies of transitions.

        Returns
        -------
        frequency_transitions : 1-D array_like
            Simulated relative frequencies of each transition.
        """
        all_transition_occurrences = np.zeros(
            shape=self.simulation.transition_set.transition_df.shape[0], dtype=np.int64
        )
        df = self.simulation.transition_set.combined_state_transitions_df
        for _, i in self.simulation.transition_set.transition_df.index:
            indices = df.index[df["transition_id"] == i].tolist()
            transition_occurrences = np.in1d(
                self.simulation.transition_series, indices
            ).nonzero()[0]
            all_transition_occurrences[i] = transition_occurrences.size

        frequency_transitions = all_transition_occurrences.astype(np.float64)

        grouper = {}
        for (
            fluorophore_comb,
            group,
        ) in self.simulation.transition_set.transition_df.groupby(level=0, sort=False):
            if "dist" in fluorophore_comb:
                pattern = r"D:\s*([^,]+),\s*A:\s*([^,]+),\s*dist:\s*([\d.]+)"
                match = re.match(pattern, fluorophore_comb)
                d, a, dist = match.group(1), match.group(2), match.group(3)
            else:
                d = fluorophore_comb
            if d in grouper:
                grouper[d] += group.index.get_level_values(1).tolist()
            else:
                grouper[d] = group.index.get_level_values(1).tolist()

        for fluorophore, indices in grouper.items():
            frequency_transitions[indices] /= np.sum(frequency_transitions[indices])

        return frequency_transitions

    def get_state_occurrences(self):
        """
        Get the relative frequencies of states.

        Returns
        -------
        frequency_states : dict
            Name of fluorophores as keys and their state's simulated relative
            frequencies (array) as values.
        """
        single_states = self.simulation.transition_set.single_states
        occurrences_states = {
            key: np.zeros(len(value)) for key, value in single_states.items()
        }
        for i, state_series_fluorophore in enumerate(self.simulation.state_series):
            fluorophore = (
                self.simulation.transition_set.fluorophore_system.fluorophores[i].name
            )
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            last_state = changes_at[-1] + 1
            changes_at_and_last = np.append(changes_at, last_state)
            states = state_series_fluorophore[changes_at_and_last]
            state_ids, state_counts = np.unique(states, return_counts=True)
            corresponding_state_indices = np.in1d(
                single_states[fluorophore], state_ids
            ).nonzero()[0]
            keep_indices = np.in1d(state_ids, single_states[fluorophore]).nonzero()[0]
            state_counts = state_counts[keep_indices]
            occurrences_states[fluorophore][corresponding_state_indices] += state_counts

        frequency_states = {
            key: array / np.sum(array) for key, array in occurrences_states.items()
        }

        return frequency_states

    def get_lifetimes(self):
        """
        Get the lifetime distributions of states and the time until occurrence
        distributions of transitions.
        Note: if transition of interest is energy transfer, the time to transition is
        only collected from the donor's point of view.

        Returns
        -------
        transition_time_distributions : Collection
            Contains 1-D array_like for each transition (time until the transition).
        lifetime_distributions : dict
            Name of fluorophores as keys and collections of their state's simulated
            lifetimes (1-D array_like) as values.
        """
        single_states = self.simulation.transition_set.single_states
        df = self.simulation.transition_set.combined_state_transitions_df
        lifetime_distributions = {
            key: [np.array([]) for _ in range(len(value))]
            for key, value in single_states.items()
        }

        transition_time_distributions = [
            np.array([])
            for _ in range(self.simulation.transition_set.transition_df.shape[0])
        ]

        for i, state_series_fluorophore in enumerate(self.simulation.state_series):
            fluorophore = (
                self.simulation.transition_set.fluorophore_system.fluorophores[i].name
            )
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            changed = changes_at + 1
            initial_single_states = state_series_fluorophore[changes_at]
            total_times = self.simulation.time_series[changed]
            time_intervals = np.diff(total_times)
            time_intervals = np.insert(time_intervals, 0, total_times[0])
            for j, state in enumerate(single_states[fluorophore]):
                time_intervals_state = time_intervals[
                    np.where(initial_single_states == state)
                ]
                lifetime_distributions[fluorophore][j] = np.concatenate(
                    [lifetime_distributions[fluorophore][j], time_intervals_state]
                )

            transitions_fluorophore = self.simulation.transition_series[changes_at]
            for h, j in self.simulation.transition_set.transition_df.index:
                indices = df.index[df["transition_id"] == j].tolist()
                transition_occurrences = np.in1d(
                    transitions_fluorophore, indices
                ).nonzero()[0]
                if "dist" in h:
                    source_donor = self.simulation.transition_set.transition_df.loc[
                        (h, j), "initial_state"
                    ].donor.value
                    donor_indices = np.where(initial_single_states == source_donor)[0]
                    transition_occurrences = transition_occurrences[
                        np.in1d(transition_occurrences, donor_indices)
                    ]

                time_intervals_transition = time_intervals[transition_occurrences]
                transition_time_distributions[j] = np.concatenate(
                    [transition_time_distributions[j], time_intervals_transition]
                )

        return transition_time_distributions, lifetime_distributions

    def infer_stats(self):
        """
        Infers statistics of states based on lifetime distributions and frequencies.

        Returns
        -------
        mean_lifetimes : dict
            Name of fluorophores as keys and their state's simulated lifetime means
            (array) as values.
        state_occupations : dict
            Name of fluorophores as keys and their state's simulated probability of
            being occupied at any given point in time (array) as values.
        """
        mean_lifetimes = {}
        state_occupations = {}
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
            state_occupations[fluorophore] /= state_occupations[fluorophore].sum()

        return mean_lifetimes, state_occupations

    def get_fluorescence_lifetimes(self, fluorophore=None):
        """
        Get the fluorescence lifetime (i.e., S1 lifetime) of the specified fluorophore.

        Parameters
        ----------
        fluorophore : str, optional
            The name of the fluorophore whose fluorescence lifetime is to be returned.

        Returns
        -------
        fluorescence_lifetimes : np.ndarray
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

        return fluorescence_lifetimes

    def get_emitting_transition_lifetimes(self, fluorophore=None):
        """
        Get the lifetimes of the emitting transitions (i.e., S1 deexcitation via photon
        emission) of the specified fluorophore.

        Parameters
        ----------
        fluorophore : str, optional
            The name of the fluorophore whose fluorescence lifetime is to be returned.

        Returns
        -------
        exp_fluorescence_lifetimes : np.ndarray
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
        exp_fluorescence_lifetimes = [
            self.transition_time_distributions[emitting_transition_f]
            for emitting_transition_f in emitting_transitions_f
        ]
        exp_fluorescence_lifetimes = np.concatenate(exp_fluorescence_lifetimes)

        return exp_fluorescence_lifetimes

    def plot_frequency_transitions(self, prediction=None, **kwargs):
        """
        Plot frequencies of transitions.

        Parameters
        ----------
        prediction : src.prediction.Prediction
            Container of mathematically derived statistical attributes and methods.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        df = self.simulation.transition_set.transition_df
        data = [np.arange(df.shape[0]), self.frequency_transitions]
        kwargs.setdefault("type_", "bar")
        kwargs.setdefault("xlabel", None)
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("xticks", range(df.shape[0]))
        kwargs.setdefault("xticklabels", dict(labels=df["abbreviation"], rotation=70))
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
        kwargs.setdefault("ylabel", "PR")
        kwargs.setdefault("title", "frequency transitions")
        kwargs.setdefault("legend", True)
        kwargs.setdefault(
            "legendhandles",
            [
                mpl.patches.Patch(color=colormap(i), label=name)
                for i, name in enumerate(df.index.get_level_values(0).unique())
            ],
        )

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                raise ValueError(
                    "prediction is based on different TransitionSet than simulation."
                )
            draw_marker = [np.arange(df.shape[0]), prediction.frequency_transitions]

        axes = fi.universal_figure(data=data, draw_marker=draw_marker, **kwargs)

        return axes

    def plot_frequency_states(self, prediction=None, **kwargs):
        """
        Plot frequencies of states.

        Parameters
        ----------
        prediction : src.prediction.Prediction
            Container of mathematically derived statistical attributes and methods.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        from src.transitions import SingleState

        single_states = self.simulation.transition_set.single_states
        colormap = mpl.colors.ListedColormap(
            [
                mpl.colormaps["Spectral"](value)
                for value in np.linspace(0, 1, len(single_states))
            ]
        )
        colors, patches, xticks, data_merged, labels = [], [], 0, [], []
        for i, (fluorophore, states) in enumerate(single_states.items()):
            colors.extend([colormap(i) for _ in range(states.size)])
            patches.append(mpl.patches.Patch(color=colormap(i), label=fluorophore))
            xticks += states.size
            data_merged.append(self.frequency_states[fluorophore])
            labels.extend([SingleState(identity).name for identity in states])
        data_merged = np.concatenate(data_merged)
        data = [np.arange(xticks), data_merged]
        kwargs.setdefault("type_", "bar")
        kwargs.setdefault("xlabel", None)
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("xticks", range(xticks))
        kwargs.setdefault("xticklabels", dict(labels=labels, rotation=70))
        kwargs.setdefault("ylabel", "PR")
        kwargs.setdefault("title", "frequency states")
        kwargs.setdefault("color", colors)
        kwargs.setdefault("legend", True)
        kwargs.setdefault("legendhandles", patches)

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                raise ValueError(
                    "prediction is based on different TransitionSet than simulation."
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

    def plot_mean_transition_times(self, prediction=None, **kwargs):
        """
        Plot mean times until transitions occur.

        Parameters
        ----------
        prediction : src.prediction.Prediction
            Container of mathematically derived statistical attributes and methods.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        df = self.simulation.transition_set.transition_df
        data = [np.arange(df.shape[0]), self.mean_transition_times]
        kwargs.setdefault("type_", "bar")
        kwargs.setdefault("xlabel", None)
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("xticks", range(df.shape[0]))
        kwargs.setdefault("xticklabels", dict(labels=df["abbreviation"], rotation=70))
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
        kwargs.setdefault("ylabel", "mean [s]")
        kwargs.setdefault("title", "time to transition")
        kwargs.setdefault("legend", True)
        kwargs.setdefault(
            "legendhandles",
            [
                mpl.patches.Patch(color=colormap(i), label=name)
                for i, name in enumerate(df.index.get_level_values(0).unique())
            ],
        )

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                raise ValueError(
                    "prediction is based on different TransitionSet than simulation."
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted mean_transition_times not available if energy transfer "
                    "possible."
                )
            draw_marker = [np.arange(df.shape[0]), prediction.mean_transition_times]

        axes = fi.universal_figure(data=data, draw_marker=draw_marker, **kwargs)

        return axes

    def plot_mean_lifetimes(self, prediction=None, **kwargs):
        """
        Plot mean lifetimes of states.

        Parameters
        ----------
        prediction : src.prediction.Prediction
            Container of mathematically derived statistical attributes and methods.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        from src.transitions import SingleState

        single_states = self.simulation.transition_set.single_states
        colormap = mpl.colors.ListedColormap(
            [
                mpl.colormaps["Spectral"](value)
                for value in np.linspace(0, 1, len(single_states))
            ]
        )
        colors, patches, xticks, data_merged, labels = [], [], 0, [], []
        for i, (fluorophore, states) in enumerate(single_states.items()):
            colors.extend([colormap(i) for _ in range(states.size)])
            patches.append(mpl.patches.Patch(color=colormap(i), label=fluorophore))
            xticks += states.size
            data_merged.append(self.mean_lifetimes[fluorophore])
            labels.extend([SingleState(identity).name for identity in states])
        data_merged = np.concatenate(data_merged)
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
        kwargs.setdefault("ylabel", "mean [s]")
        kwargs.setdefault("title", "lifetimes")

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                raise ValueError(
                    "prediction is based on different TransitionSet than simulation."
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted lifetime_distributions not available if energy "
                    "transfers possible."
                )
            draw_marker = [
                np.arange(xticks),
                np.concatenate(
                    [
                        prediction.mean_lifetimes[fluorophore]
                        for fluorophore in single_states
                    ]
                ),
            ]

        axes = fi.universal_figure(data=data, draw_marker=draw_marker, **kwargs)

        return axes

    def plot_state_occupations(self, prediction=None, **kwargs):
        """
        Plot state occupation times (relative total time spent in state).

        Parameters
        ----------
        prediction : src.prediction.Prediction
            Container of mathematically derived statistical attributes and methods.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        from src.transitions import SingleState

        single_states = self.simulation.transition_set.single_states
        colormap = mpl.colors.ListedColormap(
            [
                mpl.colormaps["Spectral"](value)
                for value in np.linspace(0, 1, len(single_states))
            ]
        )
        colors, patches, xticks, data_merged, labels = [], [], 0, [], []
        for i, (fluorophore, states) in enumerate(single_states.items()):
            colors.extend([colormap(i) for _ in range(states.size)])
            patches.append(mpl.patches.Patch(color=colormap(i), label=fluorophore))
            xticks += states.size
            data_merged.append(self.state_occupations[fluorophore])
            labels.extend([SingleState(identity).name for identity in states])
        data_merged = np.concatenate(data_merged)
        data = [np.arange(xticks), data_merged]
        kwargs.setdefault("type_", "bar")
        kwargs.setdefault("xlabel", None)
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("edgecolor", "black")
        kwargs.setdefault("xticks", range(xticks))
        kwargs.setdefault("xticklabels", dict(labels=labels, rotation=70))
        kwargs.setdefault("ylabel", "PR")
        kwargs.setdefault("title", "occupation")
        kwargs.setdefault("color", colors)
        kwargs.setdefault("legend", True)
        kwargs.setdefault("legendhandles", patches)

        draw_marker = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                raise ValueError(
                    "prediction is based on different TransitionSet than simulation."
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted state_occupations not available if energy transfers "
                    "possible."
                )
            draw_marker = [
                np.arange(xticks),
                np.concatenate(
                    [
                        prediction.state_occupations[fluorophore]
                        for fluorophore in single_states
                    ]
                ),
            ]

        axes = fi.universal_figure(data=data, draw_marker=draw_marker, **kwargs)

        return axes

    def plot_lifetime_distributions(
        self, fluorophore, state_identity, prediction=None, **kwargs
    ):
        """
        Plot lifetime distributions of states.

        Parameters
        ----------
        fluorophore : str
            The name of the fluorophore whose state's distribution is to be shown.
        state_identity : int
            The identity of the state whose distribution is to be shown.
        prediction : src.prediction.Prediction
            Container of mathematically derived statistical attributes and methods.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        from src.transitions import SingleState

        kwargs.setdefault("type_", "hist")
        kwargs.setdefault("ylabel", "PD")
        kwargs.setdefault(
            "title", rf"$\tau$ of {fluorophore} {SingleState(state_identity).name}"
        )
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("xlabel", "lifetime [s]")
        kwargs.setdefault("density", True)
        index = np.where(
            self.simulation.transition_set.single_states[fluorophore] == state_identity
        )[0][0]
        data = self.lifetime_distributions[fluorophore][index]
        plot_distribution = None
        plot_distribution_label = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                raise ValueError(
                    "prediction is based on different TransitionSet than simulation."
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted lifetime_distributions not available if energy transfer "
                    "possible."
                )
            plot_distribution = prediction.lifetime_distributions[fluorophore][index]
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

    def plot_transition_time_distributions(
        self, fluorophore, transition_id, prediction=None, **kwargs
    ):
        """
        Plot distributions of time until transition occurs.

        Parameters
        ----------
        fluorophore : str
            The name of the fluorophore whose transition's distribution is to be shown.
        transition_id : int
            The identity of the transition whose distribution is to be shown.
        prediction : src.prediction.Prediction
            Container of mathematically derived statistical attributes and methods.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        kwargs.setdefault("type_", "hist")
        kwargs.setdefault("ylabel", "PD")
        kwargs.setdefault(
            "title",
            rf"""$\tau$ of {fluorophore} 
            {self.simulation.transition_set.transition_df.loc[(fluorophore, 
                                                               transition_id), 
                                                               "abbreviation"]}""",
        )
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("xlabel", "time to transition [s]")
        kwargs.setdefault("density", True)
        data = self.transition_time_distributions[transition_id]
        plot_distribution = None
        plot_distribution_label = None
        if prediction is not None:
            if prediction.transition_set is not self.simulation.transition_set:
                raise ValueError(
                    "prediction is based on different TransitionSet than simulation."
                )
            if prediction.energy_transfer:
                raise ValueError(
                    "predicted transsition_time_distributions not available if energy "
                    "transfer possible."
                )
            plot_distribution = prediction.transition_time_distributions[transition_id]
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
