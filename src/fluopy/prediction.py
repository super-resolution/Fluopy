"""
Module prediction
"""
import logging
import re

import matplotlib as mpl
import numpy as np
from scipy.stats import expon

from . import figure as fi
from .miscellaneous import format_electronic_state, format_transition

logger = logging.getLogger(__name__)

class Prediction:
    """
    Container of mathematically derived statistical attributes and methods.

    Attributes
    ----------
    energy_transfer : bool
        Whether the prediction was carried out on energy transfer systems.
    absorbing_chain : bool
        Whether the prediction was carried out on an absorbing Markov chain.
    transition_set : fluopy.transitions.TransitionSet
        Collection of all relevant transitions and related attributes.
    frequency_transitions : 1-D array_like
        Expected relative frequencies of each transition.
    frequency_states : dict
        Name of fluorophores as keys and their state's expected relative frequencies
        (array) as values.
    transition_time_distributions : 1-D array_like
        Expected distributions of time until transition.
        Contains objects of type scipy.stats.*.rv_frozen for each transition.
    lifetime_distributions : dict
        Name of fluorophores as keys and their state's expected lifetime distributions
        (objects of type scipy.stats.*.rv_frozen) (array) as values.
    mean_transition_times : 1-D array_like
        Expected means of time until transition.
    mean_lifetimes : dict
        Name of fluorophores as keys and their state's expected lifetime means (array)
        as values.
    state_occupations : dict
        Name of fluorophores as keys and their state's expected probability of being
        occupied at any given point in time (array) as values.
    """

    def __init__(self, transition_set, accuracy=1e9):
        """
        Parameters
        ----------
        transition_set : fluopy.transitions.TransitionSet
            Collection of all relevant transitions and related attributes.
        accuracy : float
            Determines the exponent of matrix power. The higher, the more accurate up
            to the point floating point precision impairs the result.
        """
        self.energy_transfer = False
        self.absorbing_chain = False
        if transition_set.fluorophore_system.count > 2:
            raise ValueError("prediction not available for more than 2 fluorophores.")
        # too large matrix in np.linalg.matrix_power
        if transition_set.transition_matrix is None:
            raise ValueError(
                "prediction not available if transition_set not finalized."
            )
        if any(
            "dist" in fluorophore_comb
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
        if transition_set.transition_df["absorbing"].any():
            logger.warning(
                "absorbing states have a lifetime of inf and a frequency / occupation "
                "of 0. Absorbing transitions have a frequency of 0.",
                stacklevel=2,
            )
            self.absorbing_chain = True

        self.transition_set = transition_set
        if self.absorbing_chain:
            self.frequency_transitions = self.predict_transition_occurrences_abs()
        else:
            self.frequency_transitions = self.predict_transition_occurrences(
                accuracy=int(accuracy)
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

    def predict_transition_occurrences(self, accuracy):
        """
        Predict the relative frequencies of transitions.

        Parameters
        ----------
        accuracy : int
            Determines the exponent of matrix power. The higher, the more accurate up
            to the point floating point precision impairs the result.

        Returns
        -------
        frequency_transitions : 1-D array_like
            Expected relative frequencies of each transition.
        """
        matrix_power = np.linalg.matrix_power(
            self.transition_set.transition_matrix, accuracy
        )
        stationary_distribution_combined_state_transitions = matrix_power[0]
        # https://brilliant.org/wiki/stationary-distributions/
        frequency_transitions = np.zeros(self.transition_set.transition_df.shape[0])
        df = self.transition_set.combined_state_transitions_df
        for _, i in self.transition_set.transition_df.index:
            indices = df.index[df["transition_id"] == i].tolist()
            frequency_transitions[i] = (
                stationary_distribution_combined_state_transitions[indices].sum()
            )

        grouper = {}
        for fluorophore_comb, group in self.transition_set.transition_df.groupby(
            level=0, sort=False
        ):
            if "dist" in fluorophore_comb:
                pattern = r"D:\s*([^,]+),\s*A:\s*([^,]+),\s*dist:\s*([\d.]+)"
                match = re.match(pattern, fluorophore_comb)
                d, _, _ = match.group(1), match.group(2), match.group(3)
            else:
                d = fluorophore_comb
            if d in grouper:
                grouper[d] += group.index.get_level_values(1).tolist()
            else:
                grouper[d] = group.index.get_level_values(1).tolist()

        for _, indices in grouper.items():
            frequency_transitions[indices] /= np.sum(frequency_transitions[indices])

        return frequency_transitions

    def predict_transition_occurrences_abs(self):
        """
        Predict the relative frequencies of transitions. Absorbing transitions will
        have the value 0.

        Returns
        -------
        frequency_transitions : 1-D array_like
            Expected relative frequencies of each transition.
        """
        transition_abs = self.transition_set.transition_df["absorbing"]
        transition_abs_df = self.transition_set.transition_df[transition_abs]
        abs_final_state = []
        for fluorophore in self.transition_set.fluorophore_system.fluorophores:
            states = transition_abs_df["final_state"].xs(fluorophore.name, level=0)
            abs_final_state.append(states.iloc[0].value)
        abs_final_state = tuple(abs_final_state)
        abs_indices = transition_abs[transition_abs].index.get_level_values(1)
        df = self.transition_set.combined_state_transitions_df
        abs_indices_combined = df[df["transition_id"].isin(abs_indices)].index
        drop_transitions = df[df["final_state"] == abs_final_state].index
        drop_diff = abs_indices_combined[
            ~np.isin(abs_indices_combined, drop_transitions)
        ]
        Q = get_Q(
            P=self.transition_set.transition_matrix, drop_transitions=drop_transitions
        )
        I_t = get_I_t(Q=Q)
        N = get_N(I_t=I_t, Q=Q)
        expected_transient_visits = N[0]
        expected_visits = np.zeros(
            expected_transient_visits.size + drop_transitions.size,
            dtype=expected_transient_visits.dtype,
        )
        mask = np.ones(len(expected_visits), dtype=bool)
        mask[drop_transitions] = False
        expected_visits[mask] = expected_transient_visits
        expected_visits[drop_diff] = 0
        frequency_transitions = np.zeros(transition_abs.size)
        for _, i in self.transition_set.transition_df.index:
            indices = df.index[df["transition_id"] == i].tolist()
            frequency_transitions[i] = expected_visits[indices].sum()

        grouper = {}
        for fluorophore_comb, group in self.transition_set.transition_df.groupby(
            level=0, sort=False
        ):
            if "dist" in fluorophore_comb:
                pattern = r"D:\s*([^,]+),\s*A:\s*([^,]+),\s*dist:\s*([\d.]+)"
                match = re.match(pattern, fluorophore_comb)
                d, _, _ = match.group(1), match.group(2), match.group(3)
            else:
                d = fluorophore_comb
            if d in grouper:
                grouper[d] += group.index.get_level_values(1).tolist()
            else:
                grouper[d] = group.index.get_level_values(1).tolist()

        for _, indices in grouper.items():
            frequency_transitions[indices] /= np.sum(frequency_transitions[indices])

        return frequency_transitions

    def predict_state_occurrences(self):
        """
        Predict the relative frequencies of states.

        Returns
        -------
        frequency_states : dict
            Name of fluorophores as keys and their state's expected relative
            frequencies (array) as values.
        """
        single_states = self.transition_set.single_states
        frequency_states = {
            key: np.zeros(len(value)) for key, value in single_states.items()
        }
        grouped = self.transition_set.transition_df.groupby(level=0)
        for fluorophore_comb, f_transitions in grouped:
            if "dist" in fluorophore_comb:
                pattern = r"D:\s*([^,]+),\s*A:\s*([^,]+),\s*dist:\s*([\d.]+)"
                match = re.match(pattern, fluorophore_comb)
                d, a, _ = match.group(1), match.group(2), match.group(3)
                single_states_a = single_states[a]
                single_states_d = single_states[d]
                factor = 1
                for (_, identity), transition in f_transitions.iterrows():
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
                for (_, identity), transition in f_transitions.iterrows():
                    index = np.where(
                        single_states_f == transition["final_state"].value
                    )[0][0]
                    frequency_states[fluorophore_comb][
                        index
                    ] += self.frequency_transitions[identity]
        for fluorophore, state_frequencies in frequency_states.items():
            frequency_states[fluorophore] /= state_frequencies.sum()

        return frequency_states

    def predict_lifetimes(self):
        """
        Predict the lifetime distributions of states and the time until occurrence
        distributions of transitions.

        Returns
        -------
        transition_time_distributions : 1-D array_like
            Expected distributions of time until transition.
            Contains objects of type scipy.stats.*.rv_frozen for each transition.
        lifetime_distributions : dict
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
                total_rate = 0
                associated_transitions = []
                for j, transition in self.transition_set.transition_df.loc[
                    fluorophore
                ].iterrows():
                    source = transition.initial_state.value
                    if source == state:
                        total_rate += transition.rate
                        associated_transitions.append(j)
                if total_rate == 0:
                    lifetime_mean = np.inf
                    lifetime_pdf = np.inf
                else:
                    lifetime_mean = 1 / total_rate
                    lifetime_pdf = expon(scale=lifetime_mean)
                lifetime_distributions[fluorophore][i] = lifetime_pdf
                transition_time_distributions[associated_transitions] = lifetime_pdf

        return transition_time_distributions, lifetime_distributions

    def infer_stats(self):
        """
        Infers statistics of states based on lifetime distributions and frequencies.

        Returns
        -------
        mean_lifetimes : dict
            Name of fluorophores as keys and their state's expected lifetime means
            (array) as values.
        state_occupations : dict
            Name of fluorophores as keys and their state's expected probability of
            being occupied at any given point in time (array) as values.
        """
        mean_lifetimes = {}
        state_occupations = {}
        for fluorophore, distributions in self.lifetime_distributions.items():
            mean_lifetimes[fluorophore] = np.array(
                [distr.mean() if distr != np.inf else np.inf for distr in distributions]
            )
            state_occupations[fluorophore] = np.multiply(
                self.frequency_states[fluorophore],
                mean_lifetimes[fluorophore],
                where=mean_lifetimes[fluorophore] != np.inf,
                out=np.zeros(self.frequency_states[fluorophore].size),
            )
            state_occupations[fluorophore] /= state_occupations[fluorophore].sum()

        return mean_lifetimes, state_occupations

    def plot_frequency_transitions(self, **kwargs):
        """
        Plot frequencies of transitions.

        Returns
        -------
        axes : np.ndarray
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

    def plot_frequency_states(self, **kwargs):
        """
        Plot frequencies of states.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        from .transitions import SingleState

        single_states = self.transition_set.single_states
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
            labels.extend(
                [
                    format_electronic_state(SingleState(identity).name)
                    for identity in states
                ]
            )
        data_merged = np.concatenate(data_merged)
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

    def plot_mean_transition_times(self, **kwargs):
        """
        Plot mean times until transitions occur.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.energy_transfer:
            raise ValueError(
                "mean_transition_times not available if energy transfers possible."
            )
        df = self.transition_set.transition_df
        data = [np.arange(df.shape[0]), self.mean_transition_times]
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

    def plot_mean_lifetimes(self, **kwargs):
        """
        Plot mean lifetimes of states.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.energy_transfer:
            raise ValueError(
                "mean_lifetimes not available if energy transfers possible."
            )
        from .transitions import SingleState

        single_states = self.transition_set.single_states
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
            labels.extend(
                [
                    format_electronic_state(SingleState(identity).name)
                    for identity in states
                ]
            )
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
        kwargs.setdefault("ylabel", r"$\tau$ (s)")
        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_state_occupations(self, **kwargs):
        """
        Plot state occupation times (relative total time spent in state).

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.energy_transfer:
            raise ValueError(
                "state_occupations not available if energy transfers possible."
            )
        from .transitions import SingleState

        single_states = self.transition_set.single_states
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
            labels.extend(
                [
                    format_electronic_state(SingleState(identity).name)
                    for identity in states
                ]
            )
        data_merged = np.concatenate(data_merged)
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
        self, fluorophore, state_identity, x=None, **kwargs
    ):
        """
        Plot lifetime distributions of states.

        Parameters
        ----------
        fluorophore : str
            The name of the fluorophore whose state's distribution is to be shown.
        state_identity : int
            The identity of the state whose distribution is to be shown.
        x : 1-D array_like
            The x values for which the distribution is to be shown.
        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.energy_transfer:
            raise ValueError(
                "lifetime_distributions not available if energy transfers possible."
            )
        from .transitions import SingleState

        kwargs.setdefault("type_", "line")
        kwargs.setdefault("ylabel", "PD")
        kwargs.setdefault(
            "title", rf"$\tau$ of {fluorophore} {SingleState(state_identity).name}"
        )
        kwargs.setdefault("yscale", "log")
        kwargs.setdefault("xlabel", "lifetime [s]")
        index = np.where(
            self.transition_set.single_states[fluorophore] == state_identity
        )[0][0]
        if isinstance(self.lifetime_distributions[fluorophore][index], float):
            raise ValueError(
                "The lifetimes are all equal to "
                f"{self.lifetime_distributions[fluorophore][index]}"
            )

        if x is None:
            x = np.linspace(0, self.mean_lifetimes[fluorophore][index] * 10, 1000)
        data = [x, self.lifetime_distributions[fluorophore][index].pdf(x)]
        axes = fi.universal_figure(data=data, **kwargs)

        return axes

    def plot_transition_time_distributions(
        self, fluorophore, transition_id, x=None, **kwargs
    ):
        """
        Plot distributions of time until transition occurs.

        Parameters
        ----------
        fluorophore : str
            The name of the fluorophore whose transition's distribution is to be shown.
        transition_id : int
            The identity of the transition whose distribution is to be shown.
        x : 1-D array_like
            The x values for which the distribution is to be shown.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.energy_transfer:
            raise ValueError(
                "transition_time_distributions not available if energy transfers "
                "possible."
            )
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
            x = np.linspace(0, self.mean_transition_times[transition_id] * 10, 1000)
        data = [x, self.transition_time_distributions[transition_id].pdf(x)]

        axes = fi.universal_figure(data=data, **kwargs)

        return axes


def get_Q(P, drop_transitions):
    """
    Q describes the probability of transitioning from some transient state to another.

    Parameters
    ----------
    P : np.ndarray
        Transition matrix with transient states t and absorbing state r.
    drop_transitions : int or array of ints
        Index of absorbing state (i.e., photophysical transition with no return).

    Returns
    -------
    Q : np.ndarray
        Transition matrix with transient states t.
    """
    # Q takes the original transition matrix into account, because within Q the state
    # that leads to the absorbing state has to take on the probability GIVEN the
    # possibility of the transition to the absorbing state.
    Q = np.delete(P, drop_transitions, axis=0)
    Q = np.delete(Q, drop_transitions, axis=1)

    return Q


def get_I_t(Q):
    """
    I_t is the identity matrix of Q.

    Parameters
    ----------
    Q : np.ndarray
        Transition matrix with transient states t.

    Returns
    -------
    I_t : np.ndarray
        Identity matrix of Q.
    """
    I_t = np.identity(Q.shape[0])

    return I_t


def get_N(I_t, Q):
    """
    N is the fundamental matrix. At entry (i, j) it contains the expected number
    of visits to a transient state j starting from transient state i before being
    absorbed.

    Parameters
    ----------
    I_t : np.ndarray
        Identity matrix of Q.
    Q : np.ndarray
        Transition matrix with transient states t.

    Returns
    -------
    N : np.ndarray
        Fundamental matrix of absorbing Markov chain.
    """
    N = np.linalg.inv(I_t - Q)

    return N
