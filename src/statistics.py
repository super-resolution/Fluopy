"""
Module statistics
"""
from scipy.stats import expon
import numpy as np
import pandas as pd
import src.figure as fi
from matplotlib.pyplot import cm
from src.transitions import SingleState
import src.miscellaneous as mi


class Prediction:
    """
    Container of lifetimes, state and transition occurrences obtained by computation-associated attributes and methods.

    Attributes
    ----------
    transitions : src.transitions.TransitionSet
        Collection of all relevant transitions and related attributes.
    lifetime_distributions : 1-D array_like
        Contains objects of type scipy.stats.*.rv_frozen for each entry in transitions.single_states.
    transition_time_distributions : 1-D array_like
        Contains objects of type scipy.stats.*.rv_frozen for each entry in transitions.transitions.
    mean_lifetimes : 1-D array_like
        Means of lifetime_distributions.
    mean_transition_times : 1-D array_like
        Means of transition_time_distributions.
    state_occurrences : 1-D array_like
        Expected relative frequencies of each entry in transitions.single_states.
    transition_occurrences : 1-D array_like
        Expected relative frequencies of each entry in transitions.transitions.
    state_occupations : 1-D array_like
        Expected probability of occupying each entry in transitions.single_states at any given point in time.
    """
    def __init__(self, transitions):
        """
        Parameters
        ----------
        transitions : src.transitions.TransitionSet
            Collection of all relevant transitions and related attributes.
        """
        if (transitions.transition_df['energy_transfer'] == True).any():
            raise ValueError('prediction not available if energy transfers can occur.')
        else:
            self.transitions = transitions
            self.lifetime_distributions, self.transition_time_distributions = \
                self.predict_lifetimes()
            self.mean_lifetimes = np.array([distr.mean() for distr in self.lifetime_distributions])
            self.mean_transition_times = np.array([distr.mean() for distr in self.transition_time_distributions])

            self.state_occurrences = self.predict_state_occurrences()
            self.transition_occurrences = self.predict_transition_occurrences()
            occupation_time_per_any_event = self.state_occurrences * self.mean_lifetimes
            self.state_occupations = occupation_time_per_any_event / occupation_time_per_any_event.sum()

    def predict_lifetimes(self):
        """
        Predict the lifetime distributions of transitions.single_states and the time until occurrence distributions of
        transitions.transitions.

        Returns
        -------
        lifetime_distributions : 1-D array_like
            Contains objects of type scipy.stats.*.rv_frozen for each entry in transitions.single_states.
        transition_time_distributions : 1-D array_like
            Contains objects of type scipy.stats.*.rv_frozen for each entry in transitions.transitions.
        """
        lifetime_distributions = np.empty(self.transitions.single_states.size, dtype=np.object)
        transition_time_distributions = np.empty(len(self.transitions.transitions), dtype=np.object)

        for i, state in enumerate(self.transitions.single_states):
            total_rate = 0
            associated_transitions = []
            for j, transition in enumerate(self.transitions.transitions):
                source = transition.initial_state.value
                if source == state:
                    total_rate += transition.rate
                    associated_transitions.append(j)

            lifetime_mean = 1 / total_rate

            lifetime_pdf = expon(scale=lifetime_mean)
            lifetime_distributions[i] = lifetime_pdf
            transition_time_distributions[associated_transitions] = lifetime_pdf

        return lifetime_distributions, transition_time_distributions

    def predict_state_occurrences(self):
        """
        Predict the relative frequencies of occurrences of transitions.single_states.

        Returns
        -------
        state_occurrences : 1-D array_like
            Expected relative frequencies of each entry in transitions.single_states.
        """
        single_states = self.transitions.single_states
        df = self.transitions.transition_df

        i_s0 = np.where(single_states == SingleState.S0.value)[0][0]
        i_s1 = np.where(single_states == SingleState.S1.value)[0][0]

        b = np.zeros(shape=single_states.size)
        b[0] = 1

        # sum of all states has to equal number of steps
        a_arrays = np.zeros(shape=(single_states.size, single_states.size))
        a_arrays[0][:] = 1

        # S1 has to occur as often as S0
        a_arrays[1][i_s0] = 1
        a_arrays[1][i_s1] = -1

        if single_states.size > 2:
            i_t1 = np.where(single_states == SingleState.T1.value)[0][0]
            s1_transitions = df[df['initial_state'] == SingleState.S1]
            s1_direct_deexcitation = s1_transitions[s1_transitions['final_state'] == SingleState.S0]
            s1_indirect_deexcitation = s1_transitions[s1_transitions['final_state'] != SingleState.S0]

            t1_transitions = df[df['initial_state'] == SingleState.T1]
            t1_direct_deexcitation = t1_transitions[t1_transitions['final_state'] == SingleState.S0]
            t1_indirect_deexcitation = t1_transitions[t1_transitions['final_state'] != SingleState.S0]

            # T1 coverage
            s1_nuniques = s1_indirect_deexcitation['abbreviation'].nunique()
            a_arrays[2][i_s0] = -1
            a_arrays[2][i_t1] = s1_direct_deexcitation['rate'].sum() / s1_indirect_deexcitation['rate'].sum() + 1
            # + 1 adds the population of the ith state

            if s1_nuniques == 2:
                i_cis = np.where(single_states == SingleState.Cis.value)[0][0]
                # T1 coverage
                a_arrays[2][i_cis] = s1_direct_deexcitation['rate'].sum() / s1_indirect_deexcitation['rate'].sum() + 1
                # Cis coverage
                s1_uniques = s1_indirect_deexcitation['abbreviation'].unique()
                rate_1 = s1_indirect_deexcitation[s1_indirect_deexcitation['abbreviation'] == s1_uniques[0]][
                    'rate'].sum()
                rate_2 = s1_indirect_deexcitation[s1_indirect_deexcitation['abbreviation'] != s1_uniques[0]][
                    'rate'].sum()
                a_arrays[i_cis][i_t1] = -1
                a_arrays[i_cis][i_cis] = rate_1 / rate_2
            elif s1_nuniques > 2:
                raise ValueError('Only two alternative singlet deexcitation pathways implemented.')

            # OFF coverage
            t1_nuniques = t1_indirect_deexcitation['abbreviation'].nunique()
            if t1_nuniques == 1:
                i_off = np.where(single_states == SingleState.OFF.value)[0][0]
                a_arrays[i_off][i_off] = -1
                a_arrays[i_off][i_t1] = t1_indirect_deexcitation['rate'].sum() / (t1_direct_deexcitation['rate'].sum() +
                                                                                  t1_indirect_deexcitation[
                                                                                      'rate'].sum())

            elif t1_nuniques > 1:
                raise ValueError('Only one alternative triplet deexcitation pathway implemented.')

        state_occurrences = np.linalg.solve(a_arrays, b)

        return state_occurrences

    def predict_transition_occurrences(self):
        """
        Predict the relative frequencies of occurrences of transitions.transitions.

        Returns
        -------
        transition_occurrences : 1-D array_like
            Expected relative frequencies of each entry in transitions.transitions.
        """
        single_states = self.transitions.single_states
        df = self.transitions.transition_df

        transition_occurrences = np.zeros(df.index.size)
        for index, row in df.iterrows():
            source = row['initial_state']
            if source == SingleState.S0:
                i = np.where(single_states == SingleState.S0.value)[0][0]
            elif source == SingleState.S1:
                i = np.where(single_states == SingleState.S1.value)[0][0]
            elif source == SingleState.T1:
                i = np.where(single_states == SingleState.T1.value)[0][0]
            elif source == SingleState.Cis:
                i = np.where(single_states == SingleState.Cis.value)[0][0]
            elif source == SingleState.OFF:
                i = np.where(single_states == SingleState.OFF.value)[0][0]
            else:
                raise ValueError

            state_occurrence = self.state_occurrences[i]
            total_rate = 1 / self.lifetime_distributions[i].mean()  # the extra [0] is needed because np.where
            # returns array, hence the indexed array returns an array[element] and not the element
            current_rate = row['rate']
            transition_occurrence = state_occurrence * current_rate / total_rate
            transition_occurrences[index] = transition_occurrence

        return transition_occurrences

    def plot(self, mode='state_occurrences', x=None, exclude=None, **kwargs):
        """
        Plot class attributes.

        Parameters
        ----------
        mode : str
            One of 'state_occurrences', 'transition_occurrences', 'mean_lifetimes', 'mean_transition_times',
            'state_occupations', 'lifetime_distributions', 'transition_time_distributions'.
        x : 1-D array_like
            Value range of the x-axis.
            Only used if mode is a probability distribution.
        exclude : Collection
            Ids of transitions.single_states or transitions.transitions that should not be displayed.
            Only used if mode is a probability distribution.
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        single_states = self.transitions.single_states
        df = self.transitions.transition_df

        if exclude is None:
            exclude = []
        if 'distribution' not in mode:
            if mode == 'transition_occurrences':
                data = [np.arange(df.index.size), self.transition_occurrences]
            elif mode == 'state_occurrences':
                data = [np.arange(single_states.size), self.state_occurrences]
            elif mode == 'mean_lifetimes':
                data = [np.arange(single_states.size), self.mean_lifetimes]
            elif mode == 'mean_transition_times':
                data = [np.arange(df.index.size), self.mean_transition_times]
            elif mode == 'state_occupations':
                data = [np.arange(single_states.size), self.state_occupations]
            else:
                raise AttributeError(f'mode {mode} unknown.')
            axes = plot_bar(data=data, single_states=single_states, df=df, mode=mode, **kwargs)
        else:
            if mode == 'lifetime_distributions':
                labels = [SingleState(id).name for id in single_states if id not in exclude]
                indices = [index for index, id in enumerate(single_states) if id not in exclude]
                x = np.linspace(0, 1, 1000) if x is None else x
                data = [[x, distribution.pdf(x)] for i, distribution in enumerate(self.lifetime_distributions) if
                        i in indices]
            elif mode == 'transition_time_distributions':
                labels = [transition for id, transition in df['abbreviation'].iteritems() if
                          id not in exclude]
                x = np.linspace(0, 1, 1000) if x is None else x
                data = [[x, distribution.pdf(x)] for id, distribution in enumerate(self.transition_time_distributions)
                        if id not in exclude]
            else:
                raise AttributeError(f'mode {mode} unknown.')
            axes = plot_prediction_distr(data=data, mode=mode, label=labels, **kwargs)

        return axes

    def plot_all(self, x_lifetimes=None, x_transitions=None, exclude_lifetimes=None,
                 exclude_transitions=None):
        """
        Plot all class attributes.

        Parameters
        ----------
        x_lifetimes : 1-D array_like
            Value range of the x-axis of lifetime_distributions
        x_transitions : 1-D array_like
            Value range of the x-axis of transition_time_distributions.
        exclude_lifetimes : Collection
            Ids of transitions.single_states whose lifetime_distribution should not be displayed.
        exclude_transitions : Collection
            Ids of transitions.transitions whose transition_time_distribuiton should not be displayed.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        axes = self.plot(mode='state_occurrences', ncols=4, nrows=2, fig_width=20, fig_height=6, scale=0.5)
        _ = self.plot(mode='mean_lifetimes', axes=axes[0, 1])
        _ = self.plot(mode='lifetime_distributions', axes=axes[0, 2], x=x_lifetimes, exclude=exclude_lifetimes)
        _ = self.plot(mode='state_occupations', axes=axes[0, 3])
        _ = self.plot(mode='transition_occurrences', axes=axes[1, 0])
        _ = self.plot(mode='mean_transition_times', axes=axes[1, 1])
        _ = self.plot(mode='transition_time_distributions', axes=axes[1, 2], x=x_transitions,
                      exclude=exclude_transitions)
        mi.delete_subplots(axes=axes, keep_number=7)
        axes[0, 0].get_figure().tight_layout()

        return axes


class Analysis:
    """
    Container of lifetimes, state and transition occurrences obtained by simulation-associated attributes and methods.

    Attributes
    ----------
    simulation : src.simulation.Simulation
        Container for simulation-associated attributes.
    transitions : src.transitions.TransitionSet
        Container for all relevant transitions and related attributes.
    lifetime_distributions : Collection
        Contains 1-D array_like for each entry in transitions.single_states.
    transition_time_distributions : Collection
        Contains 1-D array_like for each entry in transitions.transitions.
    mean_lifetimes : 1-D array_like
        Means of lifetime_distributions.
    mean_transition_times : 1-D array_like
        Means of transition_time_distributions.
    state_occurrences : 1-D array_like
        Relative frequencies of each entry in transitions.single_states.
    transition_occurrences : 1-D array_like
        Relative frequencies of each entry in transitions.transitions.
    state_occupations : 1-D array_like
        Probability of occupying each entry in transitions.single_states at any given point in time.
    """
    def __init__(self, simulation):
        """
        Parameters
        ----------
        simulation : src.simulation.Simulation
            Container for simulation-associated attributes.
        """
        if simulation.transition_series is None:
            raise ValueError('analysis not available if simulation has not been run.')
        else:
            self.simulation = simulation
            self.transitions = simulation.transitions
            self.lifetime_distributions, self.transition_time_distributions = \
                self.get_lifetimes()
            self.mean_lifetimes = np.array([np.mean(lifetime_distribution) if lifetime_distribution.size > 0 else np.nan
                                            for lifetime_distribution in self.lifetime_distributions])
            self.mean_transition_times = np.array([np.mean(transition_time_distribution)
                                                   if transition_time_distribution.size > 0 else np.nan
                                                   for transition_time_distribution
                                                   in self.transition_time_distributions])
            total_times = np.array([np.sum(lifetime_distribution) for lifetime_distribution in
                                    self.lifetime_distributions])

            state_occurrences, transition_occurrences = self.get_occurrences()
            self.state_occurrences = state_occurrences / np.sum(state_occurrences)
            self.transition_occurrences = transition_occurrences / np.sum(transition_occurrences)
            occupation_time_per_any_event = total_times / self.simulation.transition_series.size
            self.state_occupations = occupation_time_per_any_event / occupation_time_per_any_event.sum()

    def get_lifetimes(self):
        """
        Get the simulated lifetime distributions of transitions.single_states and the simulated time until occurrence
        distributions of transitions.transitions.
        Note: if energy transfer, the time to transition is only collected from the donor's point of view.

        Returns
        -------
        lifetime_distributions : Collection
            Contains 1-D array_like for each entry in transitions.single_states.
        transition_time_distributions : Collection
            Contains 1-D array_like for each entry in transitions.transitions.
        """
        lifetime_distributions = [np.array([]) for _ in self.transitions.single_states]
        transition_time_distributions = [np.array([]) for _ in self.transitions.transition_df.index]
        for i, state_series_fluorophore in enumerate(self.simulation.state_series):
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            changed = changes_at + 1
            initial_single_states = state_series_fluorophore[changes_at]
            total_times = self.simulation.time_series[changed]
            time_intervals = np.diff(total_times)
            time_intervals = np.insert(time_intervals, 0, total_times[0])
            for j, state in enumerate(self.transitions.single_states):
                time_intervals_state = time_intervals[np.where(initial_single_states == state)]
                lifetime_distributions[j] = np.concatenate([lifetime_distributions[j], time_intervals_state])

            transitions_fluorophore = self.simulation.transition_series[changes_at]
            for j in self.transitions.transition_df.index:
                combined_state_transition_ids = self.transitions.combined_state_transitions_df[
                    self.transitions.combined_state_transitions_df['transition_id'] == j].index.values
                transition_ids = np.in1d(transitions_fluorophore, combined_state_transition_ids).nonzero()[0]
                if self.transitions.transition_df.at[j, 'energy_transfer']:
                    source_donor = self.transitions.transition_df.at[j, 'initial_state'].donor.value
                    donor_indices = np.where(initial_single_states == source_donor)[0]
                    transition_ids = transition_ids[np.in1d(transition_ids, donor_indices)]

                time_intervals_transition = time_intervals[transition_ids]
                transition_time_distributions[j] = np.concatenate([transition_time_distributions[j],
                                                                   time_intervals_transition])

        return lifetime_distributions, transition_time_distributions

    def get_occurrences(self):
        """
        Get simulated total occurrences of transitions.single_states and transitions.transitions.

        Returns
        -------
        state_occurrences : 1-D array_like
            Total occurrences of each entry in transitions.single_states.
        transition_occurrences : 1-D array_like
            Total occurrences of each entry in transitions.transitions.
        """
        state_occurrences = np.zeros(shape=self.transitions.single_states.size, dtype=np.int64)
        transition_occurrences = np.zeros(shape=self.transitions.transition_df.index.size, dtype=np.int64)

        for state_series_fluorophore in self.simulation.state_series:
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            last_state = changes_at[-1] + 1
            changes_at_and_last = np.append(changes_at, last_state)
            states = state_series_fluorophore[changes_at_and_last]
            state_ids, state_counts = np.unique(states, return_counts=True)
            corresponding_state_indices = np.in1d(self.transitions.single_states, state_ids).nonzero()[0]
            state_occurrences[corresponding_state_indices] += state_counts

        for j in self.transitions.transition_df.index:
            combined_state_transition_ids = self.transitions.combined_state_transitions_df[
                self.transitions.combined_state_transitions_df['transition_id'] == j].index.values
            transition_ids = np.in1d(self.simulation.transition_series, combined_state_transition_ids).nonzero()[0]
            transition_occurrences[j] = transition_ids.size

        return state_occurrences, transition_occurrences

    def plot(self, mode='state_occurrences', exclude=None, prediction=None, **kwargs):
        """
        Plot class attributes.

        Parameters
        ----------
        mode : str
            One of 'state_occurrences', 'transition_occurrences', 'mean_lifetimes', 'mean_transition_times',
            'state_occupations', 'lifetime_distributions', 'transition_time_distributions'.
        exclude : Collection
            Ids of transitions.single_states or transitions.transitions that should not be displayed.
            Only used if mode is a probability distribution.
        prediction : None, Prediction
            Container of lifetimes, state and transition occurrences obtained by computation-associated attributes and
            methods.
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if exclude is None:
            exclude = []
        marker = None
        x_states = np.arange(self.transitions.single_states.size)
        x_transitions = np.arange(self.transitions.transition_df.index.size)
        if 'distribution' not in mode:
            if mode == 'transition_occurrences':
                data = [x_transitions, self.transition_occurrences]
                if prediction is not None:
                    marker = [x_transitions, prediction.transition_occurrences]
            elif mode == 'state_occurrences':
                data = [x_states, self.state_occurrences]
                if prediction is not None:
                    marker = [x_states, prediction.state_occurrences]
            elif mode == 'mean_lifetimes':
                data = [x_states, np.nan_to_num(self.mean_lifetimes)]
                if prediction is not None:
                    marker = [x_states, prediction.mean_lifetimes]
            elif mode == 'mean_transition_times':
                data = [x_transitions, np.nan_to_num(self.mean_transition_times)]
                if prediction is not None:
                    marker = [x_transitions, prediction.mean_transition_times]
            elif mode == 'state_occupations':
                data = [x_states, self.state_occupations]
                if prediction is not None:
                    marker = [x_states, prediction.state_occupations]
            else:
                raise AttributeError(f'mode {mode} unknown.')
            axes = plot_bar(data=data, single_states=self.transitions.single_states, df=self.transitions.transition_df,
                            mode=mode, draw_marker=marker, **kwargs)
        else:
            plot_distribution = None
            if mode == 'lifetime_distributions':
                labels = [SingleState(id).name for id in self.transitions.single_states if id not in exclude]
                indices = [index for index, id in enumerate(self.transitions.single_states) if id not in exclude]
                data = [distribution for i, distribution in enumerate(self.lifetime_distributions) if
                        i in indices]
                if prediction is not None:
                    plot_distribution = prediction.lifetime_distributions[indices]
            elif mode == 'transition_time_distributions':
                labels = [transition for id, transition in self.transitions.transition_df['abbreviation'].iteritems() if
                          id not in exclude]
                data = [distribution for id, distribution in enumerate(self.transition_time_distributions) if
                        id not in exclude]
                if prediction is not None:
                    plot_distribution = [distribution for id, distribution in
                                         enumerate(prediction.transition_time_distributions) if id not in exclude]
            else:
                raise AttributeError(f'mode {mode} unknown.')

            kwargs.setdefault('label', labels)
            axes = plot_analysis_distr(data=data, mode=mode, plot_distribution=plot_distribution,
                                       **kwargs)

        return axes

    def plot_all(self, exclude_lifetimes=None, exclude_transitions=None, prediction=None):
        """
        Plot all class attributes.

        Parameters
        ----------
        exclude_lifetimes : Collection
            Ids of transitions.single_states whose lifetime_distribution should not be displayed.
        exclude_transitions : Collection
            Ids of transitions.transitions whose transition_time_distribution should not be displayed.
        prediction : None, Prediction
            Container of lifetimes, state and transition occurrences obtained by computation-associated attributes and
            methods.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        axes = self.plot(mode='state_occurrences', ncols=4, nrows=2, fig_width=20, fig_height=6, scale=0.5,
                         prediction=prediction)
        _ = self.plot(mode='mean_lifetimes', axes=axes[0, 1], prediction=prediction)
        _ = self.plot(mode='lifetime_distributions', axes=axes[0, 2], exclude=exclude_lifetimes,
                      prediction=prediction)
        _ = self.plot(mode='state_occupations', axes=axes[0, 3], prediction=prediction)
        _ = self.plot(mode='transition_occurrences', axes=axes[1, 0], prediction=prediction)
        _ = self.plot(mode='mean_transition_times', axes=axes[1, 1], prediction=prediction)
        _ = self.plot(mode='transition_time_distributions', axes=axes[1, 2],
                      exclude=exclude_transitions, prediction=prediction)
        mi.delete_subplots(axes=axes, keep_number=7)
        axes[0, 0].get_figure().tight_layout()

        return axes


def plot_bar(data, single_states, df, mode='state_occurrences', **kwargs):
    """
    Bar plot.

    Parameters
    ----------
    data : 2-D array_like
        Contains x and y data.
    single_states : 1-D array_like
        Contains the values of all relevant SingleStates.
    df : pd.DataFrame
        Dataframe of all transitions with non-zero rate containing their id as index and their other attributes as
        columns.
    mode : str
        One of 'state_occurrences', 'transition_occurrences', 'mean_lifetimes', 'mean_transition_times',
        'state_occupations'.
    kwargs : src.figure.universal_figure arguments

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    from src.transitions import SingleState
    kwargs.setdefault('type_', 'bar')
    kwargs.setdefault('xlabel', None)
    kwargs.setdefault('yscale', 'log')
    kwargs.setdefault('edgecolor', 'black')
    if 'transition' in mode:
        kwargs.setdefault('xticks', range(df.index.size))
        kwargs.setdefault('xticklabels', dict(labels=df['abbreviation'], rotation=70))
    else:
        kwargs.setdefault('xticks', range(single_states.size))
        kwargs.setdefault('xticklabels', dict(labels=[SingleState(id).name for id in single_states], rotation=70))
    if mode == 'state_occurrences':
        kwargs.setdefault('ylabel', 'PR')
        kwargs.setdefault('title', 'single states')
    elif mode == 'transition_occurrences':
        kwargs.setdefault('ylabel', 'PR')
        kwargs.setdefault('title', 'transitions')
    elif mode == 'mean_lifetimes':
        kwargs.setdefault('ylabel', 'mean [s]')
        kwargs.setdefault('title', 'lifetimes')
    elif mode == 'mean_transition_times':
        kwargs.setdefault('ylabel', 'mean [s]')
        kwargs.setdefault('title', 'time to transition')
    elif mode == 'state_occupations':
        kwargs.setdefault('ylabel', 'PR')
        kwargs.setdefault('title', 'occupation')

    axes = fi.universal_figure(data=data, **kwargs)

    return axes


def plot_prediction_distr(data, mode='lifetime_distributions', **kwargs):
    """
    Plot predicted distributions.

    Parameters
    ----------
    data : Collection
        Contains 2-D array_like.
    mode : str
        One of 'lifetime_distributions', 'transition_time_distributions'.
    kwargs : src.figure.universal_figure arguments

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    kwargs.setdefault('type_', 'multiple_line')
    kwargs.setdefault('ylabel', 'PD')
    kwargs.setdefault('legend', True)
    kwargs.setdefault('yscale', 'log')
    colors = cm.get_cmap('rainbow', len(data))
    kwargs.setdefault('color', colors)
    if mode == 'lifetime_distributions':
        kwargs.setdefault('xlabel', 'lifetime [s]')
    elif mode == 'transition_time_distributions':
        kwargs.setdefault('xlabel', 'time to transition [s]')

    axes = fi.universal_figure(data=data, **kwargs)

    return axes


def plot_analysis_distr(data, mode='lifetime_distributions', **kwargs):
    """
    Plot simulated distributions.

    Parameters
    ----------
    data : Collection
        Contains 1-D array_like.
    mode : str
        One of 'lifetime_distributions', 'transition_time_distributions'.
    kwargs : src.figure.universal_figure arguments

    Returns
    -------
    axes : np.ndarray
        Contains matplotlib.axes._subplots.AxesSubplots.
    """
    kwargs.setdefault('type_', 'multiple_hist')
    kwargs.setdefault('ylabel', 'PD')
    kwargs.setdefault('legend', True)
    kwargs.setdefault('yscale', 'log')
    kwargs.setdefault('density', True)
    colors = cm.get_cmap('rainbow', len(data))
    kwargs.setdefault('color', colors)
    if mode == 'lifetime_distributions':
        kwargs.setdefault('xlabel', 'lifetime [s]')
    elif mode == 'transition_time_distributions':
        kwargs.setdefault('xlabel', 'time to transition [s]')

    axes = fi.universal_figure(data=data, **kwargs)

    return axes
