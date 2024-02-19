"""
Module statistics
"""
from scipy.stats import expon
import numpy as np
import warnings
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
    stationary_distribution_states : 1-D array_like
        Expected relative frequencies of each entry in transitions.single_states.
    stationary_distribution_transitions : 1-D array_like
        Expected relative frequencies of each entry in transitions.transitions.
    state_occupations : 1-D array_like
        Expected probability of occupying each entry in transitions.single_states at any given point in time.
    energy_transfer : bool
        Whether the prediction was carried out on energy transfer systems.
    """
    def __init__(self, transitions, accuracy=int(1e9)):
        """
        Parameters
        ----------
        transitions : src.transitions.TransitionSet
            Collection of all relevant transitions and related attributes.
        accuracy : int
            Determines the exponent of matrix power. The higher, the more accurate up to the point floating point
            precision impairs the result.
        """
        self.energy_transfer = False

        if (transitions.transition_df['energy_transfer'] == True).any():
            if transitions.fluorophore_system.count > 2:
                raise ValueError('prediction not available if energy transfers can occur and number of fluorophores'
                                 'is larger than 2 due to too large matrix in np.linalg.matrix_power().')
            else:
                warnings.warn('prediction accuracy of energy transfers more difficult to tune due to larger matrix in '
                              'np.linalg.matrix_power(). Only stationary distributions available. Lifetimes and occupations not '
                              'available.', stacklevel=2)
                self.energy_transfer = True
        else:
            if transitions.fluorophore_system.count > 1:
                raise ValueError('only 1 fluorophore needed. reduce_to_1() TransitionSet first.')
        for single_state in transitions.single_states:
            if single_state not in transitions.transition_df['initial_state'].apply(lambda x: x.value).values:  
                raise ValueError('absorbing state found. remove_absorbing_states() TransitionSet first.')
        
        self.transitions = transitions
        self.stationary_distribution_transitions = \
            self.predict_transition_occurrences(accuracy=accuracy)
        self.stationary_distribution_states = self.predict_state_occurrences()
        if not self.energy_transfer:
            self.lifetime_distributions, self.transition_time_distributions = \
                self.predict_lifetimes()
            self.mean_lifetimes = np.array([distr.mean() for distr in self.lifetime_distributions])
            self.mean_transition_times = np.array([distr.mean() for distr in self.transition_time_distributions])
            occupation_time_per_any_event = self.stationary_distribution_states * self.mean_lifetimes
            self.state_occupations = occupation_time_per_any_event / occupation_time_per_any_event.sum()
        else:
            self.lifetime_distributions, self.transition_time_distributions, self.mean_lifetimes, \
                self.mean_transition_times, self.state_occupations = None, None, None, None, None

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
        lifetime_distributions = np.empty(self.transitions.single_states.size, dtype=object)
        transition_time_distributions = np.empty(self.transitions.transition_df.shape[0], dtype=object)

        for i, state in enumerate(self.transitions.single_states):
            total_rate = 0
            associated_transitions = []
            for j, transition in self.transitions.transition_df.iterrows():
                source = transition.initial_state.value
                if source == state:
                    total_rate += transition.rate
                    associated_transitions.append(j)

            lifetime_mean = 1 / total_rate

            lifetime_pdf = expon(scale=lifetime_mean)
            lifetime_distributions[i] = lifetime_pdf
            transition_time_distributions[associated_transitions] = lifetime_pdf

        return lifetime_distributions, transition_time_distributions

    def predict_transition_occurrences(self, accuracy):
        """
        Predict the relative frequencies of occurrences of transitions.transitions.

        Parameters
        ----------
        accuracy : int
            Determines the exponent of matrix power. The higher, the more accurate up to the point floating point
            precision impairs the result.

        Returns
        -------
        stationary_distribution_transitions : 1-D array_like
            Expected relative frequencies of each entry in transitions.transitions.
        """
        matrix_power = np.linalg.matrix_power(self.transitions.transition_matrix, accuracy)
        stationary_distribution_combined_state_transitions = matrix_power[0]
        # note the restrictions of this method: https://brilliant.org/wiki/stationary-distributions/
        transitions_combined = self.transitions.combined_state_transitions_df['abbreviation']
        transitions = self.transitions.transition_df['abbreviation']
        stationary_distribution_transitions = np.zeros_like(transitions)
        for i, transition in enumerate(transitions):
            indices = transitions_combined.index[transitions_combined == transition].tolist()
            stationary_distribution_transitions[i] = stationary_distribution_combined_state_transitions[indices].sum()

        return stationary_distribution_transitions

    def predict_state_occurrences(self):
        """
        Predict the relative frequencies of occurrences of transitions.single_states.

        Returns
        -------
        stationary_distribution_states : 1-D array_like
            Expected relative frequencies of each entry in transitions.single_states.
        """
        single_states = self.transitions.single_states
        stationary_distribution_states = np.zeros_like(single_states, dtype=np.float64)
        transitions = self.transitions.transition_df
        for n, (_, transition) in enumerate(transitions.iterrows()):
            final_state = transition['final_state']
            factor = 1
            if transition['energy_transfer']:
                _, acceptor_i = transition['initial_state'].single_state_values
                donor_f, acceptor_f = final_state.single_state_values
                if acceptor_i == acceptor_f:
                    indices = np.where(single_states == donor_f)[0][0]
                elif donor_f == acceptor_f:
                    indices = np.where(single_states == donor_f)[0][0]
                else:
                    indices = np.array([np.where(single_states == donor_f)[0][0], np.where(single_states == acceptor_f)[0][0]])
                    factor = 0.5
            else:
                indices = np.where(single_states == final_state.value)[0][0]

            stationary_distribution_states[indices] += self.stationary_distribution_transitions[n] * factor
            # factor to adjust that an energy transfer effects two fluorophores, not only one

        return stationary_distribution_states

    def plot(self, mode='state_occurrences', x=None, include=None, **kwargs):
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
        include : Collection
            Ids of transitions.single_states or transitions.transitions that should be displayed.
            Only used if mode is a probability distribution.
        kwargs : src.figure.universal_figure arguments

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        single_states = self.transitions.single_states
        df = self.transitions.transition_df

        if include is None:
            include = [0]
        if 'distribution' not in mode:
            if mode == 'transition_occurrences':
                data = [np.arange(df.index.size), self.stationary_distribution_transitions]
            elif mode == 'state_occurrences':
                data = [np.arange(single_states.size), self.stationary_distribution_states]
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
                labels = [SingleState(identity).name for identity in single_states if identity in include]
                indices = [index for index, identity in enumerate(single_states) if identity in include]
                x = np.linspace(0, 1, 1000) if x is None else x
                data = [[x, distribution.pdf(x)] for i, distribution in enumerate(self.lifetime_distributions) if
                        i in indices]
            elif mode == 'transition_time_distributions':
                labels = [transition for identity, transition in df['abbreviation'].items() if
                          identity in include]
                x = np.linspace(0, 1, 1000) if x is None else x
                data = [[x, distribution.pdf(x)] for identity, distribution in enumerate(self.transition_time_distributions)
                        if identity in include]
            else:
                raise AttributeError(f'mode {mode} unknown.')
            kwargs.setdefault('label', labels)
            axes = plot_prediction_distr(data=data, mode=mode, **kwargs)

        return axes

    def plot_all(self, x_lifetimes=None, x_transitions=None, include_lifetimes=None,
                 include_transitions=None, scale=0.5):
        """
        Plot all class attributes.

        Parameters
        ----------
        x_lifetimes : 1-D array_like
            Value range of the x-axis of lifetime_distributions
        x_transitions : 1-D array_like
            Value range of the x-axis of transition_time_distributions.
        include_lifetimes : Collection
            Ids of transitions.single_states whose lifetime_distribution should be displayed.
        include_transitions : Collection
            Ids of transitions.transitions whose transition_time_distribuiton should be displayed.
        scale : float
            Factor to scale the figure.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if self.energy_transfer:
            axes = self.plot(mode='state_occurrences', ncols=2, nrows=1, fig_width=20, fig_height=6, scale=scale)
            _ = self.plot(mode='transition_occurrences', axes=axes[0, 1])
        
        else:
            axes = self.plot(mode='state_occurrences', ncols=4, nrows=2, fig_width=20, fig_height=6, scale=scale)
            _ = self.plot(mode='mean_lifetimes', axes=axes[0, 1])
            _ = self.plot(mode='lifetime_distributions', axes=axes[0, 2], x=x_lifetimes, include=include_lifetimes)
            _ = self.plot(mode='state_occupations', axes=axes[0, 3])
            _ = self.plot(mode='transition_occurrences', axes=axes[1, 0])
            _ = self.plot(mode='mean_transition_times', axes=axes[1, 1])
            _ = self.plot(mode='transition_time_distributions', axes=axes[1, 2], x=x_transitions,
                        include=include_transitions)
            mi.delete_subplots(axes=axes, keep_number=7)
            axes[0, 0].get_figure().tight_layout()

        return axes


class Analysis:
    """
    Container of lifetimes, state and transition occurrences obtained by simulation-associated attributes and methods.
    Note that here, state_occurrences and transition_occurrences correspond to stationary_distribution_states and
    stationary_distribution_transitions of Prediction. If the simulation has not taken sufficient steps, the resulting
    distribution might vary from the stationary distribution.

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
    energy_transfer : bool
        Whether the analysis was carried out on energy transfer systems.
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
            self.energy_transfer = False
            self.simulation = simulation
            self.single_states = simulation.transitions.single_states
            self.transition_df = simulation.transitions.transition_df
            if (self.transition_df['energy_transfer'] == True).any():
                self.energy_transfer = True
            self.transitions = simulation.transitions

            absorbing_states = []
            for i, single_state in enumerate(self.single_states):
                single_state_obj = SingleState(single_state)
                if single_state_obj not in self.transition_df['initial_state'].values:
                    absorbing_states.append(single_state_obj)
                    self.single_states = np.delete(self.single_states, i)
                    drop_transitions = self.transition_df[self.transition_df['final_state'] == single_state_obj].index
                    self.transition_df = self.transition_df.drop(drop_transitions)
            if len(absorbing_states) > 0:
                count = 0
                for i, state_series in enumerate(self.simulation.state_series):
                    last_single_state = SingleState(state_series[-1])
                    if last_single_state in absorbing_states:
                        count += 1
                        print(f'fluorophore {i} has reached the markovian absorbing state {last_single_state}')
                print(f'{count} of {self.transitions.fluorophore_system.count} fluorophores reached the absorbing '
                      f'state. \nNote: absorbing states are not further considered in analysis.')

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
        Note: if transition of interest is energy transfer, the time to transition is only collected from the donor's
        point of view.

        Returns
        -------
        lifetime_distributions : Collection
            Contains 1-D array_like for each entry in transitions.single_states.
        transition_time_distributions : Collection
            Contains 1-D array_like for each entry in transitions.transitions.
        """
        lifetime_distributions = [np.array([]) for _ in self.single_states]
        transition_time_distributions = [np.array([]) for _ in self.transition_df.index]
        for state_series_fluorophore in self.simulation.state_series:
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            changed = changes_at + 1
            initial_single_states = state_series_fluorophore[changes_at]
            total_times = self.simulation.time_series[changed]
            time_intervals = np.diff(total_times)
            time_intervals = np.insert(time_intervals, 0, total_times[0])
            for j, state in enumerate(self.single_states):
                time_intervals_state = time_intervals[np.where(initial_single_states == state)]
                lifetime_distributions[j] = np.concatenate([lifetime_distributions[j], time_intervals_state])

            transitions_fluorophore = self.simulation.transition_series[changes_at]
            for j, index in enumerate(self.transition_df.index):
                combined_state_transition_ids = self.transitions.combined_state_transitions_df[
                    self.transitions.combined_state_transitions_df['transition_id'] == index].index.values
                transition_ids = np.in1d(transitions_fluorophore, combined_state_transition_ids).nonzero()[0]
                if self.transition_df.at[index, 'energy_transfer']:
                    source_donor = self.transition_df.at[index, 'initial_state'].donor.value
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
        state_occurrences = np.zeros(shape=self.single_states.size, dtype=np.int64)
        transition_occurrences = np.zeros(shape=self.transition_df.index.size, dtype=np.int64)

        for state_series_fluorophore in self.simulation.state_series:
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            last_state = changes_at[-1] + 1
            changes_at_and_last = np.append(changes_at, last_state)
            states = state_series_fluorophore[changes_at_and_last]
            state_ids, state_counts = np.unique(states, return_counts=True)
            corresponding_state_indices = np.in1d(self.single_states, state_ids).nonzero()[0]
            keep_indices = np.in1d(state_ids, self.single_states).nonzero()[0]
            state_counts = state_counts[keep_indices]
            state_occurrences[corresponding_state_indices] += state_counts

        for j, index in enumerate(self.transition_df.index):
            combined_state_transition_ids = self.transitions.combined_state_transitions_df[
                self.transitions.combined_state_transitions_df['transition_id'] == index].index.values
            transition_ids = np.in1d(self.simulation.transition_series, combined_state_transition_ids).nonzero()[0]
            transition_occurrences[j] = transition_ids.size

        return state_occurrences, transition_occurrences

    def plot(self, mode='state_occurrences', include=None, prediction=None, **kwargs):
        """
        Plot class attributes.

        Parameters
        ----------
        mode : str
            One of 'state_occurrences', 'transition_occurrences', 'mean_lifetimes', 'mean_transition_times',
            'state_occupations', 'lifetime_distributions', 'transition_time_distributions'.
        include : Collection
            Ids of transitions.single_states or transitions.transitions that should be displayed.
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
        if include is None:
            include = [0]
        marker = None
        x_states = np.arange(self.single_states.size)
        x_transitions = np.arange(self.transition_df.index.size)
        if prediction is not None:
            if ((not prediction.energy_transfer and self.energy_transfer) or 
                (prediction.energy_transfer and not self.energy_transfer)):
                raise AttributeError(f'prediction energy transfers: {prediction.energy_transfer}' +
                                     f' | analysis energy transfers: {self.energy_transfer}' +
                                     ' | mismatch!')
        if 'distribution' not in mode:
            if mode == 'transition_occurrences':
                data = [x_transitions, self.transition_occurrences]
                if prediction is not None:
                    marker = [x_transitions, prediction.stationary_distribution_transitions]
            elif mode == 'state_occurrences':
                data = [x_states, self.state_occurrences]
                if prediction is not None:
                    marker = [x_states, prediction.stationary_distribution_states]
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
            axes = plot_bar(data=data, single_states=self.single_states, df=self.transition_df,
                            mode=mode, draw_marker=marker, **kwargs)
        else:
            plot_distribution = None
            if mode == 'lifetime_distributions':
                labels = [SingleState(identity).name for identity in self.single_states if identity in include]
                indices = [index for index, identity in enumerate(self.single_states) if identity in include]
                data = [distribution for i, distribution in enumerate(self.lifetime_distributions) if
                        i in indices]
                if prediction is not None:
                    plot_distribution = prediction.lifetime_distributions[indices]
            elif mode == 'transition_time_distributions':
                labels = [transition for identity, transition in self.transition_df['abbreviation'].items() if
                          identity in include]
                data = [distribution for identity, distribution in enumerate(self.transition_time_distributions) if
                        identity in include]
                if prediction is not None:
                    plot_distribution = [distribution for identity, distribution in
                                         enumerate(prediction.transition_time_distributions) if identity in include]
            else:
                raise AttributeError(f'mode {mode} unknown.')
            kwargs.setdefault('label', labels)
            axes = plot_analysis_distr(data=data, mode=mode, plot_distribution=plot_distribution,
                                       **kwargs)

        return axes

    def plot_all(self, include_lifetimes=None, include_transitions=None, prediction=None, scale=0.5):
        """
        Plot all class attributes.

        Parameters
        ----------
        include_lifetimes : Collection
            Ids of transitions.single_states whose lifetime_distribution should be displayed.
        include_transitions : Collection
            Ids of transitions.transitions whose transition_time_distribution should be displayed.
        prediction : None, Prediction
            Container of lifetimes, state and transition occurrences obtained by computation-associated attributes and
            methods.
        scale : float
            Factor to scale the figure.

        Returns
        -------
        axes : np.ndarray
            Contains matplotlib.axes._subplots.AxesSubplots.
        """
        if prediction is not None:
            prediction_1 = prediction
            if prediction.energy_transfer:
                prediction_2 = None
            else:
                prediction_2 = prediction
        else:
            prediction_1 = None
            prediction_2 = None
        axes = self.plot(mode='state_occurrences', ncols=4, nrows=2, fig_width=20, fig_height=6,
                         prediction=prediction_1, scale=scale)
        _ = self.plot(mode='mean_lifetimes', axes=axes[0, 1], prediction=prediction_2)
        _ = self.plot(mode='lifetime_distributions', axes=axes[0, 2], include=include_lifetimes,
                      prediction=prediction_2)
        _ = self.plot(mode='state_occupations', axes=axes[0, 3], prediction=prediction_2)
        _ = self.plot(mode='transition_occurrences', axes=axes[1, 0], prediction=prediction_1)
        _ = self.plot(mode='mean_transition_times', axes=axes[1, 1], prediction=prediction_2)
        _ = self.plot(mode='transition_time_distributions', axes=axes[1, 2],
                      include=include_transitions, prediction=prediction_2)
        mi.delete_subplots(axes=axes, keep_number=7)
        axes[0, 0].get_figure().tight_layout()

        return axes


def get_fluorescence_lifetimes(simulation):
    """
    Get the simulated fluorescence lifetimes (lifetimes of S1).

    Parameters
    ----------
    simulation : src.simulation.Simulation
        Container for simulation-associated attributes.
        
    Returns
    -------
    lifetimes : 1-D array_like
        Contains the fluorescence lifetimes.
    """
    lifetimes = np.array([])
    for state_series_fluorophore in simulation.state_series:
        differences = np.diff(state_series_fluorophore)
        changes_at = np.where(differences != 0)[0]
        changed = changes_at + 1
        initial_single_states = state_series_fluorophore[changes_at]
        total_times = simulation.time_series[changed]
        time_intervals = np.diff(total_times)
        time_intervals = np.insert(time_intervals, 0, total_times[0])
        s1 = SingleState.S1.value
        time_intervals_s1 = time_intervals[np.where(initial_single_states == s1)]
        lifetimes = np.concatenate([lifetimes, time_intervals_s1])
    return lifetimes


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
        kwargs.setdefault('xticklabels', dict(labels=[SingleState(identity).name for identity in single_states], rotation=70))
    if mode == 'state_occurrences':
        kwargs.setdefault('ylabel', 'PR')
        kwargs.setdefault('title', 'occurrences')
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
