from scipy.stats import expon
import numpy as np
import src.custom_plot as cp
from matplotlib.pyplot import cm
from src.NEW_transitions import SingleState
import src.NEW_miscellaneous as mi


class Statistics:
    def __init__(self):
        self.prediction = None
        self.analysis = None

    def predict(self, transitions):
        self.prediction = Prediction(transitions)

    def analyze(self, simulation, transitions):
        self.analysis = Analysis(simulation, transitions)


class Prediction:
    def __init__(self, transitions):
        if (transitions.transition_df['energy_transfer'] == True).any():
            raise ValueError('prediction not available if energy transfers can occur.')
        else:
            self.lifetime_distributions, self.transition_time_distributions = \
                self.predict_lifetimes(transitions.single_states, transitions.transitions)
            self.mean_lifetimes = np.array([distr.mean() for distr in self.lifetime_distributions])
            self.mean_transition_times = np.array([distr.mean() for distr in self.transition_time_distributions])

            self.state_occurrences = self.predict_state_occurrences(transitions.single_states,
                                                                    transitions.transition_df)
            self.transition_occurrences = self.predict_transition_occurrences(transitions.single_states,
                                                                              transitions.transition_df)
            self.relative_total_times = self.state_occurrences * self.mean_lifetimes

    @staticmethod
    def predict_lifetimes(single_states, transitions):

        lifetime_distributions = np.empty(single_states.size, dtype=np.object)
        transition_time_distributions = np.empty(len(transitions), dtype=np.object)

        for i, state in enumerate(single_states):
            total_rate = 0
            associated_transitions = []
            for j, transition in enumerate(transitions):
                source = transition.initial_state.value
                if source == state:
                    total_rate += transition.rate
                    associated_transitions.append(j)

            lifetime_mean = 1 / total_rate

            lifetime_pdf = expon(scale=lifetime_mean)
            lifetime_distributions[i] = lifetime_pdf
            transition_time_distributions[associated_transitions] = lifetime_pdf

        return lifetime_distributions, transition_time_distributions

    @staticmethod
    def predict_state_occurrences(single_states, transition_df):

        i_s0 = np.where(single_states == SingleState.S0.value)[0]
        i_s1 = np.where(single_states == SingleState.S1.value)[0]
        i_t1 = np.where(single_states == SingleState.T1.value)[0]
        i_cis = np.where(single_states == SingleState.Cis.value)[0]
        i_off = np.where(single_states == SingleState.OFF.value)[0]

        b = np.zeros(shape=single_states.size)
        b[0] = 1

        # sum of all states has to equal number of steps
        a_arrays = np.zeros(shape=(single_states.size, single_states.size))
        a_arrays[0][:] = 1

        # S1 has to occur as often as S0
        a_arrays[1][i_s0] = 1
        a_arrays[1][i_s1] = -1

        if single_states.size > 2:
            s1_transitions = transition_df[transition_df['initial_state'] == SingleState.S1]
            s1_direct_deexcitation = s1_transitions[s1_transitions['final_state'] == SingleState.S0]
            s1_indirect_deexcitation = s1_transitions[s1_transitions['final_state'] != SingleState.S0]

            t1_transitions = transition_df[transition_df['initial_state'] == SingleState.T1]
            t1_direct_deexcitation = t1_transitions[t1_transitions['final_state'] == SingleState.S0]
            t1_indirect_deexcitation = t1_transitions[t1_transitions['final_state'] != SingleState.S0]

            # T1 coverage
            s1_nuniques = s1_indirect_deexcitation['abbreviation'].nunique()
            a_arrays[2][i_s0] = -1
            a_arrays[2][i_t1] = s1_direct_deexcitation['rate'].sum() / s1_indirect_deexcitation['rate'].sum() + 1
            # + 1 adds the population of the ith state

            if s1_nuniques == 2:
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
                a_arrays[i_off][i_off] = -1
                a_arrays[i_off][i_t1] = t1_indirect_deexcitation['rate'].sum() / (t1_direct_deexcitation['rate'].sum() +
                                                                                  t1_indirect_deexcitation[
                                                                                      'rate'].sum())

            elif t1_nuniques > 1:
                raise ValueError('Only one alternative triplet deexcitation pathway implemented.')

        state_occurrences = np.linalg.solve(a_arrays, b)

        return state_occurrences

    def predict_transition_occurrences(self, single_states, transition_df):

        i_s0 = np.where(single_states == SingleState.S0.value)[0]
        i_s1 = np.where(single_states == SingleState.S1.value)[0]
        i_t1 = np.where(single_states == SingleState.T1.value)[0]
        i_cis = np.where(single_states == SingleState.Cis.value)[0]
        i_off = np.where(single_states == SingleState.OFF.value)[0]

        transition_occurrences = np.zeros(transition_df.index.size)
        for index, row in transition_df.iterrows():
            source = row['initial_state']
            if source == SingleState.S0:
                i = i_s0
            elif source == SingleState.S1:
                i = i_s1
            elif source == SingleState.T1:
                i = i_t1
            elif source == SingleState.Cis:
                i = i_cis
            elif source == SingleState.OFF:
                i = i_off
            else:
                raise ValueError

            state_occurrence = self.state_occurrences[i]
            total_rate = 1 / self.lifetime_distributions[i][0].mean()  # the extra [0] is needed because np.where
            # returns array, hence the indexed array returns an array[element] and not the element
            current_rate = row['rate']
            transition_occurrence = state_occurrence * current_rate / total_rate
            transition_occurrences[index] = transition_occurrence

        return transition_occurrences

    def plot(self, transitions, mode, x=None, exclude=None, **kwargs):
        if exclude is None:
            exclude = []
        if 'distribution' not in mode:
            if mode == 'transition_occurrences':
                data = [np.arange(transitions.transition_df.index.size), self.transition_occurrences]
            elif mode == 'state_occurrences':
                data = [np.arange(transitions.single_states.size), self.state_occurrences]
            elif mode == 'mean_lifetimes':
                data = [np.arange(transitions.single_states.size), self.mean_lifetimes]
            elif mode == 'mean_transition_times':
                data = [np.arange(transitions.transition_df.index.size), self.mean_transition_times]
            elif mode == 'relative_total_times':
                data = [np.arange(transitions.single_states.size), self.relative_total_times]
            else:
                raise AttributeError(f'mode {mode} unknown.')
            fig, ax = plot_bar(transitions, mode=mode, data=data, **kwargs)
        else:
            if mode == 'lifetime_distributions':
                labels = [SingleState(i).name for i in transitions.single_states if i not in exclude]
                x = np.linspace(0, 1, 1000) if x is None else x
                data = [[x, distribution.pdf(x)] for i, distribution in enumerate(self.lifetime_distributions) if
                        i not in exclude]
            elif mode == 'transition_time_distributions':
                labels = [transition for i, transition in transitions.transition_df['abbreviation'].iteritems() if
                          i not in exclude]
                x = np.linspace(0, 1, 1000) if x is None else x
                data = [[x, distribution.pdf(x)] for i, distribution in enumerate(self.transition_time_distributions) if
                        i not in exclude]
            else:
                raise AttributeError(f'mode {mode} unknown.')
            fig, ax = plot_prediction_distr(mode=mode, data=data, label=labels, **kwargs)

        return fig, ax

    def plot_all(self, transitions, x_lifetimes=None, x_transitions=None, exclude_lifetimes=None,
                 exclude_transitions=None):
        fig, ax = self.plot(transitions, mode='state_occurrences', ncols=4, nrows=2, fig_width=20, fig_height=6,
                            scale=0.5)
        _, _ = self.plot(transitions, mode='mean_lifetimes', fig=fig, axes=ax[0, 1])
        _, _ = self.plot(transitions, mode='lifetime_distributions', fig=fig, axes=ax[0, 2], x=x_lifetimes,
                         exclude=exclude_lifetimes)
        _, _ = self.plot(transitions, mode='relative_total_times', fig=fig, axes=ax[0, 3])
        _, _ = self.plot(transitions, mode='transition_occurrences', fig=fig, axes=ax[1, 0])
        _, _ = self.plot(transitions, mode='mean_transition_times', fig=fig, axes=ax[1, 1])
        _, _ = self.plot(transitions, mode='transition_time_distributions', fig=fig, axes=ax[1, 2], x=x_transitions,
                         exclude=exclude_transitions)
        mi.delete_subplots(fig=fig, ax=ax, keep_number=7)
        fig.tight_layout()


class Analysis:
    def __init__(self, simulation, transitions):
        if simulation.transition_series is None:
            raise ValueError('analysis not available if simulation were not ran.')
        else:
            self.lifetime_distributions, self.transition_time_distributions = \
                self.get_lifetimes(simulation, transitions)
            self.mean_lifetimes = np.array([np.mean(lifetime_distribution) for lifetime_distribution in
                                            self.lifetime_distributions])
            self.mean_transition_times = np.array([np.mean(transition_time_distribution) for
                                                   transition_time_distribution in self.transition_time_distributions])
            total_times = np.array([np.sum(lifetime_distribution) for lifetime_distribution in
                                    self.lifetime_distributions])
            self.relative_total_times = total_times / simulation.transition_series.size
            state_occurrences, transition_occurrences = self.get_occurrences(simulation, transitions)
            self.state_occurrences = state_occurrences / np.sum(state_occurrences)
            self.transition_occurrences = transition_occurrences / np.sum(transition_occurrences)

    @staticmethod
    def get_lifetimes(simulation, transitions):
        lifetime_distributions = [np.array([]) for _ in transitions.single_states]
        transition_time_distributions = [np.array([]) for _ in transitions.transition_df.index]
        for i, state_series_fluorophore in enumerate(simulation.state_series):
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            changed = changes_at + 1
            initial_single_states = state_series_fluorophore[changes_at]
            total_times = simulation.time_series[changed]
            time_intervals = np.diff(total_times)
            time_intervals = np.insert(time_intervals, 0, total_times[0])
            for j, state in enumerate(transitions.single_states):
                time_intervals_state = time_intervals[np.where(initial_single_states == state)]
                lifetime_distributions[j] = np.concatenate([lifetime_distributions[j], time_intervals_state])

            transitions_fluorophore = simulation.transition_series[changes_at]
            for j in transitions.transition_df.index:
                combined_state_transition_ids = transitions.combined_state_transitions_df[
                    transitions.combined_state_transitions_df['transition_id'] == j].index.values
                transition_ids = np.in1d(transitions_fluorophore, combined_state_transition_ids).nonzero()[0]
                time_intervals_transition = time_intervals[transition_ids]
                transition_time_distributions[j] = np.concatenate([transition_time_distributions[j],
                                                                   time_intervals_transition])

        return lifetime_distributions, transition_time_distributions

    @staticmethod
    def get_occurrences(simulation, transitions):
        state_occurrences = np.empty(shape=transitions.single_states.size, dtype=np.int64)
        transition_occurrences = np.empty(shape=transitions.transition_df.index.size, dtype=np.int64)

        for state_series_fluorophore in simulation.state_series:
            differences = np.diff(state_series_fluorophore)
            changes_at = np.where(differences != 0)[0]
            last_state = changes_at[-1] + 1
            changes_at_and_last = np.append(changes_at, last_state)
            states = state_series_fluorophore[changes_at_and_last]
            state_ids, state_counts = np.unique(states, return_counts=True)
            state_occurrences[state_ids] = state_counts

        for j in transitions.transition_df.index:
            combined_state_transition_ids = transitions.combined_state_transitions_df[
                transitions.combined_state_transitions_df['transition_id'] == j].index.values
            transition_ids = np.in1d(simulation.transition_series, combined_state_transition_ids).nonzero()[0]
            transition_occurrences[j] = transition_ids.size

        return state_occurrences, transition_occurrences

    def plot(self, transitions, mode, exclude=None, prediction=None, **kwargs):
        if exclude is None:
            exclude = []
        marker = None
        x_states = np.arange(transitions.single_states.size)
        x_transitions = np.arange(transitions.transition_df.index.size)
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
                data = [x_states, self.mean_lifetimes]
                if prediction is not None:
                    marker = [x_states, prediction.mean_lifetimes]
            elif mode == 'mean_transition_times':
                data = [x_transitions, self.mean_transition_times]
                if prediction is not None:
                    marker = [x_transitions, prediction.mean_transition_times]
            elif mode == 'relative_total_times':
                data = [x_states, self.relative_total_times]
                if prediction is not None:
                    marker = [x_states, prediction.relative_total_times]
            else:
                raise AttributeError(f'mode {mode} unknown.')
            fig, ax = plot_bar(transitions, mode=mode, data=data, draw_marker=marker, **kwargs)
        else:
            plot_distribution = None
            if mode == 'lifetime_distributions':
                labels = [SingleState(i).name for i in transitions.single_states if i not in exclude]
                data = [distribution for i, distribution in enumerate(self.lifetime_distributions) if
                        i not in exclude]
                if prediction is not None:
                    plot_distribution = prediction.lifetime_distributions
            elif mode == 'transition_time_distributions':
                labels = [transition for i, transition in transitions.transition_df['abbreviation'].iteritems() if
                          i not in exclude]
                data = [distribution for i, distribution in enumerate(self.transition_time_distributions) if
                        i not in exclude]
                if prediction is not None:
                    plot_distribution = prediction.transition_time_distributions
            else:
                raise AttributeError(f'mode {mode} unknown.')

            fig, ax = plot_analysis_distr(mode=mode, data=data, label=labels, plot_distribution=plot_distribution,
                                          **kwargs)

        return fig, ax

    def plot_all(self, transitions, exclude_lifetimes=None, exclude_transitions=None, prediction=None):
        fig, ax = self.plot(transitions, mode='state_occurrences', ncols=4, nrows=2, fig_width=20, fig_height=6,
                            scale=0.5, prediction=prediction)
        _, _ = self.plot(transitions, mode='mean_lifetimes', fig=fig, axes=ax[0, 1], prediction=prediction)
        _, _ = self.plot(transitions, mode='lifetime_distributions', fig=fig, axes=ax[0, 2], exclude=exclude_lifetimes,
                         prediction=prediction)
        _, _ = self.plot(transitions, mode='relative_total_times', fig=fig, axes=ax[0, 3], prediction=prediction)
        _, _ = self.plot(transitions, mode='transition_occurrences', fig=fig, axes=ax[1, 0], prediction=prediction)
        _, _ = self.plot(transitions, mode='mean_transition_times', fig=fig, axes=ax[1, 1], prediction=prediction)
        _, _ = self.plot(transitions, mode='transition_time_distributions', fig=fig, axes=ax[1, 2],
                         exclude=exclude_transitions, prediction=prediction)
        mi.delete_subplots(fig=fig, ax=ax, keep_number=7)
        fig.tight_layout()


def plot_bar(transitions, mode, data, **kwargs):
    from src.NEW_transitions import SingleState
    kwargs.setdefault('type_', 'bar')
    kwargs.setdefault('xlabel', None)
    kwargs.setdefault('yscale', 'log')
    kwargs.setdefault('edgecolor', 'black')
    if 'transition' in mode:
        kwargs.setdefault('xticks', range(transitions.transition_df.index.size))
        kwargs.setdefault('xticklabels', dict(labels=transitions.transition_df['abbreviation'], rotation=70))
    else:
        kwargs.setdefault('xticks', range(transitions.single_states.size))
        kwargs.setdefault('xticklabels', dict(labels=[SingleState(i).name for i in transitions.single_states],
                                              rotation=70))
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
    elif mode == 'relative_total_times':
        kwargs.setdefault('ylabel', 'duration per event [s]')
        kwargs.setdefault('title', 'occupation time')

    fig, ax = cp.universal_figure(data=data, **kwargs)

    return fig, ax


def plot_prediction_distr(mode, data, **kwargs):
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

    fig, ax = cp.universal_figure(data=data, **kwargs)

    return fig, ax


def plot_analysis_distr(mode, data, **kwargs):
    kwargs.setdefault('type_', 'multiple_hist')
    kwargs.setdefault('ylabel', 'PD')
    kwargs.setdefault('legend', True)
    kwargs.setdefault('yscale', 'log')
    kwargs.setdefault('density', True)
    colors = cm.get_cmap('rainbow', len(data))
    kwargs.setdefault('color', colors)
    if mode == 'lifetime_distributions':
        kwargs.setdefault('xlabel', 'lifetime [s]')
    elif mode == 'transition_time_distribution':
        kwargs.setdefault('xlabel', 'time to transition [s]')

    fig, ax = cp.universal_figure(data=data, **kwargs)

    return fig, ax