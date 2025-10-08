"""
Run photophysical simulations.
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import iteround as it
import numpy as np
import numpy.typing as npt
import pandas as pd

from . import kappa_squared as kappa_sq
from . import network as net

if TYPE_CHECKING:
    from .fluopy_types import RandomGeneratorSeed
    from .prediction import Prediction
    from .transitions import TransitionSet


__all__: list[str] = ["Simulation"]

logger = logging.getLogger(__name__)


class Simulation:
    """
    Container of simulation-associated attributes and methods.

    Attributes
    ----------
    transition_set : fluopy.transitions.TransitionSet
        Collection of all relevant transitions and related attributes.
    time_series : 1-D array_like
        The simulated time points. At index i, they correspond to state_series[i] and
        transition_series[i - 1]. If end_time was not None, includes end_time at index
        -1 that does not correspond to any of state_series or transition_series.
    transition_series : 1-D array_like
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    state_series : np.ndarray
        Contains 1-D array_like for each fluorophore representing its state at index i
        corresponding to time_series[i].
    memmap_path : str
        The path where memmaps are stored.
    """

    def __init__(self, transition_set: TransitionSet) -> None:
        """
        Set up a simulation

        Parameters
        ----------
        transition_set
            Collection of all relevant transitions and related attributes.

        Returns
        -------
        None
        """
        self.transition_set = transition_set
        self.time_series = None
        self.transition_series = None
        self.state_series = None
        self.memmap_path = None

    def run(
        self,
        start_at: tuple[float, ...] | None = None,
        size: int = 1e5,
        end_time: float | None = None,
        kap_sq_var: bool = False,
        seed: RandomGeneratorSeed = None,
        use_memmap: str | Path | None = None,
        **kwargs: Any,
    ) -> None | npt.NDArray[np.float64]:
        """
        Runs a simulation based on the direct method of the gillespie algorithm (i.e.,
        stochastic simulation algorithm). Can either be based on maximum number of
        steps or maximum total time.

        Parameters
        ----------
        start_at
            If None, tuple of as many zeros as number of fluorophores.
            Can be any combination (size of number of fluorophores) of possible
            SingleState values. See transition_set.single_states.
        size
            If end_time is None, serves as maximum number of simulation steps.
            If end_time is not None, serves as size of random_numbers drawn at once.
        end_time
            If not None, time at which simulation ends in s.
        kap_sq_var
            If True, the first reaction method is used to simulate the data. This takes
            much longer but allows to vary the dipole orientation factor for different
            S1 states. If False, the direct method is used.
        seed
            A seed to initialize the BitGenerator.
        use_memmap
            Determines the path where memmaps shall be stored. If empty str, saved in
            current working directory.
        kwargs
            First reaction method arguments.

        Returns
        -------
        None
        """
        if start_at is None:
            start_at = tuple(
                np.zeros(shape=self.transition_set.fluorophore_system.count, dtype=int)
            )
        elif len(start_at) != self.transition_set.fluorophore_system.count:
            raise ValueError(
                "The number of starting states doesn't match the number of "
                "fluorophores."
            )
        size = int(size)
        df = self.transition_set.combined_state_transitions_df
        start_index = df[df["final_state"] == start_at].index[0]
        eval_floating_point_precision_error(
            transition_set=self.transition_set, largest_number=end_time
        )
        if end_time is None and not kap_sq_var:
            self.time_series, self.transition_series = direct_method_steps(
                transition_matrix=self.transition_set.transition_matrix,
                row_sums=self.transition_set.row_sums,
                start_index=start_index,
                size=size,
                seed=seed,
                use_memmap=use_memmap,
            )
        elif end_time is None and kap_sq_var:
            self.time_series, self.transition_series = first_reaction_method(
                transition_matrix=self.transition_set.transition_matrix,
                row_sums=self.transition_set.row_sums,
                combined_state_transitions_df=df,
                start_index=start_index,
                size=size,
                seed=seed,
                use_memmap=use_memmap,
                **kwargs,
            )
        elif end_time is not None and not kap_sq_var:
            self.time_series, self.transition_series = direct_method_time(
                transition_matrix=self.transition_set.transition_matrix,
                row_sums=self.transition_set.row_sums,
                start_index=start_index,
                size=size,
                end_time=end_time,
                seed=seed,
                use_memmap=use_memmap,
            )
        else:
            raise ValueError(
                "end_time is not None but kap_sq_var is True. Not implemented.",
            )

        final_states = self.transition_set.combined_state_transitions_df["final_state"]

        if use_memmap is not None:
            self.memmap_path = use_memmap
            self.state_series = np.memmap(
                Path(use_memmap) / "state_series",
                dtype=np.int8,
                mode="w+",
                shape=(len(final_states[0]), self.transition_series.size + 1),
            )
        else:
            self.state_series = np.empty(
                shape=(len(final_states[0]), self.transition_series.size + 1),
                dtype=np.int8,
            )
        self.state_series[:, 0] = start_at

        for i, _ in enumerate(final_states[0]):
            final_states_fluorophore = final_states.map(lambda x, i=i: x[i]).to_numpy(
                dtype=np.int8
            )
            self.state_series[i][1:] = final_states_fluorophore[self.transition_series]
        if use_memmap is not None:
            self.state_series.flush()

    def approximate(
        self, prediction: Prediction, size: float, seed: RandomGeneratorSeed
    ) -> None:
        """
        Approximates stochastic data based on the limiting distribution of a Markov
        chain. Only suitable for single fluorophore systems. Absorbing states are not
        considered. Each simple cycle should contain the most occurring state.

        Parameters
        ----------
        prediction
            Container of mathematically derived statistical attributes and methods.
        size
            Maximum number of steps. Due to rounding, actual size might vary.
        seed
            A seed to initialize the BitGenerator.

        Returns
        -------
        None
        """
        if self.transition_set is not prediction.transition_set:
            raise ValueError(
                "prediction is based on different transition_set than simulation."
            )
        if self.transition_set.fluorophore_system.count != 1:
            raise ValueError(
                "approximation only available to single fluorophore systems."
            )
        if prediction.absorbing_chain:
            logger.warning(
                "approximation ignors absorbing states, they will not occur.",
                stacklevel=2,
            )
        eval_floating_point_precision_error(
            transition_set=self.transition_set, largest_number=None
        )
        self.time_series, self.transition_series = approximation(
            prediction=prediction, size=size, seed=seed
        )
        final_states = self.transition_set.transition_df["final_state"].apply(
            lambda x: x.value
        )
        self.state_series = np.empty(
            shape=(self.transition_series.size + 1), dtype=np.int8
        )
        self.state_series[0] = (
            self.transition_set.transition_df["initial_state"].iloc[
                self.transition_series[0]
            ]
        ).value
        self.state_series[1:] = final_states.iloc[self.transition_series]
        self.state_series = np.expand_dims(self.state_series, axis=0)

    def delete_memmaps(self) -> None:
        """
        Delete the memmap variables and files. Note: if the memmaps are attempted to be
        accessed after deletion, python crashes.
        Source: https://stackoverflow.com/questions/39953501/i-cant-remove-file-created-
        by-memmap

        Returns
        -------
        None
        """
        self.transition_series._mmap.close()
        self.time_series._mmap.close()
        self.state_series._mmap.close()
        del self.transition_series
        del self.time_series
        del self.state_series
        (Path(self.memmap_path) / "transition_series").unlink(missing_ok=True)
        (Path(self.memmap_path) / "time_series").unlink(missing_ok=True)
        (Path(self.memmap_path) / "state_series").unlink(missing_ok=True)


def direct_method_steps(
    transition_matrix: npt.ArrayLike,
    row_sums: npt.ArrayLike,
    start_index: int = 0,
    size: int = 10,
    seed: RandomGeneratorSeed = None,
    use_memmap: str | Path | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    The direct method of the gillespie algorithm (i.e., stochastic simulation #
    algorithm). Here, the propensities are equal to the rate constants because the
    population is always 1. Additionally, the state change vector is redundant because
    each transition leads to a shift in populations by 1. This version is based on a
    maximum number of steps.

    Parameters
    ----------
    transition_matrix
        Contains the normalized rate constants (i.e., point probabilities) for each
        possible combined_state_transition at the corresponding index pair.
    row_sums
        Contains the sum of each row of non-normalized transition rates, i.e., the sum
        of rates of all possible combined_state_transitions.
    start_index
        Starting index. The final state of the transition indexed is the starting state
        configuration.
    size
        Maximum number of simulation steps.
    seed
        A seed to initialize the BitGenerator.
    use_memmap
        Determines the path where memmaps shall be stored. If empty str, saved in
        current working directory.

    Returns
    -------
    time_series : npt.NDArray[np.float64]
        The simulated time points. At index i, they correspond to
        transition_series[i - 1].
    transition_series : npt.NDArray[np.int64]
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    """
    rng = np.random.default_rng(seed)

    if use_memmap is not None:
        time_step_series = np.memmap(
            Path(use_memmap) / "time_step_series",
            dtype=np.float32,
            mode="w+",
            shape=(size + 1,),
        )
        transition_series = np.memmap(
            Path(use_memmap) / "transition_series",
            dtype=np.uint32,
            mode="w+",
            shape=(size,),
        )
        time_series = np.memmap(
            Path(use_memmap) / "time_series",
            dtype=np.float64,
            mode="w+",
            shape=(size + 1,),
        )
    else:
        time_step_series = np.empty(size + 1, dtype=np.float32)
        transition_series = np.empty(size, dtype=np.uint32)
        time_series = np.empty(size + 1, dtype=np.float64)
    random_numbers = rng.uniform(
        low=0, high=1, size=(size, 2)
    )  # never a memmap, is initialized on RAM anyways
    time_step_series[0] = 0

    # a random index (in this case the first index) at which the final state of a
    # transition equals start_at
    current_state_index = start_index

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(
        arr=transition_matrix, indices=transition_matrix_sorted_indices, axis=1
    )
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)
    absorbing_state_reached = False

    for i in range(size):
        current_state_lambda = row_sums[current_state_index]

        if current_state_lambda == 0:  # there is no outgoing transition
            # the Markov chain has encountered an absorbing state
            absorbing_state_reached = True
            break
        # inverse transform sampling using the quantile function of the exponential
        # distribution
        transition_time = 1 / current_state_lambda * np.log(1 / random_numbers[i, 0])

        sorted_index = np.searchsorted(
            cumsum_sorted_trm[current_state_index], random_numbers[i, 1]
        )

        next_transition = transition_matrix_sorted_indices[
            current_state_index, sorted_index
        ]

        transition_series[i] = current_state_index = next_transition

        time_step_series[i + 1] = transition_time

    if absorbing_state_reached:
        if use_memmap is not None:
            time_step_series.flush()
            transition_series.flush()
            time_series.flush()
            time_step_series = np.memmap(
                Path(use_memmap) / "time_step_series",
                mode="r+",
                dtype=np.float32,
                shape=(i + 1,),
            )
            transition_series = np.memmap(
                Path(use_memmap) / "transition_series",
                mode="r+",
                dtype=np.uint32,
                shape=(i,),
            )
            time_series = np.memmap(
                Path(use_memmap) / "time_series",
                dtype=np.float64,
                mode="r+",
                shape=(i + 1,),
            )
        else:
            time_step_series.resize((i + 1,))
            transition_series.resize((i,))
            time_series.resize((i + 1,))

    time_series[:] = np.cumsum(time_step_series, dtype=np.float64)

    if use_memmap is not None:
        del time_step_series
        gc.collect()
        (Path(use_memmap) / "time_step_series").unlink(missing_ok=True)
        transition_series.flush()
        time_series.flush()

    return time_series, transition_series


def direct_method_time(
    transition_matrix: npt.ArrayLike,
    row_sums: npt.ArrayLike,
    start_index: int = 0,
    size: int = 10,
    end_time: float = 10,
    seed: RandomGeneratorSeed = None,
    use_memmap: str | Path | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    The direct method of the gillespie algorithm (i.e., stochastic simulation
    algorithm). Here, the propensities are equal to the rate constants because the
    population is always 1. Additionally, the state change vector is redundant because
    each transition leads to a shift in populations by 1. This version is based on a
    maximum total time.

    Parameters
    ----------
    transition_matrix
        Contains the normalized rate constants (i.e., point probabilities) for each
        possible combined_state_transition at the corresponding index pair.
    row_sums
        Contains the sum of each row of non-normalized transition rates, i.e., the sum
        of rates of all possible combined_state_transitions.
    start_index
        Starting index. The final state of the transition indexed is the starting state
        configuration.
    size
        Size of random_numbers drawn at once.
    end_time
        Time at which simulation ends in s.
    seed
        A seed to initialize the BitGenerator.
    use_memmap
        Determines the path where memmaps shall be stored. If empty str, saved in
        current working directory.

    Returns
    -------
    time_series : npt.NDArray[np.float64]
        The simulated time points. At index i, they correspond to
        transition_series[i - 1]. Includes end_time at index -1 that does not
        correspond to any of transition_series.
    transition_series : npt.NDArray[np.int64]
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    """
    rng = np.random.default_rng(seed)

    current_state_index = start_index

    if use_memmap is not None:
        transition_series = np.memmap(
            Path(use_memmap) / "transition_series",
            dtype=np.uint32,
            mode="w+",
            shape=(size,),
        )
        time_series = np.memmap(
            Path(use_memmap) / "time_series",
            dtype=np.float64,
            mode="w+",
            shape=(size + 1,),
        )
    else:
        transition_series = np.empty(size, dtype=np.uint32)
        time_series = np.empty(size + 1, dtype=np.float64)
        # time_series size + 1 and not size + 2 because random_numbers (and
        # transition_series) has size transition_series then 'loses' one transition, so
        # its final size will be not higher than size - 1
    random_numbers = rng.uniform(
        low=0, high=1, size=(size, 2)
    )  # never a memmap, is initialized on RAM anyways

    time_series[0] = 0
    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(
        arr=transition_matrix, indices=transition_matrix_sorted_indices, axis=1
    )
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    abso = 0
    i = 0
    j = 1
    while time_series[i] < end_time:
        current_state_lambda = row_sums[current_state_index]

        if current_state_lambda == 0:
            abso = 1
            break
        transition_time = (
            1 / current_state_lambda * np.log(1 / random_numbers[i - (j - 1) * size, 0])
        )

        sorted_index = np.searchsorted(
            cumsum_sorted_trm[current_state_index],
            random_numbers[i - (j - 1) * size, 1],
        )

        next_transition = transition_matrix_sorted_indices[
            current_state_index, sorted_index
        ]

        transition_series[i] = next_transition
        time_series[i + 1] = time_series[i] + transition_time
        current_state_index = next_transition

        i += 1
        if i == j * size:
            j += 1
            random_numbers = rng.uniform(low=0, high=1, size=(size, 2))
            if use_memmap is not None:
                time_series.flush()
                transition_series.flush()
                time_series = np.memmap(
                    Path(use_memmap) / "time_series",
                    mode="r+",
                    dtype=np.float64,
                    shape=(j * size + 1,),
                )
                transition_series = np.memmap(
                    Path(use_memmap) / "transition_series",
                    mode="r+",
                    dtype=np.uint32,
                    shape=(j * size,),
                )
            else:
                time_series.resize((j * size + 1,))
                transition_series.resize((j * size,))

    time_series[i + abso] = end_time

    if use_memmap is not None:
        time_series.flush()
        transition_series.flush()
        time_series.base.resize(
            8 * (i + 1 + abso)
        )  # 8 because it is 64/8 byte per number and resize works with byte
        transition_series.base.resize(4 * (i - 1 + abso))
        time_series.flush()
        transition_series.flush()
        time_series = np.memmap(
            Path(use_memmap) / "time_series",
            mode="r+",
            dtype=np.float64,
            shape=(i + 1 + abso,),
        )
        transition_series = np.memmap(
            Path(use_memmap) / "transition_series",
            mode="r+",
            dtype=np.uint32,
            shape=(i - 1 + abso,),
        )
    else:
        transition_series = transition_series[: i - 1 + abso]
        time_series = time_series[: i + 1 + abso]

    return time_series, transition_series


def first_reaction_method(
    transition_matrix: npt.ArrayLike,
    row_sums: npt.ArrayLike,
    combined_state_transitions_df: pd.DataFrame,
    include_kap_sq: bool = True,
    minimum_rate: float = 0,
    start_index: int = 0,
    size: int = 10,
    seed: RandomGeneratorSeed = None,
    use_memmap: str | Path | None = None,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    The first reaction method of the gillespie algorithm. Here, the propensities are
    equal to the rate constants because the population is always 1. Additionally, the
    state change vector is redundant because each transition leads to a shift in
    populations by 1. This version is based on a maximum number of steps.

    Needed to efficiently incooperate a variable dipole orientation factor
    kappa_squared. The variable kappa_squared is sampled from the static regime.
    The FRET transitions should be given for kappa_squared = 2/3.

    Parameters
    ----------
    transition_matrix
        Contains the normalized rate constants (i.e., point probabilities) for each
        possible combined_state_transition at the corresponding index pair.
    row_sums
        Contains the sum of each row of non-normalized transition rates, i.e., the sum
        of rates of all possible combined_state_transitions.
    combined_state_transitions_df
        Contains realizable combined_state_transitions with their id as index and their
        other attributes as columns.
    include_kap_sq
        If True, the dipole orientation factor kappa_squared is sampled from the static
        regime and used to scale the FRET transition rates.
    minimum_rate
        Minimum rate for a FRET transition to be considered following the static regime.
    start_index
        Starting index. The final state of the transition indexed is the starting state
        configuration.
    size
        Maximum number of simulation steps.
    seed
        A seed to initialize the BitGenerator.
    use_memmap
        Determines the path where memmaps shall be stored. If empty str, saved in
        current working directory.

    Returns
    -------
    time_series : npt.NDArray[np.float64]
        The simulated time points. At index i, they correspond to
        transition_series[i - 1].
    transition_series : npt.NDArray[np.int64]
        The simulated transitions. At index i, they correspond to time_series[i + 1].
    kappa_squared : npt.NDArray[np.float64]
        The sampled kappa_squared values for each step.
    """
    rng = np.random.default_rng(seed)

    if use_memmap is not None:
        time_step_series = np.memmap(
            Path(use_memmap) / "time_step_series",
            dtype=np.float32,
            mode="w+",
            shape=(size + 1,),
        )
        transition_series = np.memmap(
            Path(use_memmap) / "transition_series",
            dtype=np.uint32,
            mode="w+",
            shape=(size,),
        )
        time_series = np.memmap(
            Path(use_memmap) / "time_series",
            dtype=np.float64,
            mode="w+",
            shape=(size + 1,),
        )
    else:
        time_step_series = np.empty(size + 1, dtype=np.float32)
        transition_series = np.empty(size, dtype=np.uint32)
        time_series = np.empty(size + 1, dtype=np.float64)

    row_sums_exp = np.tile(np.expand_dims(row_sums, axis=1), reps=row_sums.size)
    transition_rate_matrix = transition_matrix * row_sums_exp

    time_step_series[0] = 0

    mask = (combined_state_transitions_df["fluorophore_ids"].apply(len) > 1) & (
        combined_state_transitions_df["rate"] > minimum_rate
    )
    fret_indices = combined_state_transitions_df.index[mask]

    if include_kap_sq:
        # static orientation regime
        N = 100000
        d = kappa_sq.random_unit_vector(size=N, seed=rng)
        a = kappa_sq.random_unit_vector(size=N, seed=rng)
        r = kappa_sq.random_unit_vector(size=N, seed=rng)
        k2_values = kappa_sq.kappa_squared(d=d, a=a, r=r)

        # sampling of kappa_squared
        kappa_squared_sampled = kappa_sq.sample_kappa_squared_distribution(
            k2_values=k2_values, size=size, seed=rng
        )

        kappa_squared_ratio = kappa_squared_sampled / (2 / 3)
    else:
        kappa_squared_ratio = np.ones(size)
    current_state_index = start_index
    absorbing_state_reached = False
    for i in range(size):
        if row_sums[current_state_index] == 0:
            # the Markov chain has encountered an absorbing state
            absorbing_state_reached = True
            break
        rates = transition_rate_matrix[current_state_index, :].copy()
        rates[fret_indices] *= kappa_squared_ratio[i]
        non_zero_indices = np.nonzero(rates)[0]
        non_zero_rates = rates[non_zero_indices]
        times = rng.exponential(scale=1 / non_zero_rates)
        min_index = np.argmin(times)
        transition_series[i] = current_state_index = non_zero_indices[min_index]
        transition_time = times[min_index]
        time_step_series[i + 1] = transition_time

    if absorbing_state_reached:
        if use_memmap is not None:
            time_step_series.flush()
            transition_series.flush()
            time_series.flush()
            time_step_series = np.memmap(
                Path(use_memmap) / "time_step_series",
                mode="r+",
                dtype=np.float32,
                shape=(i + 1,),
            )
            transition_series = np.memmap(
                Path(use_memmap) / "transition_series",
                mode="r+",
                dtype=np.uint32,
                shape=(i,),
            )
            time_series = np.memmap(
                Path(use_memmap) / "time_series",
                dtype=np.float64,
                mode="r+",
                shape=(i + 1,),
            )
        else:
            time_step_series.resize((i + 1,))
            transition_series.resize((i,))
            time_series.resize((i + 1,))

    time_series[:] = np.cumsum(time_step_series, dtype=np.float64)

    if use_memmap is not None:
        del time_step_series
        gc.collect()
        (Path(use_memmap) / "time_step_series").unlink(missing_ok=True)
        transition_series.flush()
        time_series.flush()

    return time_series, transition_series


def approximation(
    prediction: Prediction, size: float, seed: RandomGeneratorSeed
) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.int64]]:
    """
    Approximates stochastic data based on the limiting distribution of a Markov chain.
    The transitions are ordered via a topological sort and processed accordingly.
    Successor transitions are placed behind their predecessors. The topological sort is
    possilbe via a temporary conversion of the graph to a directed acyclic graph (DAG).
    Only suitable for single fluorophore systems. Absorbing states are not considered.
    Each simple cycle should contain the most occurring state.

    Parameters
    ----------
    prediction
        Container of mathematically derived statistical attributes and methods.
    size
        Maximum number of steps. Due to rounding, actual size might vary.
    seed
        A seed to initialize the BitGenerator.

    Returns
    -------
    time_series : npt.NDArray[np.float64]
        The simulated (approximated) time points. At index i, they correspond to
        transition_series[i - 1].
    transition_series : npt.NDArray[np.int64]
        The simulated (approximated) transitions. At index i, they correspond to
        time_series[i + 1].
    """
    transition_occurrences = prediction.frequency_transitions * int(size)
    transition_occurrences = transition_occurrences.astype(np.int64)
    maximum_transition_index = np.argmax(transition_occurrences)
    starting_transition = maximum_transition_index
    fluorophore = prediction.transition_set.fluorophore_system.fluorophores[0].name
    G = net.construct_transition_graph(
        transition_df=prediction.transition_set.transition_df
    )
    graph_suited, _ = net.check_graph_suitable(G=G, starting_node=starting_transition)
    if not graph_suited:
        raise ValueError(
            "graph not suited for approximation. Check for loops that do not contain "
            "the most occurring state."
        )
    transition_order = net.determine_node_order(G=G, starting_node=starting_transition)

    rng = np.random.default_rng(seed)
    transition_series = np.full(
        transition_occurrences[maximum_transition_index], fill_value=starting_transition
    )

    for transition in transition_order:
        transition_indices = np.where(transition_series == transition)[0]
        occurrences = transition_indices.size
        rng.shuffle(transition_indices)
        follow_up_transitions = np.array(list(G.successors(transition)))
        follow_up_transitions = follow_up_transitions[
            ~prediction.transition_set.transition_df["absorbing"].loc[
                (fluorophore, follow_up_transitions)
            ]
        ]
        rates = (
            prediction.transition_set.transition_df["rate"]
            .loc[(fluorophore, follow_up_transitions)]
            .to_numpy()
        )
        repeats = rates * occurrences / rates.sum()
        rounded_repeats = it.saferound(
            repeats, places=0, topline=occurrences
        )  # topline such that each transition will get a followup transition
        rounded_repeats = np.array(rounded_repeats, dtype=np.int64)
        if starting_transition in follow_up_transitions:
            indices_starting_transition = np.where(
                follow_up_transitions == starting_transition
            )[0]
            follow_up_transitions = np.delete(
                follow_up_transitions, indices_starting_transition
            )
            number_of_deletions = rounded_repeats[indices_starting_transition].sum()
            rounded_repeats = np.delete(rounded_repeats, indices_starting_transition)
            transition_indices = transition_indices[:-number_of_deletions]
        follow_up_transitions = np.repeat(
            follow_up_transitions, repeats=rounded_repeats
        )
        insert_at = transition_indices + 1
        transition_series = np.insert(
            arr=transition_series, obj=insert_at, values=follow_up_transitions
        )

    time_step_series = np.empty(transition_series.size + 1, dtype=np.float64)
    time_step_series[0] = 0
    for i, transition_time_distribution in enumerate(
        prediction.transition_time_distributions
    ):
        indices = np.where(transition_series == i)[0]
        drawn_lifetimes = transition_time_distribution.rvs(
            indices.size, random_state=rng
        )
        time_step_series[indices + 1] = drawn_lifetimes

    time_series = np.empty_like(time_step_series, dtype=np.float64)
    time_series[:] = np.cumsum(time_step_series, dtype=np.float64)

    return time_series, transition_series


def simulate_experiment(
    transition_matrix: npt.ArrayLike,
    row_sums: npt.ArrayLike,
    emitting_transition_ids: dict[int, float],
    start_index: int = 0,
    size: int = 1e5,
    frames: int = 10,
    frame_time: str = "5ms",
    store_time_points: bool = False,
    seed: RandomGeneratorSeed = None,
) -> tuple[npt.NDArray[np.float64], pd.Series]:
    """
    Simulates experimental data (i.e., number of photons per frame). Methodically the
    direct method of the gillespie algorithm. Stores only the number of photons per
    frame, making it memory-wise computationally easy.

    Parameters
    ----------
    transition_matrix
        Contains the normalized rate constants (i.e., point probabilities) for each
        possible combined_state_transition at the corresponding index pair.
    row_sums
        Contains the sum of each row of non-normalized transition rates, i.e., the sum
        of rates of all possible combined_state_transitions.
    emitting_transition_ids
        Contains the combined_state_transition indices as keys and their probability of
        passing a bandpass filter as values.
    start_index
        Starting index. The final state of the transition indexed is the starting state
        configuration.
    size
        Size of random_numbers drawn at once.
    frames
        Total number of frames to be simulated.
    frame_time
        For possible input values, see
        https://pandas.pydata.org/docs/user_guide/timeseries.html -> Offset aliases.
    store_time_points
        Whether to also create an array which contains the time points at which photons
        are detected.
    seed
        A seed to initialize the BitGenerator.

    Returns
    -------
    event_time_points : npt.NDArray[np.float64]
        The time points at which emissions are detected.
    event_time_series : pd.Series
        Contains the time points (increasing by a defined time interval) as index and
        the number of events (i.e., detected emissions) as values.
    """

    transition_matrix_sorted_indices = np.argsort(transition_matrix, axis=1)
    sorted_transition_matrix = np.take_along_axis(
        arr=transition_matrix, indices=transition_matrix_sorted_indices, axis=1
    )
    cumsum_sorted_trm = np.cumsum(sorted_transition_matrix, axis=1)

    rng = np.random.default_rng(seed)
    current_state_index = start_index

    frame_time = pd.Timedelta(frame_time) / np.timedelta64(1, "s")
    time_stamps = np.linspace(0, frame_time * frames, frames + 1)
    time_stamps = np.round(time_stamps, decimals=12)
    photon_collector = np.zeros(time_stamps.size)
    time = 0
    if store_time_points:
        time_points = []
    else:
        event_time_points = None

    frame = 0
    skip = False
    while frame < frames:
        frame += 1
        photons = 0
        random_numbers = rng.uniform(low=0, high=1, size=(size, 3))
        i = 0
        j = 1
        while time < frame_time:
            if not skip:
                current_state_lambda = row_sums[current_state_index]

                if current_state_lambda == 0:
                    photon_collector[frame] = photons
                    event_time_series = pd.Series(
                        photon_collector, index=time_stamps, dtype=np.int64
                    )
                    logger.warning(
                        "All fluorophores underwent photobleaching or entered "
                        "another Markov chain absorbing state.",
                        stacklevel=2,
                    )
                    if store_time_points:
                        event_time_points = np.array(time_points)
                    return event_time_points, event_time_series

                transition_time = (
                    1
                    / current_state_lambda
                    * np.log(1 / random_numbers[i - (j - 1) * size, 0])
                )
                time += transition_time
                if time > frame_time:
                    frame_diff = int(np.floor(time / frame_time)) - 1
                    time -= (frame_diff + 1) * frame_time
                    skip = True
                    break

            skip = False
            sorted_index = np.searchsorted(
                cumsum_sorted_trm[current_state_index],
                random_numbers[i - (j - 1) * size, 1],
            )
            next_transition = transition_matrix_sorted_indices[
                current_state_index, sorted_index
            ]
            if next_transition in emitting_transition_ids:
                if (
                    random_numbers[i - (j - 1) * size, 2]
                    < emitting_transition_ids[next_transition]
                ):
                    photons += 1
                    if store_time_points:
                        time_points.append(frame * frame_time + time)

            current_state_index = next_transition
            i += 1
            if i == j * size:
                j += 1
                random_numbers = rng.uniform(low=0, high=1, size=(size, 3))

        photon_collector[frame] = photons
        frame += frame_diff

    event_time_series = pd.Series(photon_collector, index=time_stamps, dtype=np.int64)
    if store_time_points:
        event_time_points = np.array(time_points)

    return event_time_points, event_time_series


def eval_floating_point_precision_error(
    transition_set: TransitionSet, largest_number: float | None = None
) -> None:
    """
    Evaluates the floating point precision error of the transition_set. The larger the
    rates, the more significant the error.

    Parameters
    ----------
    transition_set
        Collection of all relevant transitions and related attributes.
    largest_number
        The largest number used in the simulation. If None, the smallest increment is
        calculated for a probability of 0.001.

    Returns
    -------
    None
    """
    max_rate_index = np.argmax(transition_set.row_sums)
    max_rate = transition_set.row_sums[max_rate_index]
    states = transition_set.combined_state_transitions_df["final_state"].loc[
        max_rate_index
    ]
    states = [int(state) for state in states]

    if largest_number is not None:
        smallest_increment = np.nextafter(largest_number, np.inf) - largest_number

        probability_smallest_increment = 1 - np.exp(-max_rate * smallest_increment)

        logger.warning(
            "Floating point precision error warning:\n "
            f"The smallest safe increment is {smallest_increment:.2e}."
            "\n Everything drawn below this number might be rounded to zero\n when "
            "approaching the time limit of this simulation."
            "\n Using the highest possible rate which occurs for example in "
            f"state combination {states}\n gives a probability of "
            f"{probability_smallest_increment:.2e} for a smaller increment"
            " to be drawn.",
            stacklevel=2,
        )

    else:
        probability_to_check = 0.001
        smallest_increment = -np.log(1 - probability_to_check) / max_rate
        log_space = np.logspace(-3, 4, 8)
        for i, large_number in enumerate(log_space):
            if np.nextafter(large_number, np.inf) - large_number < smallest_increment:
                continue
            else:
                logger.warning(
                    "Floating point precision error warning:\n "
                    f"The higher limit of smallest increment with a probability of "
                    f"{probability_to_check:.2e} is {smallest_increment:.2e}."
                    "\n This was estimated using the highest possible rate "
                    f"which occurs for example in state combination {states}."
                    "\n Everything drawn below this number will be rounded "
                    f"to zero starting somewhere between {log_space[i - 1]:.2e}"
                    f" - {large_number:.2e}.",
                    stacklevel=2,
                )
                break
