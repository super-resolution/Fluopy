import numpy as np
import src.fluorophore_systems as fs


def multiple_simulations(n_simulations, class_name, class_args, simulate_args, emitting_args, seed):
    """
    Instantiate multiple objects of subclasses of FluorophoreSystem, run their methods and collect the objects.

    Parameters
    ----------
    n_simulations : int
        Number of simulations.
    class_name : str
        One of "Jablonski", "OnOff".
    class_args : dict
        Arguments to pass to the subclass.
    simulate_args : dict
        Arguments to pass to the method simulate().
    emitting_args : dict
        Arguments to pass to the method emitters().
    seed : None, int, BitGenerator, Generator
        Seed to initialize a BitGenerator.

    Returns
    -------
    object_collector : list
        Contains all instantiated objects.
    """
    rng = np.random.default_rng(seed)
    object_collector = []

    class_lookup = {"Jablonski": fs.JablonskiModel, "OnOff": fs.OnOffModel}
    for i in range(n_simulations):
        system = class_lookup[class_name](**class_args)
        system.simulate(seed=rng, **simulate_args)
        system.emitters(**emitting_args)
        object_collector.append(system)

    return object_collector
