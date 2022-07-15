import numpy as np
import src.fluorophore_systems as fs


def multiple_simulations(n_simulations, class_name, class_args, simulate_args, emitting_args, seed):
    rng = np.random.default_rng(seed)
    object_collector = []

    class_lookup = {"Jablonski": fs.JablonskiModel, "OnOff": fs.OnOffModel}
    for i in range(n_simulations):
        system = class_lookup[class_name](**class_args)
        system.simulate(seed=rng, **simulate_args)
        system.emitters(**emitting_args)
        object_collector.append(system)

    return object_collector
