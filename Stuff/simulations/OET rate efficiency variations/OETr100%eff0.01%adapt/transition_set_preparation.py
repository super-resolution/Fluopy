import numpy as np
import src.fluorophores as fl
import src.transitions as tr


def prepare_transition_set_ofret(number_fluorophores, distance):
    fluorophores = fl.construct_fluorophores(
    name="cy5_dna", distance=distance, count=number_fluorophores, shape="square"
    )
    fluorophore_system = fl.FluorophoreSystem(fluorophores=fluorophores)

    transitions = fluorophore_system.load_transitions(
    summarize=False,
    irradiance=5,
    wavelength=640,
    bleaching=True,
    energy_transfer=True,
    dstorm=True,
    dstorm_parameters={'reducing_agent':'mea',
    'concentration':100,
    'ph':7.5},
    energy_transfer_parameters={'overwrite': {'off': [1, 0.0001]}, 
                                'exclude': ['s0'],
                                'include': {'t1': [(tr.TransitionType.S_T_OFF, 0.1)],
                                            's1': [(tr.TransitionType.S_S_OFF, 0.1)]}}
    )
    rad_escape = tr.Transition(tr.TransitionType.RAD_ESCAPE, rate=1e2, fluorophore_ids=np.arange(0, number_fluorophores, dtype=int).tolist())
    rad_relax = tr.Transition(tr.TransitionType.RAD_RELAX, rate=1e4, fluorophore_ids=np.arange(0, number_fluorophores, dtype=int).tolist())
    transitions['cy5_dna'].extend([rad_escape, rad_relax])
    transition_set = tr.TransitionSet(transitions, fluorophore_system)
    transition_set.finalize()

    return transition_set
