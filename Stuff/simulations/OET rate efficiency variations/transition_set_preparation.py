import numpy as np
import src.fluorophores as fl
import src.transitions as tr


def prepare_transition_set_ofret(number_fluorophores, distance, rate, efficiency):
    """
    Prepare a transition set for a given number of fluorophores and distance between them.

    Parameters
    ----------
    number_fluorophores : int
        Number of fluorophores in the system.
    distance : float
        Distance between the fluorophores.
    rate : float
        Factor to apply to the energy transfer rate.
    efficiency : float
        Factor to apply to the energy transfer efficiency.
    
    Returns
    -------
    transition_set : TransitionSet
        Transition set prepared for the given parameters
    """
    fluorophores = fl.construct_fluorophores(
    name="cy5_dna", distance=distance, count=number_fluorophores, shape="square"
    )
    fluorophore_system = fl.FluorophoreSystem(fluorophores=fluorophores)

    transitions = fluorophore_system.load_transitions(
    summarize=True,
    irradiance=5,
    wavelength=640,
    bleaching=True,
    energy_transfer=True,
    dstorm=True,
    dstorm_parameters={'reducing_agent':'mea',
    'concentration':100,
    'ph':7.5},
    energy_transfer_parameters={'overwrite': {'off': [rate, efficiency]}, 
                                'exclude': ['s0']}
    )
    transition_set = tr.TransitionSet(transitions, fluorophore_system)
    transition_set.finalize()

    return transition_set
