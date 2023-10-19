import pytest
import src.fluorophores as fl
import src.transitions as tr
import src.simulation as si
import src.emissions as em
import src.fcs as fcs
import src.blinking as bl
import src.cy5_properties as cy5


@pytest.fixture()
def cy5_object():
    return cy5.Cy5(parameter_set='test')


@pytest.fixture()
def fluorophore_object(request):
    return fl.Fluorophore(*request.param)


@pytest.fixture()
def fluorophore_system_object(request):
    fluorophores = [fl.Fluorophore(*params) for params in request.param]
    return fl.FluorophoreSystem(fluorophores)


@pytest.fixture()
def transition_object(request):
    return tr.Transition(*request.param)


@pytest.fixture()
def transitionlist(request):
    return [tr.Transition(*param) for param in request.param]


@pytest.fixture()
def transition_set_object():
    fluorophore_1 = fl.Fluorophore(name='Cy5', position=[1, 1])
    fluorophore_2 = fl.Fluorophore(name='Cy5', position=[1, 8])
    fluorophore_3 = fl.Fluorophore(name='Cy5', position=[8, 1])
    fluorophore_4 = fl.Fluorophore(name='Cy5', position=[8, 8])
    fluorophore_system = fl.FluorophoreSystem(fluorophores=[fluorophore_1, fluorophore_2, fluorophore_3, fluorophore_4])

    excitation = tr.Transition(transition_type=tr.TransitionType.EXCITATION, rate=5)
    emission = tr.Transition(transition_type=tr.TransitionType.FLUORESCENT_EMISSION, rate=4)
    ics = tr.Transition(transition_type=tr.TransitionType.INTERNAL_CONVERSION_S, rate=1)
    isc = tr.Transition(transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_ST, rate=0)
    homo_fret_1 = tr.Transition(transition_type=tr.TransitionType.HOMO_FRET, rate=9, distance=7)
    homo_fret_2 = tr.Transition(transition_type=tr.TransitionType.HOMO_FRET, rate=9, distance=9.899)

    transitions = [excitation, emission, ics, isc, homo_fret_1, homo_fret_2]

    return tr.TransitionSet(transitions=transitions, fluorophore_system=fluorophore_system)


@pytest.fixture()
def transition_set_object_bleach():
    fluorophore_1 = fl.Fluorophore(name='Cy5', position=[1, 1])
    fluorophore_2 = fl.Fluorophore(name='Cy5', position=[1, 8])
    fluorophore_system = fl.FluorophoreSystem(fluorophores=[fluorophore_1, fluorophore_2])

    excitation = tr.Transition(transition_type=tr.TransitionType.EXCITATION, rate=5)
    emission = tr.Transition(transition_type=tr.TransitionType.FLUORESCENT_EMISSION, rate=4)
    ics = tr.Transition(transition_type=tr.TransitionType.INTERNAL_CONVERSION_S, rate=1)
    iscst = tr.Transition(transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_ST, rate=2)
    iscts = tr.Transition(transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_TS, rate=5)
    bleach = tr.Transition(transition_type=tr.TransitionType.PHOTOBLEACHING_1, rate=4)

    transitions = [excitation, emission, ics, iscst, iscts, bleach]

    return tr.TransitionSet(transitions=transitions, fluorophore_system=fluorophore_system)


@pytest.fixture()
def simulation_object_1(transition_set_object):
    transition_set_object.finalize()
    object = si.Simulation(transition_set_object)
    object.run(start_at=None, size=1000, end_time=None, seed=3, use_memmap=None)
    return object


@pytest.fixture()
def simulation_object_2(transition_set_object):
    transition_set_object.finalize()
    object = si.Simulation(transition_set_object)
    object.run(start_at=None, size=1000, end_time=25, seed=3, use_memmap=None)
    return object


@pytest.fixture()
def emissions_object_1(simulation_object_1):
    object = em.Emissions(simulation_object_1)
    return object


@pytest.fixture()
def emissions_object_2(simulation_object_2):
    object = em.Emissions(simulation_object_2)
    return object


@pytest.fixture()
def fcs_object(emissions_object_1):
    object = fcs.FCS(emissions_object_1)
    return object


@pytest.fixture()
def blinking_object(emissions_object_1):
    object = bl.Blinking(emissions_object_1)
    return object
