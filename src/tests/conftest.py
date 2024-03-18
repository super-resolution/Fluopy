import pytest
import src.fluorophores as fl
import src.transitions as tr

@pytest.fixture()
def flu_obj_cy5_1():
    return fl.Fluorophore(name='cy5', position=[0, 0])

@pytest.fixture()
def flu_obj_cy5_2():
    return fl.Fluorophore(name='cy5', position=[0, 1])

@pytest.fixture()
def flu_obj_atto643():
    return fl.Fluorophore(name='atto643', position=[0, 2])

@pytest.fixture()
def flu_obj_unknown():
    return fl.Fluorophore(name='aa', position=[0, 3])

@pytest.fixture()
def flu_sys_unk(flu_obj_unknown):
    return fl.FluorophoreSystem(fluorophores=[flu_obj_unknown])

@pytest.fixture()
def flu_sys_unk_cy5(flu_obj_unknown, flu_obj_cy5_1):
    return fl.FluorophoreSystem(fluorophores=[flu_obj_unknown, flu_obj_cy5_1])

@pytest.fixture()
def flu_sys_2xcy5_1xatto643(flu_obj_cy5_1, flu_obj_cy5_2, flu_obj_atto643):
    return fl.FluorophoreSystem(fluorophores=[flu_obj_cy5_1, flu_obj_cy5_2, flu_obj_atto643])

@pytest.fixture()
def tr_set_bl_et(flu_sys_2xcy5_1xatto643):
    transitions = flu_sys_2xcy5_1xatto643.load_transitions(irradiance=2, wavelength=640, 
                                                           bleaching=True, energy_transfer=True,
                                                           dstorm=False)
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_2xcy5_1xatto643)
    tset.finalize()
    return tset

# import pytest
# import pandas as pd
# from dataclasses import asdict
# import src.fluorophores as fl
# import src.transitions as tr
# import src.simulation as si
# import src.emissions as em
# import src.fcs as fcs
# import src.blinking as bl
# from src.fluo_data import Cy5 as cy5


# @pytest.fixture()
# def transition_pd_series(request):
#     series = pd.Series(asdict(*request.param))
#     series = series.drop('identity')
#     return series


# @pytest.fixture()
# def transition_set_object():
#     fluorophore_1 = fl.Fluorophore(name='Cy5', position=[1, 1])
#     fluorophore_2 = fl.Fluorophore(name='Cy5', position=[1, 8])
#     fluorophore_3 = fl.Fluorophore(name='Cy5', position=[8, 1])
#     fluorophore_4 = fl.Fluorophore(name='Cy5', position=[8, 8])
#     fluorophore_system = fl.FluorophoreSystem(fluorophores=[fluorophore_1, fluorophore_2, fluorophore_3, fluorophore_4])

#     excitation = tr.Transition(transition_type=tr.TransitionType.EXCITATION, rate=5)
#     emission = tr.Transition(transition_type=tr.TransitionType.FLUORESCENT_EMISSION, rate=4)
#     ics = tr.Transition(transition_type=tr.TransitionType.INTERNAL_CONVERSION_S, rate=1)
#     isc = tr.Transition(transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_ST, rate=0)
#     homo_fret_1 = tr.Transition(transition_type=tr.TransitionType.HOMO_FRET, rate=9, distance=7)
#     homo_fret_2 = tr.Transition(transition_type=tr.TransitionType.HOMO_FRET, rate=9, distance=9.899)

#     transitions = [excitation, emission, ics, isc, homo_fret_1, homo_fret_2]

#     return tr.TransitionSet(transitions=transitions, fluorophore_system=fluorophore_system)


# @pytest.fixture()
# def transition_set_object_bleach():
#     fluorophore_1 = fl.Fluorophore(name='Cy5', position=[1, 1])
#     fluorophore_2 = fl.Fluorophore(name='Cy5', position=[1, 8])
#     fluorophore_system = fl.FluorophoreSystem(fluorophores=[fluorophore_1, fluorophore_2])

#     excitation = tr.Transition(transition_type=tr.TransitionType.EXCITATION, rate=5)
#     emission = tr.Transition(transition_type=tr.TransitionType.FLUORESCENT_EMISSION, rate=4)
#     ics = tr.Transition(transition_type=tr.TransitionType.INTERNAL_CONVERSION_S, rate=1)
#     iscst = tr.Transition(transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_ST, rate=2)
#     iscts = tr.Transition(transition_type=tr.TransitionType.INTERSYSTEM_CROSSING_TS, rate=5)
#     bleach = tr.Transition(transition_type=tr.TransitionType.PHOTOBLEACHING_1, rate=4)

#     transitions = [excitation, emission, ics, iscst, iscts, bleach]

#     return tr.TransitionSet(transitions=transitions, fluorophore_system=fluorophore_system)


# @pytest.fixture()
# def simulation_object_1(transition_set_object):
#     transition_set_object.finalize()
#     obj = si.Simulation(transition_set_object)
#     obj.run(start_at=None, size=1000, end_time=None, seed=3, use_memmap=None)
#     return obj


# @pytest.fixture()
# def simulation_object_2(transition_set_object):
#     transition_set_object.finalize()
#     obj = si.Simulation(transition_set_object)
#     obj.run(start_at=None, size=1000, end_time=25, seed=3, use_memmap=None)
#     return obj


# @pytest.fixture()
# def emissions_object(request):
#     obj = em.Emissions(*request.param)
#     return obj


# @pytest.fixture()
# def emissions_object_1(simulation_object_1):
#     obj = em.Emissions()
#     obj.extract(simulation_object_1)
#     return obj


# @pytest.fixture()
# def fcs_object(emissions_object_1):
#     obj = fcs.FCS(emissions_object_1)
#     return obj


# @pytest.fixture()
# def blinking_object(emissions_object_1):
#     obj = bl.Blinking(emissions_object_1)
#     return obj
