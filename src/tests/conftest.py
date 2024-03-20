import pytest
import src.fluorophores as fl
import src.transitions as tr
import src.prediction as pr
import src.simulation as si

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
def flu_sys_cy5(flu_obj_cy5_1):
    return fl.FluorophoreSystem(fluorophores=[flu_obj_cy5_1])

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
def flu_sys_2xcy5(flu_obj_cy5_1, flu_obj_cy5_2):
    return fl.FluorophoreSystem(fluorophores=[flu_obj_cy5_1, flu_obj_cy5_2])

@pytest.fixture()
def flu_sys_1xcy5_1xatto643(flu_obj_cy5_1, flu_obj_atto643):
    return fl.FluorophoreSystem(fluorophores=[flu_obj_cy5_1, flu_obj_atto643])

@pytest.fixture()
def tr_set_bl_et_3f(flu_sys_2xcy5_1xatto643):
    transitions = flu_sys_2xcy5_1xatto643.load_transitions(irradiance=2, wavelength=640, 
                                                           bleaching=True, energy_transfer=True,
                                                           dstorm=False)
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_2xcy5_1xatto643)
    tset.finalize()
    return tset

@pytest.fixture()
def tr_set_bl_et_2f_diff(flu_sys_1xcy5_1xatto643):
    transitions = flu_sys_1xcy5_1xatto643.load_transitions(irradiance=2, wavelength=640, 
                                                           bleaching=True, energy_transfer=True,
                                                           dstorm=False)
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_1xcy5_1xatto643)
    tset.finalize()
    return tset

@pytest.fixture()
def tr_set_bl_et_2f_same(flu_sys_2xcy5):
    transitions = flu_sys_2xcy5.load_transitions(irradiance=2, wavelength=640,
                                                bleaching=True, energy_transfer=True,
                                                dstorm=False)
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_2xcy5)
    tset.finalize()
    return tset

@pytest.fixture()
def tr_set_et_2f_diff(flu_sys_1xcy5_1xatto643):
    transitions = flu_sys_1xcy5_1xatto643.load_transitions(irradiance=2, wavelength=640,
                                                bleaching=False, energy_transfer=True,
                                                dstorm=False)
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_1xcy5_1xatto643)
    tset.finalize()
    return tset

@pytest.fixture()
def tr_set_2f_diff(flu_sys_1xcy5_1xatto643):
    transitions = flu_sys_1xcy5_1xatto643.load_transitions(irradiance=2, wavelength=640,
                                                bleaching=False, energy_transfer=False,
                                                dstorm=False)
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_1xcy5_1xatto643)
    tset.finalize()
    return tset

@pytest.fixture()
def tr_set_1f_bl(flu_sys_cy5):
    transitions = flu_sys_cy5.load_transitions(irradiance=2, wavelength=640,
                                                bleaching=True, energy_transfer=False,
                                                dstorm=False)
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_cy5)
    tset.finalize()
    return tset

@pytest.fixture()
def tr_set_1f(flu_sys_cy5):
    transitions = flu_sys_cy5.load_transitions(irradiance=2, wavelength=640,
                                                bleaching=False, energy_transfer=False,
                                                dstorm=False)
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_cy5)
    tset.finalize()
    return tset

@pytest.fixture()
def pred_tr_set_1f(tr_set_1f):
    pred = pr.Prediction(transition_set=tr_set_1f)
    return pred

@pytest.fixture()
def pred_tr_set_1f_bl(tr_set_1f_bl):
    pred = pr.Prediction(transition_set=tr_set_1f_bl)
    return pred

@pytest.fixture()
def sim_tr_set_1f_bl(tr_set_1f_bl):
    sim = si.Simulation(transition_set=tr_set_1f_bl)
    sim.run(size=1000, seed=1)
    return sim

@pytest.fixture()
def sim_tr_set_et_2f_diff(tr_set_et_2f_diff):
    sim = si.Simulation(transition_set=tr_set_et_2f_diff)
    sim.run(size=1000, end_time=5e-7, seed=1)
    return sim

@pytest.fixture()
def sim_tr_set_2f_diff(tr_set_2f_diff):
    sim = si.Simulation(transition_set=tr_set_2f_diff)
    sim.run(size=1000, seed=1)
    return sim

########################################################################################

# import copy
# import warnings


# with warnings.catch_warnings():
#     warnings.simplefilter("error")
#     tr.TransitionSet(transitions=old_transitions, fluorophore_system=transition_set_object.fluorophore_system)
# with pytest.warns(UserWarning):
#     tr.TransitionSet(transitions=new_transitions, fluorophore_system=transition_set_object.fluorophore_system)


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
