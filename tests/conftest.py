from pathlib import Path

import pytest

from fluopy import emissions as em
from fluopy import fluorophores as fl
from fluopy import prediction as pr
from fluopy import simulation as si
from fluopy import transitions as tr


@pytest.fixture()
def flu_obj_cy5_1():
    return fl.Fluorophore(name="testfluo_1", position=[0, 0])


@pytest.fixture()
def flu_obj_cy5_2():
    return fl.Fluorophore(name="testfluo_1", position=[0, 1])


@pytest.fixture()
def flu_obj_atto643():
    return fl.Fluorophore(name="testfluo_2", position=[0, 2])


@pytest.fixture()
def flu_obj_unknown():
    return fl.Fluorophore(name="aa", position=[0, 3])


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
    return fl.FluorophoreSystem(
        fluorophores=[flu_obj_cy5_1, flu_obj_cy5_2, flu_obj_atto643]
    )


@pytest.fixture()
def flu_sys_2xcy5(flu_obj_cy5_1, flu_obj_cy5_2):
    return fl.FluorophoreSystem(fluorophores=[flu_obj_cy5_1, flu_obj_cy5_2])


@pytest.fixture()
def flu_sys_1xcy5_1xatto643(flu_obj_cy5_1, flu_obj_atto643):
    return fl.FluorophoreSystem(fluorophores=[flu_obj_cy5_1, flu_obj_atto643])


@pytest.fixture()
def tr_set_bl_et_3f(flu_sys_2xcy5_1xatto643):
    transitions = flu_sys_2xcy5_1xatto643.load_transitions(
        irradiance=2,
        wavelength=640,
        bleaching=True,
        energy_transfer=True,
        dstorm=False,
        energy_transfer_parameters={"refractive_index": 1},
    )
    tset = tr.TransitionSet(
        transitions=transitions, fluorophore_system=flu_sys_2xcy5_1xatto643
    )
    tset.finalize()
    return tset


@pytest.fixture()
def tr_set_bl_et_2f_diff(flu_sys_1xcy5_1xatto643):
    transitions = flu_sys_1xcy5_1xatto643.load_transitions(
        irradiance=2,
        wavelength=640,
        bleaching=True,
        energy_transfer=True,
        dstorm=False,
        energy_transfer_parameters={"refractive_index": 1},
    )
    tset = tr.TransitionSet(
        transitions=transitions, fluorophore_system=flu_sys_1xcy5_1xatto643
    )
    tset.finalize()
    return tset


@pytest.fixture()
def tr_set_bl_et_2f_same(flu_sys_2xcy5):
    transitions = flu_sys_2xcy5.load_transitions(
        irradiance=2,
        wavelength=640,
        bleaching=True,
        energy_transfer=True,
        dstorm=False,
        energy_transfer_parameters={"refractive_index": 1},
    )
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_2xcy5)
    tset.finalize()
    return tset


@pytest.fixture()
def tr_set_et_2f_diff(flu_sys_1xcy5_1xatto643):
    transitions = flu_sys_1xcy5_1xatto643.load_transitions(
        irradiance=2,
        wavelength=640,
        bleaching=False,
        energy_transfer=True,
        dstorm=False,
        energy_transfer_parameters={"refractive_index": 1},
    )
    tset = tr.TransitionSet(
        transitions=transitions, fluorophore_system=flu_sys_1xcy5_1xatto643
    )
    tset.finalize()
    return tset


@pytest.fixture()
def tr_set_2f_diff(flu_sys_1xcy5_1xatto643):
    transitions = flu_sys_1xcy5_1xatto643.load_transitions(
        irradiance=2,
        wavelength=640,
        bleaching=False,
        energy_transfer=False,
        dstorm=False,
    )
    tset = tr.TransitionSet(
        transitions=transitions, fluorophore_system=flu_sys_1xcy5_1xatto643
    )
    tset.finalize()
    return tset


@pytest.fixture()
def tr_set_1f_bl(flu_sys_cy5):
    transitions = flu_sys_cy5.load_transitions(
        irradiance=2,
        wavelength=640,
        bleaching=True,
        energy_transfer=False,
        dstorm=False,
    )
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_cy5)
    tset.finalize()
    return tset


@pytest.fixture()
def tr_set_1f_bl_2(flu_sys_cy5):
    transitions = flu_sys_cy5.load_transitions(
        irradiance=5,
        wavelength=640,
        bleaching=True,
        energy_transfer=False,
        dstorm=False,
    )
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_cy5)
    tset.finalize()
    return tset


@pytest.fixture()
def tr_set_1f(flu_sys_cy5):
    transitions = flu_sys_cy5.load_transitions(
        irradiance=2,
        wavelength=640,
        bleaching=False,
        energy_transfer=False,
        dstorm=False,
    )
    tset = tr.TransitionSet(transitions=transitions, fluorophore_system=flu_sys_cy5)
    tset.finalize()
    return tset


@pytest.fixture()
def tr_set_large(flu_sys_cy5):
    transitions = flu_sys_cy5.load_transitions(
        irradiance=2,
        wavelength=640,
        bleaching=True,
        energy_transfer=False,
        dstorm=True,
        dstorm_parameters={"reducing_agent": "test", "ph": 8},
    )
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
def pred_tr_set_1f_bl_2(tr_set_1f_bl_2):
    pred = pr.Prediction(transition_set=tr_set_1f_bl_2)
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


@pytest.fixture()
def sim_dstorm(tr_set_large):
    sim = si.Simulation(transition_set=tr_set_large)
    sim.run(size=1e6, seed=1)
    return sim


@pytest.fixture()
def em_tr_set_1f_bl(sim_tr_set_1f_bl):
    emis = em.Emissions(frame_time="5ms", bandpass=None, seed=1)
    emis.extract(simulation=sim_tr_set_1f_bl)
    return emis


@pytest.fixture()
def em_tr_set_et_2f_diff(sim_tr_set_et_2f_diff):
    emis = em.Emissions(frame_time="5ms", bandpass=None, seed=1)
    emis.extract(simulation=sim_tr_set_et_2f_diff)
    return emis


@pytest.fixture()
def em_large():
    emis = em.Emissions.load(
        path=Path(__file__).parent,
        name_extension="_em_large",
    )
    return emis


@pytest.fixture()
def em_very_large():
    emis = em.Emissions.load(
        path=Path(__file__).parent,
        name_extension="_em_very_large",
    )
    return emis
