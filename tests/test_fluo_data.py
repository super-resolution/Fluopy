from fluopy import fluo_data as fd


def test_init_FluorophoreData():
    fluophore_data = fd.FluorophoreData()
    assert fluophore_data.QUANTUM_YIELD == 0


def test_init_cy5_dna():
    fluophore_data = fd.cy5_dna
    # print(fluophore_data.__doc__)
    assert fluophore_data.QUANTUM_YIELD == 0.27


def test_init_atto643():
    fluophore_data = fd.atto643
    assert fluophore_data.QUANTUM_YIELD == 0.6
