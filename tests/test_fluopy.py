import fluopy


def test_version():
    assert fluopy.__version__


def test_logging():
    assert fluopy.logger

    for module_name in [
        "analysis",
        # 'blinking',
        # 'distributions',
        "emissions",
        # 'fcs',
        # 'figure',
        # 'fitting',
        # 'fluo_data',
        "fluorophores",
        # 'formulas',
        # 'kappa_squared',
        # 'miscellaneous',
        # 'network',
        "prediction",
        # 'routines',
        "simulation",
        "simulation_tcspc",
        # 'transitions'
    ]:
        module = getattr(fluopy, module_name)
        assert module.logger
