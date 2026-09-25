import importlib
import sys

import fluopy


def test_version():
    assert fluopy.__version__


def test_version_when_version_module_is_unavailable(monkeypatch):
    with monkeypatch.context() as context:
        context.setitem(sys.modules, "fluopy._version", None)
        reloaded_fluopy = importlib.reload(fluopy)

        assert reloaded_fluopy.__version__ == "not-installed"

    importlib.reload(fluopy)


def test_logging():
    assert fluopy.logger

    for module_name in [
        "analysis",
        # 'blinking',
        # 'distributions',
        "emissions",
        # 'fcs',
        # 'plotting',
        # 'fitting',
        # 'fluo_data',
        "fluorophores",
        # 'photophysics',
        # 'kappa_squared',
        # 'miscellaneous',
        "prediction",
        # 'routines',
        "simulation",
        "tcspc",
        # 'transitions'
    ]:
        module = getattr(fluopy, module_name)
        assert module.logger
