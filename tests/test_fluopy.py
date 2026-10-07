import importlib
import logging
import sys

import fluopy


def test_version():
    assert fluopy.__version__


def test_version_when_version_module_is_unavailable(monkeypatch):
    try:
        with monkeypatch.context() as context:
            context.setitem(sys.modules, "fluopy._version", None)
            reloaded_fluopy = importlib.reload(fluopy)

            assert reloaded_fluopy.__version__ == "not-installed"
    finally:
        importlib.reload(fluopy)


def test_logging():
    loggers = {"fluopy": fluopy.logger}
    for module_name in (
        "analysis",
        "emissions",
        "fcs",
        "fluorophores",
        "prediction",
        "simulation",
        "tcspc",
        "transitions",
    ):
        module = getattr(fluopy, module_name)
        loggers[f"fluopy.{module_name}"] = module.logger

    for name, logger in loggers.items():
        assert isinstance(logger, logging.Logger)
        assert logger.name == name
