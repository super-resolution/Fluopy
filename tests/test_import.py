import fluopy


def test_import_fluopy():
    assert fluopy.__all__ == [
        "analysis",
        "blinking",
        "emissions",
        "fcs",
        "fluo_data",
        "fluorophores",
        "kappa_squared",
        "photophysics",
        "plotting",
        "prediction",
        "simulation",
        "tcspc",
        "transitions",
    ]


def test_root_api_exposes_modules_only():
    assert fluopy.fluorophores.Fluorophore is not None
    assert fluopy.fluorophores.FluorophoreSystem is not None
    assert not hasattr(fluopy, "Fluorophore")
    assert not hasattr(fluopy, "FluorophoreSystem")
