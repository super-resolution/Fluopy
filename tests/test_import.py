import fluopy


def test_import_fluopy():
    # print(dir(fluopy))
    # print(fluopy.__all__)
    assert "fluorophores" not in fluopy.__all__
    assert "Fluorophore" in fluopy.__all__


def test_fluorophore_root_api():
    assert "Fluorophore" in fluopy.__all__
    assert "FluorophoreSystem" in fluopy.__all__
    assert "get_distances" not in fluopy.__all__
