import fluopy


def test_import_fluopy():
    # print(dir(fluopy))
    # print(fluopy.__all__)
    assert "fluorophores" not in fluopy.__all__
    assert "Fluorophore" in fluopy.__all__
