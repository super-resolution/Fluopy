import matplotlib.pyplot as plt
import pytest

from fluopy.figure import universal_figure


def test_universal_figure():
    ax = universal_figure()
    assert ax.figure is not None


def test_universal_figure_modifies_supplied_axis():
    _, expected = plt.subplots()

    ax = universal_figure(ax=expected, data=([0, 1], [1, 2]))

    assert ax is expected
    assert len(ax.lines) == 1


def test_universal_figure_multiple_line_defaults():
    data = [([0, 1], [1, 2]), ([0, 1], [2, 3])]

    ax = universal_figure(type_="multiple_line", data=data)

    lines = ax.lines
    assert len(lines) == 2
    assert [line.get_color() for line in lines] == ["blue", "blue"]
    assert ax.get_legend_handles_labels() == ([], [])


def test_universal_figure_multiple_line_scalar_label_and_color():
    data = [([0, 1], [1, 2]), ([0, 1], [2, 3])]

    ax = universal_figure(type_="multiple_line", data=data, label="data", color="red")

    lines = ax.lines
    assert [line.get_color() for line in lines] == ["red", "red"]
    assert [line.get_label() for line in lines] == ["data", "data"]


@pytest.mark.visual
def test_universal_figure_visual():
    ax = universal_figure()
    assert ax.figure is not None
    plt.show()
