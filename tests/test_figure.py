import matplotlib.pyplot as plt
import pytest

from fluopy.figure import universal_figure


def test_universal_figure():
    axes = universal_figure()
    assert len(axes) == 1


def test_universal_figure_multiple_line_defaults():
    data = [([0, 1], [1, 2]), ([0, 1], [2, 3])]

    axes = universal_figure(type_="multiple_line", data=data)

    lines = axes[0, 0].lines
    assert len(lines) == 2
    assert [line.get_color() for line in lines] == ["blue", "blue"]
    assert axes[0, 0].get_legend_handles_labels() == ([], [])


def test_universal_figure_multiple_line_scalar_label_and_color():
    data = [([0, 1], [1, 2]), ([0, 1], [2, 3])]

    axes = universal_figure(type_="multiple_line", data=data, label="data", color="red")

    lines = axes[0, 0].lines
    assert [line.get_color() for line in lines] == ["red", "red"]
    assert [line.get_label() for line in lines] == ["data", "data"]


@pytest.mark.visual
def test_universal_figure_visual():
    axes = universal_figure()
    assert len(axes) == 1
    plt.show()
