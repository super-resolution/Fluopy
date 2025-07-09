import matplotlib.pyplot as plt
import pytest

from fluopy.figure import universal_figure


def test_universal_figure():
    axes = universal_figure()
    assert len(axes) == 1


@pytest.mark.visual
def test_universal_figure_visual():
    axes = universal_figure()
    assert len(axes) == 1
    plt.show()
