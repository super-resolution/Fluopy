import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pytest
from matplotlib.patches import Patch
from scipy.stats import norm, poisson

from fluopy.plotting import plot_data


def test_plot_data():
    ax = plot_data()
    assert ax.figure is not None
    assert ax.figure.dpi == plt.rcParamsDefault["figure.dpi"]
    assert ax.figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert all(spine.get_linewidth() == 2 for spine in ax.spines.values())


def test_plot_data_modifies_supplied_axis():
    _, expected = plt.subplots()

    ax = plot_data(ax=expected, data=([0, 1], [1, 2]))

    assert ax is expected
    assert len(ax.lines) == 1


def test_plot_data_multiple_line_defaults():
    data = [([0, 1], [1, 2]), ([0, 1], [2, 3])]

    ax = plot_data(type_="multiple_line", data=data)

    lines = ax.lines
    assert len(lines) == 2
    assert [line.get_color() for line in lines] == ["blue", "blue"]
    assert ax.get_legend_handles_labels() == ([], [])


def test_plot_data_multiple_line_scalar_label_and_color():
    data = [([0, 1], [1, 2]), ([0, 1], [2, 3])]

    ax = plot_data(type_="multiple_line", data=data, label="data", color="red")

    lines = ax.lines
    assert [line.get_color() for line in lines] == ["red", "red"]
    assert [line.get_label() for line in lines] == ["data", "data"]


@pytest.mark.parametrize(
    "type_, data, kwargs, artist_attribute, expected_count",
    [
        ("hist", [0, 1, 1, 2], {"bins": 2}, "patches", 2),
        ("multiple_hist", [[0, 1], [1, 2]], {"bins": 2}, "patches", 4),
        ("2d_hist", ([0, 1], [1, 2]), {"bins": 2}, "collections", 1),
        ("bar", ([0, 1], [1, 2]), {}, "patches", 2),
        ("line", ([0, 1], [1, 2]), {}, "lines", 1),
        ("step", ([0, 1], [1, 2]), {}, "lines", 1),
        ("stair", ([0, 1, 2], [1, 2]), {}, "patches", 1),
        ("errorbar", ([0, 1], [1, 2], [0.1, 0.2]), {}, "lines", 1),
        ("scatter", ([0, 1], [1, 2]), {}, "collections", 1),
        ("boxplot", [1, 2, 3], {"label": "data"}, "lines", 7),
    ],
)
def test_plot_data_plot_types(type_, data, kwargs, artist_attribute, expected_count):
    ax = plot_data(type_=type_, data=data, **kwargs)

    assert len(getattr(ax, artist_attribute)) == expected_count


def test_plot_data_grouped_bars():
    ax = plot_data(
        type_="bar",
        data=([0, 1], [[1, 2], [3, 4]]),
        color=["red", "blue"],
        label=["first", "second"],
        width=0.4,
    )

    assert len(ax.patches) == 4
    assert ax.get_legend_handles_labels()[1] == ["first", "second"]
    assert [patch.get_x() for patch in ax.patches[2:]] == pytest.approx([0.2, 1.2])


def test_plot_data_grouped_bars_without_width():
    ax = plot_data(
        type_="bar",
        data=([0, 1], [[1, 2], [3, 4]]),
        color=["red", "blue"],
        label=["first", "second"],
    )

    assert [patch.get_x() for patch in ax.patches[2:]] == pytest.approx([-0.4, 0.6])


def test_plot_data_histogram_dot_and_distribution_overlays():
    dot_ax = plot_data(
        type_="hist",
        data=[-1, 0, 1, 2],
        histtype="dot",
        bins=4,
        plot_distribution=poisson(1),
        plot_distribution_label="Poisson",
    )
    discrete_ax = plot_data(
        type_="hist",
        data=[0, 1, 2],
        bins=3,
        plot_distribution=poisson(1),
    )
    continuous_ax = plot_data(
        type_="hist",
        data=[-1, 0, 1],
        bins=3,
        plot_distribution=norm(),
    )

    assert len(dot_ax.patches) == 0
    assert len(dot_ax.collections) == 1
    assert dot_ax.lines[0].get_label() == "Poisson"
    assert discrete_ax.lines[0].get_xdata()[0] == 0
    assert continuous_ax.lines[0].get_xdata().size == 100


def test_plot_data_multiple_histogram_options():
    weighted_ax = plot_data(
        type_="multiple_hist",
        data=[[-1, 0, 1], [0, 1, 2], [-1, 0, 1]],
        color=lambda index: ["red", "blue", "green"][index],
        label="data",
        bins=3,
        weights=True,
        plot_distribution=[poisson(1), poisson(1), norm()],
    )
    listed_ax = plot_data(
        type_="multiple_hist",
        data=[[0, 1], [1, 2]],
        color=["red", "blue"],
        label=["first", "second"],
        bins=2,
    )

    assert len(weighted_ax.lines) == 3
    assert weighted_ax.get_legend_handles_labels()[1][:2] == ["data", "pred"]
    assert listed_ax.get_legend_handles_labels()[1] == ["first", "second"]


def test_plot_data_multiple_line_callable_and_list_options():
    data = [([0, 1], [1, 2]), ([0, 1], [2, 3])]

    callable_ax = plot_data(
        type_="multiple_line",
        data=data,
        color=lambda index: ["red", "blue"][index],
        label=["first", "second"],
    )
    listed_ax = plot_data(
        type_="multiple_line",
        data=data,
        color=["green", "orange"],
    )

    assert [line.get_color() for line in callable_ax.lines] == ["red", "blue"]
    assert [line.get_label() for line in callable_ax.lines] == ["first", "second"]
    assert [line.get_color() for line in listed_ax.lines] == ["green", "orange"]


def test_plot_data_applies_axis_and_legend_options():
    ax = plot_data(
        data=([1, 2], [3, 4]),
        title="title",
        xlabel="time",
        ylabel="signal",
        xlim=(0, 3),
        ylim=(2, 5),
        xticks=[1, 2],
        yticks=[3, 4],
        label="data",
        legend=True,
        draw_marker=([1.5], [3.5]),
        second_axis_x=True,
        second_axis_y=True,
    )

    assert ax.get_title() == "title"
    assert ax.get_xlabel() == "time"
    assert ax.get_ylabel() == "signal"
    assert ax.get_xlim() == pytest.approx((0, 3))
    assert ax.get_ylim() == pytest.approx((2, 5))
    assert len(ax.collections) == 1
    assert ax.get_legend().get_texts()[0].get_text() == "data"
    assert len(ax.child_axes) == 2


def test_plot_data_applies_tick_options():
    ax = plot_data(
        data=([1, 2], [10, 100]),
        adjust_x=2,
        adjust_y=0.5,
        xticks=[1, 2],
        yticks=[10, 100],
        xticklabels={"labels": ["one", "two"], "rotation": 30},
        yticklabels={"labels": ["ten", "hundred"]},
        tick_spacing_x=0.5,
        tick_spacing_y=10,
        tick_params={"direction": "in"},
    )

    assert isinstance(ax.xaxis.get_major_locator(), ticker.MultipleLocator)
    assert isinstance(ax.yaxis.get_major_locator(), ticker.MultipleLocator)
    np.testing.assert_allclose(
        np.diff(ax.xaxis.get_major_locator().tick_values(0, 1)),
        0.5,
    )
    np.testing.assert_allclose(
        np.diff(ax.yaxis.get_major_locator().tick_values(0, 20)),
        10,
    )


def test_plot_data_log_minor_ticks():
    both_minor = plot_data(
        data=([1, 10], [1, 10]),
        xscale="log",
        yscale="log",
        xminor=True,
        yminor=True,
    )
    minor_disabled = plot_data(
        data=([1, 10], [1, 10]),
        xscale="log",
        yscale="log",
        xminor=False,
        yminor=False,
    )

    assert both_minor.get_xscale() == "log"
    assert both_minor.get_yscale() == "log"
    assert isinstance(both_minor.xaxis.get_minor_locator(), ticker.LogLocator)
    assert isinstance(both_minor.yaxis.get_minor_locator(), ticker.LogLocator)
    assert len(minor_disabled.xaxis.get_minorticklocs()) == 0
    assert len(minor_disabled.yaxis.get_minorticklocs()) == 0


def test_plot_data_scientific_tick_labels():
    ax = plot_data(
        data=([1e6, 2e6], [1e-6, 2e-6]),
        xlabel="time",
        ylabel="signal",
        tick_style_x="sci",
        tick_style_y="sci",
    )

    assert "10" in ax.get_xlabel()
    assert "10" in ax.get_ylabel()


def test_plot_data_uses_custom_legend_handles():
    handle = Patch(color="red", label="custom")

    ax = plot_data(
        data=([0, 1], [1, 2]),
        legend=True,
        legendhandles=[handle],
        legendargs={"loc": "upper left"},
    )

    assert ax.get_legend().get_texts()[0].get_text() == "custom"


def test_plot_data_rejects_unknown_plot_type():
    with pytest.raises(ValueError, match="Invalid type_ argument"):
        plot_data(type_="unknown")


@pytest.mark.visual
def test_plot_data_visual():
    ax = plot_data()
    assert ax.figure is not None
    plt.show()
