import matplotlib.pyplot as plt
import pytest

from fluopy.figure import universal_figure


def test_universal_figure():
    ax = universal_figure()
    assert ax.figure is not None
    assert ax.figure.dpi == plt.rcParamsDefault["figure.dpi"]
    assert ax.figure.get_facecolor() == (1.0, 1.0, 1.0, 1.0)
    assert all(spine.get_linewidth() == 2 for spine in ax.spines.values())


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
def test_universal_figure_plot_types(
    type_, data, kwargs, artist_attribute, expected_count
):
    ax = universal_figure(type_=type_, data=data, **kwargs)

    assert len(getattr(ax, artist_attribute)) == expected_count


def test_universal_figure_grouped_bars():
    ax = universal_figure(
        type_="bar",
        data=([0, 1], [[1, 2], [3, 4]]),
        color=["red", "blue"],
        label=["first", "second"],
        width=0.4,
    )

    assert len(ax.patches) == 4
    assert ax.get_legend_handles_labels()[1] == ["first", "second"]
    assert [patch.get_x() for patch in ax.patches[2:]] == pytest.approx([0.2, 1.2])


def test_universal_figure_applies_axis_and_legend_options():
    ax = universal_figure(
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


def test_universal_figure_rejects_unknown_plot_type():
    with pytest.raises(ValueError, match="Invalid type_ argument"):
        universal_figure(type_="unknown")


@pytest.mark.visual
def test_universal_figure_visual():
    ax = universal_figure()
    assert ax.figure is not None
    plt.show()
