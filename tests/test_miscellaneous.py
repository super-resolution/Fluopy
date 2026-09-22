import matplotlib.pyplot as plt
import pandas as pd
import pytest
from PIL import Image

from fluopy.miscellaneous import (
    add_table,
    compute_tight_bbox,
    create_row_subtitles,
    crop_to_content_with_padding,
    delete_subplots,
    format_axis_labels,
    format_electronic_state,
    format_transition,
    get_figure,
    print_class,
)


def test_delete_subplots():
    fig, axes = plt.subplots(2, 3)
    assert delete_subplots(axes=axes.ravel(), keep_number=1, del_positions=None) is None
    assert fig.axes == [axes[0, 0]]
    plt.close(fig)


def test_delete_subplots_by_position():
    fig, axes = plt.subplots(2, 2)

    delete_subplots(axes=axes, del_positions=[[0, 1], [1, 0]])

    assert fig.axes == [axes[0, 0], axes[1, 1]]


def test_delete_subplots_requires_one_selection_method():
    _, axes = plt.subplots(1, 2)

    with pytest.raises(ValueError, match="Only one"):
        delete_subplots(axes=axes, keep_number=1, del_positions=[[0, 1]])
    with pytest.raises(ValueError, match="Either keep_number"):
        delete_subplots(axes=axes)


def test_create_row_subtitles():
    fig, axes = plt.subplots(2, 3)
    create_row_subtitles(axes=axes.ravel(), nrows=2, ncols=3, titles=["one", "two"])

    assert [ax.get_title() for ax in fig.axes[-2:]] == ["one", "two"]
    assert all(not ax.axison for ax in fig.axes[-2:])


def test_add_table():
    fig, ax = plt.subplots(2, 1)
    return_value = add_table(
        axes=ax,
        data=pd.Series([1, 2, 3], index=["one", "two", "three"]),
        grid=212,
    )
    assert return_value is ax
    assert len(fig.axes[-1].tables) == 1


def test_get_figures():
    fig, axes = plt.subplots(2, 3)
    return_value = get_figure()
    assert return_value is fig
    return_value = get_figure(axes=axes.ravel())
    assert return_value is fig
    return_value = get_figure(axes=axes.ravel()[4])
    assert return_value is fig
    plt.close(fig)


def test_print_class(capsys):
    instance = plt.Figure()
    assert print_class(class_instance=instance) is None
    captured = capsys.readouterr()
    assert "Figure" in captured.out
    plt.close()


def test_format_electronic_state():
    assert format_electronic_state(label="S1") == r"S$_{1}$"
    assert format_electronic_state(label="___S1_T1__") == "___S1_T1__"


def test_format_transition():
    assert format_transition(label="123_456") == "123$_{456}$"
    assert format_transition(label="S1") == "S1"


def test_format_axis_labels():
    assert format_axis_labels(label="___(1)___", offset="e12") == (
        "___($10^{12} \\times$ 1)___"
    )
    assert format_axis_labels(label="value [s]", offset="e-3") == (
        "value [$10^{-3} \\times$ s]"
    )
    assert format_axis_labels(label="value", offset="e3") == (
        "value ($ \\times 10^{3}$)"
    )


def test_compute_tight_bbox_preserves_figure_width():
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_title("title")

    bbox = compute_tight_bbox(fig, pad_inches=0.1)

    assert bbox.width == pytest.approx(4)
    assert bbox.height > 0


def test_crop_to_content_with_padding(tmp_path):
    source = tmp_path / "source.tiff"
    destination = tmp_path / "cropped.tiff"
    image = Image.new("RGB", (10, 10), "white")
    image.paste("black", (2, 3, 6, 7))
    image.save(source)

    crop_to_content_with_padding(
        source, destination, dpi=10, pad_inches=0.1, threshold=128
    )

    with Image.open(destination) as cropped:
        assert cropped.size == (6, 6)


def test_crop_to_content_rejects_blank_image(tmp_path):
    source = tmp_path / "blank.tiff"
    Image.new("RGB", (5, 5), "white").save(source)

    with pytest.raises(ValueError, match="no pixels below the threshold"):
        crop_to_content_with_padding(source, tmp_path / "cropped.tiff")
