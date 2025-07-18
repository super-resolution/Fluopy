import matplotlib.pyplot as plt
import pandas as pd

from fluopy.miscellaneous import (
    add_table,
    create_row_subtitles,
    delete_subplots,
    format_axis_labels,
    format_electronic_state,
    format_transition,
    get_figure,
    print_class,
)


def test_delete_subplots():
    fig, axes = plt.subplots(2, 3)
    assert (
        delete_subplots(axes=axes.ravel(), keep_number=1, del_positions=[1, 1]) is None
    )
    assert axes.shape == (2, 3)
    plt.close()


def test_create_row_subtitles():
    fig, axes = plt.subplots(2, 3)
    create_row_subtitles(axes=axes.ravel(), nrows=2, ncols=3, titles=["one", "two"])
    # plt.show()
    plt.close()


def test_add_table():
    fig, ax = plt.subplots(2, 1)
    return_value = add_table(
        axes=ax,
        data=pd.Series([1, 2, 3], index=["one", "two", "three"]),
        grid=212,
    )
    assert return_value is ax
    # plt.show()
    plt.close()


def test_get_figures():
    fig, axes = plt.subplots(2, 3)
    return_value = get_figure()
    assert return_value is fig
    return_value = get_figure(axes=axes.ravel())
    assert return_value is fig
    return_value = get_figure(axes=axes.ravel()[4])
    assert return_value is fig
    plt.close()


def test_print_class(capsys):
    instance = plt.Figure()
    assert print_class(class_instance=instance) is None
    captured = capsys.readouterr()
    assert "Figure" in captured.out
    plt.close()


def test_format_transition():
    return_value = format_transition(label="123_456")
    # print(return_value)
    assert return_value == "123$_{456}$"


def test_format_electronic_state():
    return_value = format_electronic_state(label="___S1_T1__")
    # print(return_value)
    assert return_value == "___S1_T1__"


def test_format_axis_labels():
    return_value = format_axis_labels(label="___(1)___", offset="e12")
    # print(return_value)
    assert return_value == "___($10^{12}$ x 1)___"
