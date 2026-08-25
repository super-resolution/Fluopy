from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import numpy as np
import pytest

from fluopy import fluo_data as fd


def test_init_spectrum():
    spectrum = fd.Spectrum(
        wavelengths=[600, 610, 620],
        values=[0.1, 1.0, 0.4],
    )

    np.testing.assert_array_equal(
        spectrum.wavelengths,
        np.array([600.0, 610.0, 620.0]),
    )
    np.testing.assert_array_equal(
        spectrum.values,
        np.array([0.1, 1.0, 0.4]),
    )
    assert spectrum.wavelengths.dtype == np.float64
    assert spectrum.values.dtype == np.float64


def test_spectrum_is_frozen():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510],
        values=[0.2, 0.8],
    )

    with pytest.raises(FrozenInstanceError):
        spectrum.values = np.array([0.3, 0.7])


def test_spectrum_copies_input_arrays():
    wavelengths = np.array([600.0, 610.0])
    values = np.array([0.2, 0.8])

    spectrum = fd.Spectrum(wavelengths=wavelengths, values=values)
    wavelengths[0] = 500
    values[0] = 1

    assert spectrum.wavelengths[0] == 600
    assert spectrum.values[0] == 0.2


def test_spectrum_arrays_are_read_only():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510],
        values=[0.2, 0.8],
    )

    with pytest.raises(ValueError, match="read-only"):
        spectrum.values[0] = 1

    with pytest.raises(ValueError, match="read-only"):
        spectrum.wavelengths[0] = 400


@pytest.mark.parametrize(
    "wavelengths, values, message",
    [
        (
            [[600, 610], [620, 630]],
            [0.1, 0.2, 0.3, 0.4],
            "spectrum wavelengths must be one-dimensional.",
        ),
        (
            [600, 610, 620, 630],
            [[0.1, 0.2], [0.3, 0.4]],
            "spectrum values must be one-dimensional.",
        ),
        (
            [600, 610],
            [0.1, 0.2, 0.3],
            "spectrum wavelengths and values must have the same length.",
        ),
        (
            [600],
            [0.1],
            "a spectrum must contain at least two data points.",
        ),
        (
            [600, np.nan],
            [0.1, 0.2],
            "spectrum wavelengths must be finite.",
        ),
        (
            [600, 610],
            [0.1, np.inf],
            "spectrum values must be finite.",
        ),
        (
            [600, 620, 610],
            [0.1, 0.2, 0.3],
            "spectrum wavelengths must be strictly increasing.",
        ),
        (
            [600, 600, 610],
            [0.1, 0.2, 0.3],
            "spectrum wavelengths must be strictly increasing.",
        ),
        (
            [600, 610],
            [0.1, -0.2],
            "spectrum values must be non-negative.",
        ),
    ],
)
def test_spectrum_errors(wavelengths, values, message):
    with pytest.raises(ValueError, match=message):
        fd.Spectrum(wavelengths=wavelengths, values=values)


def test_init_FluorophoreData():
    fluorophore_data = fd.FluorophoreData()
    assert fluorophore_data.QUANTUM_YIELD == 0
    assert fluorophore_data.emission_spectrum is None
    assert fluorophore_data.absorption_spectra == {}


def test_fluorophore_data_is_frozen():
    fluorophore_data = fd.FluorophoreData(
        QUANTUM_YIELD=0.5,
        FLUORESCENCE_LIFETIME=2e-9,
    )

    with pytest.raises(FrozenInstanceError):
        fluorophore_data.QUANTUM_YIELD = 0.9


def test_absorption_spectra_are_read_only():
    absorption = fd.Spectrum(
        wavelengths=[500, 510],
        values=[1000, 2000],
    )
    fluorophore_data = fd.FluorophoreData(absorption_spectra={"s0": absorption})

    with pytest.raises(TypeError):
        fluorophore_data.absorption_spectra["t1"] = absorption


def test_fluorophore_data_copies_absorption_mapping():
    absorption = fd.Spectrum(
        wavelengths=[500, 510],
        values=[1000, 2000],
    )
    absorption_spectra = {"s0": absorption}
    fluorophore_data = fd.FluorophoreData(absorption_spectra=absorption_spectra)

    absorption_spectra.clear()

    assert fluorophore_data.absorption_spectra == {"s0": absorption}


def test_fluorophore_data_with_spectra():
    emission = fd.Spectrum(
        wavelengths=[600, 610, 620],
        values=[0.1, 1.0, 0.4],
    )
    absorption = fd.Spectrum(
        wavelengths=[500, 510, 520],
        values=[1000, 2000, 500],
    )

    fluorophore_data = fd.FluorophoreData(
        QUANTUM_YIELD=0.7,
        FLUORESCENCE_LIFETIME=3e-9,
        emission_spectrum=emission,
        absorption_spectra={"s0": absorption},
    )

    assert fluorophore_data.emission_spectrum is emission
    assert fluorophore_data.absorption_spectra["s0"] is absorption


def test_fluorophore_data_emission_spectrum_error():
    with pytest.raises(
        TypeError,
        match="emission_spectrum must be a Spectrum or None.",
    ):
        fd.FluorophoreData(emission_spectrum=[0.1, 0.2])


def test_fluorophore_data_absorption_spectrum_error():
    with pytest.raises(
        TypeError,
        match="absorption spectrum for state 's0' must be a Spectrum.",
    ):
        fd.FluorophoreData(absorption_spectra={"s0": [0.1, 0.2]})


@pytest.mark.parametrize(
    "name, value",
    [
        ("QUANTUM_YIELD", -0.1),
        ("QUANTUM_YIELD", 1.1),
        ("STA_EFFICIENCY", 1.1),
        ("RAD_ESCAPE_EFFICIENCY", np.nan),
        ("BISO_EFFICIENCY", np.inf),
    ],
)
def test_fluorophore_data_efficiency_errors(name, value):
    with pytest.raises(
        ValueError,
        match=rf"{name} must be finite and between 0 and 1.",
    ):
        fd.FluorophoreData(**{name: value})


@pytest.mark.parametrize(
    "name, value",
    [
        ("FLUORESCENCE_LIFETIME", -1e-9),
        ("ISC_ST_RATE", -1),
        ("PHOTOBLEACH_T1_RATE", np.nan),
        ("BISO_CROSS_SECTION", np.inf),
    ],
)
def test_fluorophore_data_non_negative_value_errors(name, value):
    with pytest.raises(
        ValueError,
        match=rf"{name} must be finite and non-negative.",
    ):
        fd.FluorophoreData(**{name: value})


def test_partial_fluorophore_data_allows_zero_lifetime():
    fluorophore_data = fd.FluorophoreData()

    assert fluorophore_data.FLUORESCENCE_LIFETIME == 0


def test_deepcopy_fluorophore_data_with_absorption_spectra():
    absorption = fd.Spectrum(
        wavelengths=[500, 510],
        values=[1000, 2000],
    )
    fluorophore_data = fd.FluorophoreData(absorption_spectra={"s0": absorption})

    copied = deepcopy(fluorophore_data)

    assert copied is fluorophore_data


def test_spectrum_from_arrays():
    spectrum = fd.Spectrum.from_arrays(
        wavelengths=[500, 510, 520],
        values=[0.1, 0.8, 0.2],
    )

    np.testing.assert_array_equal(
        spectrum.wavelengths,
        np.array([500.0, 510.0, 520.0]),
    )
    np.testing.assert_array_equal(
        spectrum.values,
        np.array([0.1, 0.8, 0.2]),
    )


def test_spectrum_from_csv(tmp_path):
    path = tmp_path / "spectrum.csv"
    path.write_text("Wavelengths,y\n500,0.1\n510,0.8\n520,0.2\n", encoding="utf-8")

    spectrum = fd.Spectrum.from_csv(path)

    np.testing.assert_array_equal(
        spectrum.wavelengths,
        np.array([500.0, 510.0, 520.0]),
    )
    np.testing.assert_array_equal(
        spectrum.values,
        np.array([0.1, 0.8, 0.2]),
    )


def test_spectrum_from_csv_custom_columns(tmp_path):
    path = tmp_path / "spectrum.csv"
    path.write_text("wavelength,intensity\n500,0.1\n510,0.8\n", encoding="utf-8")

    spectrum = fd.Spectrum.from_csv(
        path,
        wavelength_column="wavelength",
        value_column="intensity",
    )

    np.testing.assert_array_equal(
        spectrum.wavelengths,
        np.array([500.0, 510.0]),
    )
    np.testing.assert_array_equal(
        spectrum.values,
        np.array([0.1, 0.8]),
    )


def test_spectrum_from_csv_missing_column(tmp_path):
    path = tmp_path / "spectrum.csv"
    path.write_text("wavelength,intensity\n500,0.1\n510,0.8\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="spectrum CSV is missing columns: Wavelengths, y.",
    ):
        fd.Spectrum.from_csv(path)


def test_spectrum_from_existing_csv():
    data_dir = Path(fd.__file__).parent / "fluorophore_spectra" / "testing_data_1"

    spectrum = fd.Spectrum.from_csv(data_dir / "emission.csv")

    assert spectrum.wavelengths[0] == 200
    assert spectrum.wavelengths[-1] == 1000
    assert spectrum.wavelengths.size == spectrum.values.size


def test_spectrum_at_existing_wavelength():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510, 520],
        values=[1000, 2000, 500],
    )

    assert spectrum.at(510) == 2000


def test_spectrum_at_interpolated_wavelength():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510, 520],
        values=[1000, 2000, 500],
    )

    assert spectrum.at(505) == 1500


def test_spectrum_at_boundaries():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510, 520],
        values=[1000, 2000, 500],
    )

    assert spectrum.at(500) == 1000
    assert spectrum.at(520) == 500


@pytest.mark.parametrize("wavelength", [499, 521])
def test_spectrum_at_outside_range(wavelength):
    spectrum = fd.Spectrum(
        wavelengths=[500, 510, 520],
        values=[1000, 2000, 500],
    )

    with pytest.raises(
        ValueError,
        match="is outside the spectrum range",
    ):
        spectrum.at(wavelength)


def test_spectrum_at_non_finite_wavelength():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510],
        values=[1000, 2000],
    )

    with pytest.raises(ValueError, match="wavelength must be finite."):
        spectrum.at(np.nan)


def test_spectrum_integral():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510, 520],
        values=[0, 1, 0],
    )

    assert spectrum.integral() == pytest.approx(10)


def test_spectrum_partial_integral():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510, 520],
        values=[0, 1, 0],
    )

    assert spectrum.integral(505, 515) == pytest.approx(7.5)


def test_spectrum_integral_clips_to_range():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510, 520],
        values=[0, 1, 0],
    )

    assert spectrum.integral(400, 600) == pytest.approx(10)
    assert spectrum.integral(400, 450) == 0


def test_spectrum_integral_limits_error():
    spectrum = fd.Spectrum(
        wavelengths=[500, 510],
        values=[0, 1],
    )

    with pytest.raises(
        ValueError,
        match=("the lower integration limit must be smaller than the upper limit."),
    ):
        spectrum.integral(510, 500)


def test_init_cy5_dna():
    fluorophore_data = fd.cy5_dna
    # print(fluophore_data.__doc__)
    assert fluorophore_data.QUANTUM_YIELD == 0.27


def test_init_atto643():
    fluorophore_data = fd.atto643
    assert fluorophore_data.QUANTUM_YIELD == 0.6
