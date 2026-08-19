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


def test_spectrum_copies_input_arrays():
    wavelengths = np.array([600.0, 610.0])
    values = np.array([0.2, 0.8])

    spectrum = fd.Spectrum(wavelengths=wavelengths, values=values)
    wavelengths[0] = 500
    values[0] = 1

    assert spectrum.wavelengths[0] == 600
    assert spectrum.values[0] == 0.2


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


def test_init_cy5_dna():
    fluorophore_data = fd.cy5_dna
    # print(fluophore_data.__doc__)
    assert fluorophore_data.QUANTUM_YIELD == 0.27


def test_init_atto643():
    fluorophore_data = fd.atto643
    assert fluorophore_data.QUANTUM_YIELD == 0.6
