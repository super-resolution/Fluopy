import pytest
import numpy as np
from fluopy import formulas as fo


@pytest.mark.parametrize(
    "wavenumber, wavelength, frequency, expected",
    [
        [None, None, None, "ValueError"],
        [1, 1, 1, "ValueError"],
        [1, 1, None, "ValueError"],
        [1, None, None, (np.array(1), np.array(10000000), np.array(2.9979e10))],
        [
            None,
            [1e3, 2e3],
            None,
            (
                np.array([10000, 5000]),
                np.array([1000, 2000]),
                np.array([2.9979e14, 1.499e14]),
            ),
        ],
    ],
)
def test_convert_wavenumber_wavelength_frequency(
    wavenumber, wavelength, frequency, expected
):
    if expected == "ValueError":
        with pytest.raises(ValueError):
            fo.convert_wavenumber_wavelength_frequency(
                wavenumber=wavenumber, wavelength=wavelength, frequency=frequency
            )
    else:
        result = fo.convert_wavenumber_wavelength_frequency(
            wavenumber=wavenumber, wavelength=wavelength, frequency=frequency
        )
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-4)


@pytest.mark.parametrize(
    "irradiance, frequency, expected",
    [[2, 4.5e14, np.array(6.7075e25)], [[0, 1], 1e6, np.array([0, 1.5092e34])]],
)
def test_calculate_photon_flux(irradiance, frequency, expected):
    result = fo.calculate_photon_flux(irradiance=irradiance, frequency=frequency)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "photon_flux, extinction_coefficient, absorption_cross_section ,expected",
    [
        [1e10, None, None, "ValueError"],
        [1e10, 1, 1, "ValueError"],
        [1e10, 1, None, 3.8235e-15],
        [1e10, None, 1, 1000000],
        [[1e10, 1e5], None, 1, np.array([1e6, 1e1])],
        [1e10, [1, 5], None, np.array([3.8235e-15, 1.91177e-14])],
    ],
)
def test_calculate_excitation_rate(
    photon_flux, extinction_coefficient, absorption_cross_section, expected
):
    if isinstance(expected, str):
        if expected == "ValueError":
            with pytest.raises(ValueError):
                fo.calculate_excitation_rate(
                    photon_flux=photon_flux,
                    extinction_coefficient=extinction_coefficient,
                    absorption_cross_section=absorption_cross_section,
                )
    else:
        result = fo.calculate_excitation_rate(
            photon_flux=photon_flux,
            extinction_coefficient=extinction_coefficient,
            absorption_cross_section=absorption_cross_section,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize(
    "quantum_yield, fluorescence_lifetime, expected",
    [
        [1, 2, 0.5],
        [[1, 2], 2, np.array([0.5, 1])],
        [[1, 2], [2, 4], np.array([0.5, 0.5])],
        [1, [2, 4], np.array([0.5, 0.25])],
    ],
)
def test_calculate_emission_rate(quantum_yield, fluorescence_lifetime, expected):
    result = fo.calculate_emission_rate(
        quantum_yield=quantum_yield, fluorescence_lifetime=fluorescence_lifetime
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "quantum_yield, emission_rate, other_outgoing_rate, expected",
    [
        [0.5, 1e5, 0, 100000],
        [[0.5, 0.1], 1e5, 0, np.array([100000, 900000])],
        [0.5, 1e5, 1e5, 0],
        [0.5, 1e5, 2e5, "ValueError"],
        [1.4, 1e5, 1e5, "ValueError"],
    ],
)
def test_calculate_internal_conversion_rate(
    quantum_yield, emission_rate, other_outgoing_rate, expected
):
    if isinstance(expected, str):
        if expected == "ValueError":
            with pytest.raises(ValueError):
                fo.calculate_internal_conversion_rate(
                    quantum_yield=quantum_yield,
                    emission_rate=emission_rate,
                    other_outgoing_rate=other_outgoing_rate,
                )
    else:
        result = fo.calculate_internal_conversion_rate(
            quantum_yield=quantum_yield,
            emission_rate=emission_rate,
            other_outgoing_rate=other_outgoing_rate,
        )
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize(
    "photon_flux, absorption_cross_section, expected",
    [
        [0.5, 2, 0.0001],
        [[0.5, 1], 2, np.array([0.0001, 0.0002])],
        [[0.5, 1], [2, 1], np.array([0.0001, 0.0001])],
    ],
)
def test_calculate_back_isomerization_rate(
    photon_flux, absorption_cross_section, expected
):
    result = fo.calculate_back_isomerization_rate(
        photon_flux=photon_flux, absorption_cross_section=absorption_cross_section
    )
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("ph, pka, concentration, expected", [[8, 9.6, 143, 3.50398]])
def test_henderson_hasselbalch_equation(ph, pka, concentration, expected):
    base_concentration = fo.henderson_hasselbalch_equation(
        ph=ph, pka=pka, concentration=concentration
    )
    np.testing.assert_allclose(base_concentration, expected, rtol=1e-4)


@pytest.mark.parametrize(
    "reducing_agent, concentration, k_pet, ph, expected",
    [
        ["ßme", 100, 1, 8, "ValueError"],
        ["test", 100, 1, 8, 0.003065],
        ["betaME", 100, 1, 8, 0.00245],
    ],
)
def test_calculate_pet_rate(reducing_agent, concentration, k_pet, ph, expected):
    if expected == "ValueError":
        with pytest.raises(ValueError):
            fo.calculate_pet_rate(
                reducing_agent=reducing_agent,
                concentration=concentration,
                k_pet=k_pet,
                ph=ph,
            )
    else:
        result = fo.calculate_pet_rate(
            reducing_agent=reducing_agent,
            concentration=concentration,
            k_pet=k_pet,
            ph=ph,
        )
        np.testing.assert_allclose(result, expected, rtol=1e-3)


@pytest.mark.parametrize(
    "donor, acceptor, wavelengths, expected",
    [
        [[1, 2, 3], [1], [1, 2, 3], "ValueError"],
        [[1, 2, 3], [2, 3, 4], [3], "ValueError"],
        [[1, 2, 3], [2, 3, 4], [100, 200, 300], 1.4575e10],
    ],
)
def test_calculate_spectral_overlap_integral(donor, acceptor, wavelengths, expected):
    if expected == "ValueError":
        with pytest.raises(ValueError):
            fo.calculate_spectral_overlap_integral(
                donor=donor, acceptor=acceptor, wavelengths=wavelengths
            )
    else:
        result = fo.calculate_spectral_overlap_integral(
            donor=donor, acceptor=acceptor, wavelengths=wavelengths
        )
        np.testing.assert_allclose(result, expected)


def test_calculate_fret_rate():
    result = fo.calculate_fret_rate(
        distance=10,
        emission_rate=5e8,
        spectral_overlap_integral=1e16,
        dipole_orientation_factor=2 / 3,
        refractive_index=1,
    )
    np.testing.assert_allclose(result, 292833333.3333, rtol=1e-4)


def test_calculate_fret_efficiency():
    result = fo.calculate_fret_efficiency(fret_rate=1e8, fluorescence_lifetime=1e-8)
    np.testing.assert_allclose(result, 0.5)


def test_calcualte_photon_collection_rate():
    result = fo.calculate_photon_collection_rate(NA=1.45, n1=1.51)
    np.testing.assert_allclose(result, 0.3604549)
