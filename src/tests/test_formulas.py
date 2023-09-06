import os
import sys
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

import pytest
import numpy as np
import formulas as fo


@pytest.mark.parametrize('parameters,expected',
                         [[[None, None, None], 'ValueError'],
                          [[1, 1, 1], 'ValueError'],
                          [[1, 1, None], 'ValueError'],
                          [[1, None, None], (np.array(1), np.array(10000000), np.array(2.9979e10))],
                          [[None, [1e3, 2e3], None], (np.array([10000, 5000]), np.array([1000, 2000]),
                                                      np.array([2.9979e14, 1.499e14]))]])
def test_convert_wavenumber_wavelength_frequency(parameters, expected):
    if expected == 'ValueError':
        with pytest.raises(ValueError):
            fo.convert_wavenumber_wavelength_frequency(*parameters)
    else:
        result = fo.convert_wavenumber_wavelength_frequency(*parameters)
        np.testing.assert_allclose(np.array(result), np.array(expected), rtol=1e-4)


@pytest.mark.parametrize('parameters,expected',
                         [[[2, 4.5e14], np.array(6.7075e25)],
                          [[[0, 1], 1e6], np.array([0, 1.5092e34])]])
def test_calculate_photon_flux(parameters, expected):
    result = fo.calculate_photon_flux(*parameters)
    np.testing.assert_allclose(result, expected, rtol=1e-4)


@pytest.mark.parametrize('parameters,expected',
                         [[[1e10, None, None], 'ValueError'],
                          [[1e10, 1, 1], 'ValueError'],
                          [[1e10, 1, None], 3.8235e-15],
                          [[1e10, None, 1], 1000000],
                          [[[1e10, 1e5], None, 1], np.array([1e6, 1e1])],
                          [[1e10, [1, 5], None], np.array([3.8235e-15, 1.91177e-14])]])
def test_calculate_excitation_rate(parameters, expected):
    if expected == 'ValueError':
        with pytest.raises(ValueError):
            fo.calculate_excitation_rate(*parameters)
    else:
        result = fo.calculate_excitation_rate(*parameters)
        np.testing.assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.parametrize('parameters,expected',
                         [[[1, 2], 0.5],
                          [[[1, 2], 2], np.array([0.5, 1])],
                          [[[1, 2], [2, 4]], np.array([0.5, 0.5])],
                          [[1, [2, 4]], np.array([0.5, 0.25])]])
def test_calculate_emission_rate(parameters, expected):
    result = fo.calculate_emission_rate(*parameters)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('parameters,expected',
                         [[[0.5, 1e5], 100000],
                          [[[0.5, 0.1], 1e5], np.array([100000, 900000])],
                          [[0.5, 1e5, 1e5], 0],
                          [[0.5, 1e5, 2e5], 'ValueError'],
                          [[1.4, 1e5, 1e5], 'ValueError']])
def test_calculate_internal_conversion_rate(parameters, expected):
    if expected == 'ValueError':
        with pytest.raises(ValueError):
            fo.calculate_internal_conversion_rate(*parameters)
    else:
        result = fo.calculate_internal_conversion_rate(*parameters)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('parameters,expected',
                         [[[0.5, 2], 0.0001],
                          [[[0.5, 1], 2], np.array([0.0001, 0.0002])],
                          [[[0.5, 1], [2, 1]], np.array([0.0001, 0.0001])]])
def test_calculate_back_isomerization_rate(parameters, expected):
    result = fo.calculate_back_isomerization_rate(*parameters)
    np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('parameters,expected',
                         [[['mea', 100, 1, 7, True], 'AttributeError'],
                          [['mea', 100, 1, 8, True], 0.1],
                          [['βME', 100, 1, 8, True], 0.1],
                          [['βME', 100, 1, 8, False], 0.7]])
def test_calculate_reduction_rate(parameters, expected):
    if expected == 'AttributeError':
        with pytest.raises(AttributeError):
            fo.calculate_reduction_rate(*parameters)
    else:
        result = fo.calculate_reduction_rate(*parameters)
        np.testing.assert_allclose(result, expected)


@pytest.mark.parametrize('parameters,expected',
                         [[[[1, 2, 3], [1], [1, 2, 3]], 'AttributeError'],
                          [[[1, 2, 3], [2, 3, 4], [3]], 'AttributeError'],
                          [[[1, 2, 3], [2, 3, 4], [100, 200, 300]], 1690098112]])
def test_calculate_spectral_overlap_integral(parameters, expected):
    if expected == 'AttributeError':
        with pytest.raises(AttributeError):
            fo.calculate_spectral_overlap_integral(*parameters)
    else:
        result = fo.calculate_spectral_overlap_integral(*parameters)
        np.testing.assert_allclose(result, expected)


def test_calculate_fret_rate():
    result = fo.calculate_fret_rate(10, 5e8, 1e16, 2/3, 1)
    np.testing.assert_allclose(result, 292833333.3333, rtol=1e-4)


def test_calculate_fret_efficiency():
    result = fo.calculate_fret_efficiency(1e8, 1e-8)
    np.testing.assert_allclose(result, 0.5)
