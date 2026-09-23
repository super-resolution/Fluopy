from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from matplotlib.axes import Axes as mplAxes

from fluopy import fcs as fcs_p


@pytest.fixture()
def small_emissions():
    return SimpleNamespace(
        event_time_points=np.array([0.1, 0.4, 0.9, 1.6], dtype=np.float64),
        event_time_series=pd.Series(
            [1, 2, 3], index=np.array([0.0, 0.5, 1.0], dtype=np.float64)
        ),
    )


def test_fcs(em_very_large):
    fcs_obj = fcs_p.FCS(emissions=em_very_large)
    assert fcs_obj.emissions == em_very_large
    assert fcs_obj.autocorrelation is None
    assert fcs_obj.tau is None


def test_fcs_plot(small_emissions):
    fcs_obj = fcs_p.FCS(emissions=small_emissions)
    fcs_obj.tau = np.array([0.001, 0.002])
    fcs_obj.autocorrelation = np.array([2.0, 1.0])

    ax = fcs_obj.plot_matplotlib(normalize_to=0, unit="ms")
    assert isinstance(ax, mplAxes)
    np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1.0, 2.0])
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [1.0, 0.5])
    assert ax.get_xlabel() == r"$\tau \ (ms)$"

    ax = fcs_obj.plot(normalize_to=1)
    np.testing.assert_array_equal(ax.lines[0].get_ydata(), [2.0, 1.0])


@pytest.mark.parametrize("method", ["plot", "plot_matplotlib"])
def test_fcs_plot_requires_correlation_data(small_emissions, method):
    fcs_obj = fcs_p.FCS(emissions=small_emissions)

    with pytest.raises(RuntimeError, match="has not been calculated"):
        getattr(fcs_obj, method)()


@pytest.mark.parametrize(
    "attribute, method",
    [
        ("event_time_points", "autocorrelate_time_points"),
        ("event_time_series", "autocorrelate_time_series"),
    ],
)
def test_fcs_autocorrelation_requires_emission_data(small_emissions, attribute, method):
    setattr(small_emissions, attribute, None)
    fcs_obj = fcs_p.FCS(emissions=small_emissions)

    with pytest.raises(ValueError, match=f"{attribute} is None"):
        getattr(fcs_obj, method)()


def test_fcs_autocorrelate_time_points(em_very_large, caplog):
    fcs_obj = fcs_p.FCS(emissions=em_very_large)

    fcs_obj.autocorrelate_time_points(
        exp_min=-8, exp_max=0, points_per_base=4, base=10, normalize=True
    )
    assert caplog.record_tuples == [
        (
            "fluopy.fcs",
            30,
            "The exp_max 0 yields a base to the power of exp_max 1 that is larger than the duration of the measurement: 0.1676677881662439. Therefore, exp_max is adjusted to -1.",
        )
    ]

    fcs_obj.autocorrelate_time_points(
        exp_min=-8, exp_max=-2, points_per_base=4, base=10, normalize=True
    )

    exp_tau = np.array(
        [
            1.38913971e-08,
            2.47027854e-08,
            4.39284546e-08,
            7.81170663e-08,
            1.38913971e-07,
            2.47027854e-07,
            4.39284546e-07,
            7.81170663e-07,
            1.38913971e-06,
            2.47027854e-06,
            4.39284546e-06,
            7.81170663e-06,
            1.38913971e-05,
            2.47027854e-05,
            4.39284546e-05,
            7.81170663e-05,
            1.38913971e-04,
            2.47027854e-04,
            4.39284546e-04,
            7.81170663e-04,
            1.38913971e-03,
            2.47027854e-03,
            4.39284546e-03,
            7.81170663e-03,
        ]
    )

    exp_autocorrelation = np.array(
        [
            2.06435690,
            2.01768995,
            2.12727187,
            2.05275592,
            1.99816576,
            1.98797910,
            1.95591432,
            1.88561919,
            1.76440802,
            1.59249428,
            1.38506233,
            1.18985137,
            1.06280453,
            1.00597039,
            1.00242705,
            1.00037999,
            0.99954654,
            0.99949195,
            1.00278158,
            1.00227431,
            1.00252596,
            1.00257885,
            1.01024521,
            1.01470773,
        ]
    )

    np.testing.assert_allclose(fcs_obj.tau, exp_tau, rtol=1e-7)
    np.testing.assert_allclose(fcs_obj.autocorrelation, exp_autocorrelation, rtol=1e-7)

    fcs_obj.autocorrelate_time_points(
        exp_min=-8, exp_max=-2, points_per_base=4, base=10, normalize=False
    )
    exp_autocorrelation = np.array(
        [
            1.96715984e11,
            1.92269029e11,
            2.02711302e11,
            1.95609055e11,
            1.90407196e11,
            1.89436656e11,
            1.86381442e11,
            1.79683403e11,
            1.68131134e11,
            1.51745857e11,
            1.31973631e11,
            1.13373510e11,
            1.01267101e11,
            9.58528669e10,
            9.55156639e10,
            9.53070335e10,
            9.51821660e10,
            9.50988486e10,
            9.52973257e10,
            9.50896884e10,
            9.47900356e10,
            9.40562215e10,
            9.30588163e10,
            9.07828090e10,
        ]
    )
    np.testing.assert_allclose(fcs_obj.autocorrelation, exp_autocorrelation, rtol=1e-5)


def test_fcs_autocorrelate_time_series(em_very_large):
    fcs_obj = fcs_p.FCS(emissions=em_very_large)
    fcs_obj.autocorrelate_time_series(log=True, m=4, normalize=True)
    exp_tau = np.array(
        [
            1.000e-03,
            2.000e-03,
            3.000e-03,
            4.000e-03,
            6.000e-03,
            8.000e-03,
            1.200e-02,
            1.600e-02,
            2.400e-02,
            3.200e-02,
            4.800e-02,
            6.400e-02,
            9.600e-02,
            1.280e-01,
            1.920e-01,
            2.560e-01,
            3.840e-01,
            5.120e-01,
            7.680e-01,
            1.024e00,
            1.536e00,
            2.048e00,
            3.072e00,
            4.096e00,
        ]
    )

    exp_autocorrelation = np.array(
        [
            5.94214309e01,
            5.91541007e01,
            5.87133369e01,
            5.84414754e01,
            5.75675514e01,
            5.68095541e01,
            5.54987281e01,
            5.40390608e01,
            5.12610578e01,
            4.84373327e01,
            4.27520521e01,
            3.71776150e01,
            2.65483709e01,
            1.53279999e01,
            9.78758170e-01,
            9.72039474e-01,
            9.58333333e-01,
            9.44256757e-01,
            9.14930556e-01,
            8.83928571e-01,
            7.79296875e-01,
            6.97916667e-01,
            3.72395833e-01,
            4.68750000e-02,
        ]
    )

    np.testing.assert_allclose(fcs_obj.tau, exp_tau, rtol=1e-7)
    np.testing.assert_allclose(fcs_obj.autocorrelation, exp_autocorrelation, rtol=1e-7)

    fcs_obj.autocorrelate_time_series(log=True, m=4, normalize=False)
    exp_autocorrelation = np.array(
        [
            9.49218090e07,
            9.44780100e07,
            9.37549760e07,
            9.33036960e07,
            9.18677525e07,
            9.06186670e07,
            8.84539390e07,
            8.60521292e07,
            8.14782139e07,
            7.68382531e07,
            6.75248142e07,
            5.84173066e07,
            4.11230254e07,
            2.29969024e07,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        ]
    )
    np.testing.assert_allclose(fcs_obj.autocorrelation, exp_autocorrelation, rtol=1e-5)


@pytest.mark.parametrize(
    "normalize, expected",
    [
        (False, [8.0, 3.0]),
        (True, [1.0, 0.75]),
    ],
)
def test_fcs_autocorrelate_linear_time_series(small_emissions, normalize, expected):
    fcs_obj = fcs_p.FCS(emissions=small_emissions)

    returned = fcs_obj.autocorrelate_time_series(log=False, normalize=normalize)

    assert returned is fcs_obj
    np.testing.assert_array_equal(fcs_obj.tau, [0.5, 1.0])
    np.testing.assert_allclose(fcs_obj.autocorrelation, expected)


def test_fit_dark():
    exp_autocorrelation = np.array([0.485225, 0.294304, 0.178504])
    exp_norm = 0.2
    autocorrelation, norm = fcs_p.fit_dark(
        tau=[1, 2, 3], dark_lifetime=2, dark_occupation=0.8
    )
    np.testing.assert_allclose(autocorrelation, exp_autocorrelation, rtol=1e-5)
    np.testing.assert_allclose(norm, exp_norm, rtol=1e-5)


@pytest.mark.parametrize("dark_occupation", [-0.1, 1.0])
def test_fit_dark_rejects_invalid_occupation(dark_occupation):
    with pytest.raises(ValueError, match="between 0 and 1"):
        fcs_p.fit_dark(tau=[1], dark_lifetime=2, dark_occupation=dark_occupation)


def test_fit_antibunching():
    exp_autocorrelation = np.array([-0.2725, -0.0742, -0.0202])
    autocorrelation = fcs_p.fit_antibunching(
        tau=[1, 2, 3], excitation_rate=0.8, s1_lifetime=2
    )
    np.testing.assert_allclose(autocorrelation, exp_autocorrelation, rtol=1e-2)


def test_fit_triplet_cis():
    exp_autocorrelation = np.array([0.5143, 0.5010, 0.3860])
    autocorrelation, norm = fcs_p.fit_triplet_cis(
        tau=[1e-8, 1e-7, 1e-6],
        k_isc=8.3e5,
        k_T=5e5,
        k_01=7.27e6,
        k_10=1e9,
        k_iso=2e7,
        k_biso_eff=1.37e5,
    )
    np.testing.assert_allclose(autocorrelation, exp_autocorrelation, rtol=1e-3)
    np.testing.assert_allclose(norm, 0.4841126, rtol=1e-4)


def test_coincidence_backends_agree_for_controlled_events():
    detector_1 = np.array([0.0, 1.0, 3.0])
    detector_2 = np.array([0.2, 1.4, 2.6])

    numpy_hist, numpy_bins = fcs_p.coincidence_numpy(
        detector_1, detector_2, tau_max=0.75, bin_width=0.5
    )
    numba_hist, numba_bins = fcs_p.coincidence_numba(
        detector_1, detector_2, tau_max=0.75, bin_width=0.5
    )

    np.testing.assert_array_equal(numba_bins, numpy_bins)
    np.testing.assert_array_equal(numba_hist, numpy_hist)
    np.testing.assert_array_equal(numpy_hist, [1.0, 1.0, 1.0])


def test_event_time_correlation_python_implementation():
    first = np.array([0.0, 2.0])
    second = np.array([0.5, 1.5, 3.0])
    bins = np.array([0.0, 1.0, 2.0])

    unnormalized = fcs_p._event_time_correlation.py_func(
        first,
        second,
        bins,
        False,
        0.0,
        4.0,
    )
    normalized = fcs_p._event_time_correlation.py_func(
        first,
        second,
        bins,
        True,
        0.0,
        4.0,
    )

    np.testing.assert_array_equal(unnormalized, [1.0, 2.0])
    np.testing.assert_array_equal(normalized, [0.75, 2.0])


def test_coincidence_numba_python_implementation_matches_numpy():
    detector_1 = np.array([0.0, 3.0])
    detector_2 = np.array([0.2, 1.4, 2.6])

    expected_hist, expected_bins = fcs_p.coincidence_numpy(
        detector_1, detector_2, tau_max=0.75, bin_width=0.5
    )
    actual_hist, actual_bins = fcs_p.coincidence_numba.py_func(
        detector_1, detector_2, tau_max=0.75, bin_width=0.5
    )

    np.testing.assert_array_equal(actual_bins, expected_bins)
    np.testing.assert_array_equal(actual_hist, expected_hist)


@pytest.mark.parametrize("method", ["numpy", "numba"])
def test_coincidence_is_reproducible_and_normalized(method):
    arrival_times = np.linspace(0.1, 9.9, 100)

    first = fcs_p.coincidence(
        arrival_times,
        tau_max=0.5,
        bin_width=0.1,
        seed=4,
        method=method,
        end_time=10,
    )
    second = fcs_p.coincidence(
        arrival_times,
        tau_max=0.5,
        bin_width=0.1,
        seed=4,
        method=method,
        end_time=10,
    )

    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    assert np.all(np.isfinite(first[0]))


def test_coincidence_rejects_unknown_method():
    with pytest.raises(ValueError, match="Unknown method: invalid"):
        fcs_p.coincidence(
            np.array([0.1, 0.2]),
            tau_max=0.1,
            bin_width=0.01,
            seed=1,
            method="invalid",
        )
