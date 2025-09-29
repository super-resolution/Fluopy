import matplotlib.pyplot as plt  # needed for visual inspection  # noqa: F401
import numpy as np
from matplotlib.axes import Axes as mplAxes

from fluopy import fcs as fcs_p


def test_fcs(em_very_large):
    fcs_obj = fcs_p.FCS(emissions=em_very_large)
    assert fcs_obj.emissions == em_very_large
    assert fcs_obj.autocorrelation is None
    assert fcs_obj.tau is None


def test_fcs_plot(em_very_large):
    fcs_obj = fcs_p.FCS(emissions=em_very_large)
    fcs_obj.autocorrelate_time_points(
        exp_min=-8, exp_max=-2, points_per_base=4, base=10, normalize=True
    )
    ax = fcs_obj.plot_matplotlib()
    assert isinstance(ax, mplAxes)
    # plt.show()

    fcs_obj.plot()
    # plt.show()


def test_fcs_autocorrelate_time_points(em_very_large):
    fcs_obj = fcs_p.FCS(emissions=em_very_large)
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
            2.06433366,
            2.01766724,
            2.12724792,
            2.05273281,
            1.99814326,
            1.98795672,
            1.9558923,
            1.88559796,
            1.76438816,
            1.59247635,
            1.38504674,
            1.18983797,
            1.06279256,
            1.00595906,
            1.00241576,
            1.00036872,
            0.99953527,
            0.99948068,
            1.00277025,
            1.00226295,
            1.00251455,
            1.00256734,
            1.01023344,
            1.01469558,
        ]
    )

    np.testing.assert_array_almost_equal(fcs_obj.tau, exp_tau)
    np.testing.assert_array_almost_equal(fcs_obj.autocorrelation, exp_autocorrelation)

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

    np.testing.assert_array_almost_equal(fcs_obj.tau, exp_tau)
    np.testing.assert_array_almost_equal(fcs_obj.autocorrelation, exp_autocorrelation)

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
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
            1.00000000e00,
        ]
    )
    np.testing.assert_allclose(fcs_obj.autocorrelation, exp_autocorrelation, rtol=1e-5)


def test_fit_dark():
    exp_autocorrelation = np.array([0.485225, 0.294304, 0.178504])
    exp_norm = 0.2
    autocorrelation, norm = fcs_p.fit_dark(
        tau=[1, 2, 3], dark_lifetime=2, dark_occupation=0.8
    )
    np.testing.assert_allclose(autocorrelation, exp_autocorrelation, rtol=1e-5)
    np.testing.assert_allclose(norm, exp_norm, rtol=1e-5)


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
