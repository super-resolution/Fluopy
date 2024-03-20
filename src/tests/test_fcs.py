# import pytest
# import numpy as np
# import src.fcs as fcs


# def test_fcs(fcs_object):
#     assert fcs_object.autocorrelation is None
#     assert fcs_object.tau is None


# @pytest.mark.parametrize('parameters,expected',
#                          [[[-10, 2, 4, 10, True], 'ValueError'],
#                           [[-10, 1, 4, 10, True], '']])
# def test_autocorrelate_time_points(fcs_object, parameters, expected):
#     if expected == 'ValueError':
#         with pytest.raises(ValueError):
#             fcs_object.autocorrelate_time_points(*parameters)
#     else:
#         fcs_object.autocorrelate_time_points(*parameters)
#         assert fcs_object.autocorrelation.size == fcs_object.tau.size


# @pytest.mark.parametrize('parameters',
#                          [[True, 4, True],
#                           [True, 4, False],
#                           [False, 4, True],
#                           [False, 4, False]])
# def test_autocorrelate_time_series(fcs_object, parameters):
#     fcs_object.autocorrelate_time_series(*parameters)
#     assert fcs_object.autocorrelation.size == fcs_object.tau.size


# @pytest.mark.parametrize('parameters,expected',
#                          [[[[1, 2, 3], 2, 1.1], 'ValueError'],
#                           [[[1, 2, 3], 2, 0.8], [np.array([0.4852, 0.2943, 0.1785]), 0.2]]])
# def test_fit_dark(parameters, expected):
#     if expected == 'ValueError':
#         with pytest.raises(ValueError):
#             fcs.fit_dark(*parameters)
#     else:
#         auto, norm = fcs.fit_dark(*parameters)
#         np.testing.assert_allclose(auto, expected[0], rtol=1e-4)
#         np.testing.assert_allclose(norm, expected[1], rtol=1e-4)


# def test_fit_antibunching():
#     auto = fcs.fit_antibunching([1, 2, 3], 0.8, 2)
#     np.testing.assert_allclose(auto, np.array([-0.2725, -0.0742, -0.0202]), rtol=1e-2)


# def test_fit_triplet_cis():
#     auto, norm = fcs.fit_triplet_cis(tau=[1e-8, 1e-7, 1e-6], k_isc=8.3e5, k_T=5e5, k_01=7.27e6, k_10=1e9, k_iso=2e7,
#                                      k_biso_eff=1.37e5)
#     np.testing.assert_allclose(auto, np.array([0.5143, 0.5010, 0.3860]), rtol=1e-3)
#     np.testing.assert_allclose(norm, 0.4841126, rtol=1e-4)
