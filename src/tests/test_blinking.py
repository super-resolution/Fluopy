# import pytest
# import numpy as np
# import pandas as pd
# import src.blinking as bl


# @pytest.mark.parametrize('parameters,use_series,expected',
#                          [[[0, 0], 0, [np.array([4, 1]), np.array([1]), np.array([1, 6]), np.array([5])]],
#                           [[2, 0], 0, [np.array([3, 1]), np.array([1]), np.array([2, 6]), np.array([5])]],
#                           [[0, 1], 0, [np.array([6]), np.array([]), np.array([1]), np.array([])]],
#                           [[0, 0], 1, [np.array([4]), np.array([1]), np.array([1]), np.array([5])]],
#                           [[0, 2], 0, [np.array([6]), np.array([]), np.array([1]), np.array([])]],
#                           [[0, 0], 2, [np.array([]), np.array([]), np.array([]), np.array([])]],
#                           [[0, 0], 3, [np.array([3, 2]), np.array([1, 1]), np.array([0, 4]), np.array([3, 6])]],
#                           [[1, 0], 3, [np.array([2, 2]), np.array([1, 1]), np.array([1, 4]), np.array([3, 6])]],
#                           [[1, 1], 4, [np.array([2, 3]), np.array([2]), np.array([1, 5]), np.array([3])]]])
# def test_get_blinking_statistics(parameters, use_series, expected):
#     event_time_series_0 = pd.Series(np.array([0, 1, 3, 4, 7, 0, 4, 0, 0]), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
#     event_time_series_1 = pd.Series(np.array([0, 1, 3, 4, 7, 0, 4, 5]), np.array([0, 1, 2, 3, 4, 5, 6, 7]))
#     event_time_series_2 = pd.Series(np.array([0, 1, 3, 4, 7, 1, 4, 5]), np.array([0, 1, 2, 3, 4, 5, 6, 7]))
#     event_time_series_3 = pd.Series(np.array([1, 2, 3, 0, 4, 5, 0, 3]), np.array([0, 1, 2, 3, 4, 5, 6, 7]))
#     event_time_series_4 = pd.Series(np.array([1, 2, 3, 0, 0, 5, 0, 3, 0]), np.array([0, 1, 2, 3, 4, 5, 6, 7, 8]))
#     event_time_series = [event_time_series_0, event_time_series_1, event_time_series_2, event_time_series_3,
#                          event_time_series_4]
#     on_periods, off_periods, on_periods_frames, off_periods_frames = \
#         bl.get_blinking_statistics(event_time_series[use_series], *parameters)
#     np.testing.assert_allclose(on_periods, expected[0])
#     np.testing.assert_allclose(off_periods, expected[1])
#     np.testing.assert_allclose(on_periods_frames, expected[2])
#     np.testing.assert_allclose(off_periods_frames, expected[3])
