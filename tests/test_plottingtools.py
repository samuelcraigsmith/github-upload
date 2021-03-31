"""Test plottingtools."""

import unittest
from qcext import plottingtools as pt
import math
from collections import Counter
from parameterized import parameterized


def compute_f(n_runs, n_fails):
    """Compute f."""
    return n_fails / n_runs


def compute_df(n_runs, n_fails, c=0.01):
    """Compute df."""
    f = compute_f(n_runs, n_fails)
    return math.sqrt(max(f, c) * (1 - max(f, c)) / n_runs)


class TestExtractThreshold(unittest.TestCase):
    def setUp(self):
        self.data_set = [  # fake data.
            [{"error_probability": 0.001, "logical_failure_rate": 0.13133333333333333, "n_fail": 197, "n_k_d": [61, 1, 9], "n_run": 1500, "n_success": 1303, "time_steps": 150}],
            [{"error_probability": 0.001, "logical_failure_rate": 0.3006666666666667, "n_fail": 451, "n_k_d": [61, 1, 9], "n_run": 1500, "n_success": 1049, "time_steps": 250}],
            [{"error_probability": 0.001, "logical_failure_rate": 0.09, "n_fail": 135, "n_k_d": [217, 1, 17], "n_run": 1500, "n_success": 1365, "time_steps": 150,}], #  d=17, time_steps=150
            [{"error_probability": 0.001, "logical_failure_rate": 0.29933333333333334, "n_fail": 449, "n_k_d": [217, 1, 17], "n_run": 1500, "n_success": 1051, "time_steps": 250,}]
        ]
        self.data_set_batch = self.data_set + [  # more fake data d=9, time_steps=150
            [{"error_probability": 0.001, "logical_failure_rate": 0.09666666666666666, "n_fail": 145, "n_k_d": [61, 1, 9], "n_run": 1500, "n_success": 1355, "time_steps": 150}],
        ]


    def test_extract_threshold_x_data_error_probabilities(self):
        threshold_data, labels = pt.extract_threshold_data(self.data_set,
                                                           x_data="time_steps")
        expected_threshold_data = {
            "9": [
                [150, 250],
                [compute_f(1500, 197), compute_f(1500, 451)],
                [compute_df(1500, 197), compute_df(1500, 451)],
            ],
            "17": [
                [150, 250],
                [compute_f(1500, 135), compute_f(1500, 449)],
                [compute_df(1500, 135), compute_df(1500, 449)],
            ],
        }
        expected_labels = ["time_steps", "f", "df"]

        self.assertEqual(labels, expected_labels)

        self.assertEqual(Counter(threshold_data.keys()),
                         Counter(expected_threshold_data.keys()))

        for key, data in threshold_data.items():
            expected_data = expected_threshold_data[key]
            for x, expected_x in zip(data[0], expected_data[0]):
                self.assertEqual(x, expected_x)
            for f, expected_f in zip(data[1], expected_data[1]):
                self.assertAlmostEqual(f, expected_f)
            for df, expected_df in zip(data[2], expected_data[2]):
                self.assertAlmostEqual(df, expected_df)


    def test_extract_threshold_batch_files(self):
        threshold_data, labels = pt.extract_threshold_data(self.data_set_batch,
                                                           x_data="time_steps")
        expected_threshold_data = {
            "9": [
                [150, 250],
                [compute_f(3000, 197+145), compute_f(1500, 451)],
                [compute_df(3000, 197+145), compute_df(1500, 451)],
            ],
            "17": [
                [150, 250],
                [compute_f(1500, 135), compute_f(1500, 449)],
                [compute_df(1500, 135), compute_df(1500, 449)],
            ],
        }
        expected_labels = ["time_steps", "f", "df"]

        self.assertEqual(labels, expected_labels)

        self.assertEqual(Counter(threshold_data.keys()),
                         Counter(expected_threshold_data.keys()))

        for key, data in threshold_data.items():
            expected_data = expected_threshold_data[key]
            for x, expected_x in zip(data[0], expected_data[0]):
                self.assertEqual(x, expected_x)
            for f, expected_f in zip(data[1], expected_data[1]):
                self.assertAlmostEqual(f, expected_f)
            for df, expected_df in zip(data[2], expected_data[2]):
                self.assertAlmostEqual(df, expected_df)



