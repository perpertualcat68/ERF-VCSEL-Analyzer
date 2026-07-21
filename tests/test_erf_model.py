import os
import sys
import unittest

import numpy as np
from scipy.special import erf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcsel_analyzer.core.erf_model import build_erf_model, data_driven_initial_params, num_components


class TestErfModel(unittest.TestCase):
    def test_num_components(self):
        self.assertEqual(num_components(97), 32)
        self.assertEqual(num_components(1), 0)
        self.assertEqual(num_components(4), 1)

    def test_build_matches_manual_single_component(self):
        x = np.linspace(-5, 5, 101)
        params = [2.0, 1.5, 0.8, 0.5]  # baseline, amp, width, pos
        expected = 2.0 + 1.5 * erf(0.8 * (x - 0.5))
        np.testing.assert_allclose(build_erf_model(x, params), expected)

    def test_build_baseline_only(self):
        x = np.linspace(0, 1, 10)
        np.testing.assert_allclose(build_erf_model(x, [3.0]), np.full_like(x, 3.0))

    def test_data_driven_initial_params_length_and_reproducibility(self):
        x = np.linspace(0, 100, 500)
        y = build_erf_model(x, [1.0, 1.0, 0.5, 30.0, -1.0, 0.5, 70.0])
        p1 = data_driven_initial_params(x, y, total_params=97, seed=0)
        p2 = data_driven_initial_params(x, y, total_params=97, seed=0)
        self.assertEqual(len(p1), 97)
        self.assertEqual(p1, p2)  # deterministic for a fixed seed


if __name__ == '__main__':
    unittest.main()
