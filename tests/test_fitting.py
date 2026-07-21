import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcsel_analyzer.config import ERF_CONFIG
from vcsel_analyzer.core.erf_model import build_erf_model
from vcsel_analyzer.core.fitting import fit_erf


class TestFitting(unittest.TestCase):
    def test_fit_recovers_single_component(self):
        true_params = [2.0, 1.5, 0.4, 25.0]
        x = np.linspace(0, 50, 400)
        y = build_erf_model(x, true_params)

        p0 = [1.0, 1.0, 0.3, 20.0]
        result = fit_erf(x, y, p0, ERF_CONFIG)

        np.testing.assert_allclose(result.params, true_params, rtol=1e-3, atol=1e-3)
        # Noiseless data: fit should be essentially exact. Use a tolerance that
        # is robust across SciPy/BLAS builds rather than an over-tight bound.
        self.assertLess(result.mse, 1e-6)
        self.assertGreaterEqual(result.rmse, 0.0)
        self.assertGreaterEqual(result.elapsed_s, 0.0)

    def test_bounded_keeps_widths_positive(self):
        true_params = [0.0, 1.0, 0.5, 25.0]
        x = np.linspace(0, 50, 400)
        y = build_erf_model(x, true_params)

        # start with a negative width guess; bounded fit must keep width >= 0
        p0 = [0.0, 1.0, -0.5, 25.0]
        result = fit_erf(x, y, p0, ERF_CONFIG, bounded=True)
        self.assertGreater(result.params[2], 0)
        self.assertLess(result.mse, 1e-9)


if __name__ == '__main__':
    unittest.main()
