import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcsel_analyzer.core import thickness as T
from vcsel_analyzer.core.erf_model import build_erf_model


class TestThickness(unittest.TestCase):
    def test_edges_from_params_sorted(self):
        # two components with positions 70 then 30 -> should sort to [30, 70]
        params = [0.0, 1.0, 0.5, 70.0, -1.0, 0.5, 30.0]
        positions, indices, amps, widths = T.edges_from_params(params)
        self.assertEqual(positions, [30.0, 70.0])
        self.assertEqual(indices, [6, 3])
        self.assertEqual(amps, [-1.0, 1.0])
        self.assertEqual(widths, [0.5, 0.5])

    def test_layer_thicknesses(self):
        self.assertEqual(T.layer_thicknesses([10.0, 25.0, 45.0]), [15.0, 20.0])

    def test_assign_materials_alternates(self):
        thick = [10.0, 20.0, 30.0, 40.0]
        errs = [1.0, 2.0, 3.0, 4.0]
        lt, le = T.assign_materials(thick, errs)
        self.assertEqual(lt['Material_A'], [10.0, 30.0])
        self.assertEqual(lt['Material_B'], [20.0, 40.0])
        self.assertEqual(le['Material_A'], [1.0, 3.0])
        self.assertEqual(le['Material_B'], [2.0, 4.0])

    def test_thickness_errors_residual_fallback_runs(self):
        params = [0.0, 1.0, 0.5, 30.0, -1.0, 0.5, 70.0]
        x = np.linspace(0, 100, 400)
        y = build_erf_model(x, params)
        positions, indices, amps, widths = T.edges_from_params(params)
        errs, source = T.thickness_errors(
            positions, amps, widths, indices,
            covariance=None, fit_x=x, fit_y=y,
            model_fn=lambda xx: build_erf_model(xx, params),
        )
        self.assertEqual(source, "residual-based")
        self.assertEqual(len(errs), 1)
        self.assertGreaterEqual(errs[0], 0.0)


if __name__ == '__main__':
    unittest.main()
