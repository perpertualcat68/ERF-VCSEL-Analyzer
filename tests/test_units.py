import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcsel_analyzer.core.units import convert_scale_to_nm


class TestUnits(unittest.TestCase):
    def test_nm_identity(self):
        self.assertEqual(convert_scale_to_nm(2.5, 'nm'), 2.5)

    def test_micron_to_nm(self):
        self.assertEqual(convert_scale_to_nm(1.0, 'um'), 1000.0)
        self.assertEqual(convert_scale_to_nm(1.0, 'µm'), 1000.0)

    def test_angstrom_to_nm(self):
        self.assertTrue(math.isclose(convert_scale_to_nm(10.0, 'angstrom'), 1.0))

    def test_none_units_assumed_nm(self):
        self.assertEqual(convert_scale_to_nm(0.246, None), 0.246)

    def test_reciprocal_units_rejected(self):
        self.assertIsNone(convert_scale_to_nm(1.0, '1/nm'))
        self.assertIsNone(convert_scale_to_nm(1.0, 'rad'))

    def test_unrecognized_units_assumed_nm(self):
        self.assertEqual(convert_scale_to_nm(3.0, 'furlong'), 3.0)

    def test_non_numeric_scale_returns_none(self):
        self.assertIsNone(convert_scale_to_nm('abc', 'nm'))


if __name__ == '__main__':
    unittest.main()
