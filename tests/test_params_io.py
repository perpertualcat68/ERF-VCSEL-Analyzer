import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vcsel_analyzer.io.params_io import (
    format_parameters_file,
    parse_parameters_text,
    read_parameters_file,
    write_parameters_file,
)


class TestParamsIO(unittest.TestCase):
    def test_round_trip_text(self):
        params = [1.0, -0.5, 0.4, 25.0, 0.5, 0.4, 60.0]
        text = format_parameters_file(params, final_loss=1.23e-9, timestamp="2026-07-04 00:00:00")
        parsed, errors = parse_parameters_text(text)
        self.assertEqual(errors, [])
        self.assertEqual(len(parsed), len(params))
        for a, b in zip(parsed, params):
            self.assertLess(abs(a - b), 1e-9)

    def test_parse_ignores_comments_and_blanks(self):
        text = "# header\n\nk1: 1.0\n2.0\n# comment\nk3: 3.0\n"
        parsed, errors = parse_parameters_text(text)
        self.assertEqual(parsed, [1.0, 2.0, 3.0])
        self.assertEqual(errors, [])

    def test_parse_reports_non_finite(self):
        text = "k1: 1.0\nk2: inf\nk3: 3.0\n"
        parsed, errors = parse_parameters_text(text)
        self.assertEqual(parsed, [1.0, 3.0])
        self.assertEqual(len(errors), 1)

    def test_file_round_trip(self):
        params = [0.0, 1.0, 0.5, 10.0]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "params.txt")
            write_parameters_file(path, params, timestamp="2026-07-04 00:00:00")
            parsed, errors = read_parameters_file(path)
        self.assertEqual(errors, [])
        self.assertEqual(len(parsed), len(params))


if __name__ == '__main__':
    unittest.main()
