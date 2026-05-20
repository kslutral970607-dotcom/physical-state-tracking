import unittest

from physical_state_tracking.error_patterns import compute_error_patterns


class ErrorPatternsTest(unittest.TestCase):
    def test_detects_d_positive_shortcut_pattern(self):
        rows = [
            {
                "prediction": {"z": 12, "d": 1, "k": 0, "w": "red"},
                "parsed_output": {"z": 12, "d": 1, "k": 0, "w": "red"},
                "initial_state": {"z": 8, "d": -1, "k": 0, "w": "red"},
                "steps": [{"step_size": 4}],
            }
        ]

        patterns = compute_error_patterns(rows)

        self.assertEqual(patterns["pred_d_is_1_rate"], 1.0)
        self.assertEqual(patterns["z_plus_step_rate"], 1.0)
        self.assertEqual(patterns["z_minus_step_rate"], 0.0)
        self.assertEqual(patterns["d_copy_rate"], 0.0)
        self.assertEqual(patterns["d_flip_rate"], 1.0)
        self.assertEqual(patterns["k_copy_rate"], 1.0)
        self.assertEqual(patterns["w_copy_rate"], 1.0)

    def test_invalid_and_extra_keys_rates(self):
        rows = [
            {
                "prediction": None,
                "parsed_output": None,
                "initial_state": {"z": 5, "d": 1, "k": 0, "w": "blue"},
                "steps": [{"step_size": 3}],
            },
            {
                "prediction": {"z": 8, "d": 1, "k": 0, "w": "blue", "extra": 1},
                "parsed_output": {"z": 8, "d": 1, "k": 0, "w": "blue", "extra": 1},
                "initial_state": {"z": 5, "d": 1, "k": 0, "w": "blue"},
                "steps": [{"step_size": 3}],
            },
        ]

        patterns = compute_error_patterns(rows)

        self.assertEqual(patterns["null_or_invalid_rate"], 0.5)
        self.assertEqual(patterns["extra_keys_rate"], 0.5)


if __name__ == "__main__":
    unittest.main()
