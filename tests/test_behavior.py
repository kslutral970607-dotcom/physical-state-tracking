import unittest

from physical_state_tracking.behavior import compute_metrics, evaluate_dataset, parse_json_object


def sample_with_context(**overrides):
    sample = {
        "prompt": "prompt",
        "shell": "symbolic",
        "steps": [{"index": 1, "step_size": 3, "text": "Apply one transition with step_size 3."}],
        "trajectory": [{"z": 8, "d": 1, "k": 0, "w": "blue"}],
        "transition_rule_metadata": {"lower_boundary": 0, "upper_boundary": 10},
        "initial_state": {"z": 5, "d": 1, "k": 0, "w": "blue"},
        "final_ground_truth_state": {"z": 8, "d": 1, "k": 0, "w": "blue"},
        "z": 5,
        "d": 1,
        "k": 0,
        "w": "blue",
    }
    sample.update(overrides)
    return sample


class BehaviorTest(unittest.TestCase):
    def test_parse_json_object_from_noisy_output(self):
        parsed = parse_json_object('thinking...\n{"A": "4", "B": 2}\nextra')

        self.assertEqual(parsed, {"A": "4", "B": 2})

    def test_parse_python_style_dict_from_noisy_output(self):
        parsed = parse_json_object("final answer: {'A': 4, 'B': 2}")

        self.assertEqual(parsed, {"A": 4, "B": 2})

    def test_evaluate_dataset_normalizes_wrapped_prediction_values(self):
        samples = [
            sample_with_context(
                initial_state={"A": 1},
                final_ground_truth_state={"A": 4},
                z=0,
                k=1,
                w=0,
            )
        ]

        rows = evaluate_dataset(samples, lambda _: '{"final_state": {"A": "4"}}')

        self.assertTrue(rows[0]["json_valid"])
        self.assertTrue(rows[0]["exact_match"])
        self.assertEqual(rows[0]["prediction"], {"A": 4})

    def test_evaluate_dataset_accepts_ground_truth_wrapper(self):
        samples = [
            sample_with_context(
                initial_state={"A": 1},
                final_ground_truth_state={"A": 4},
                z=0,
                k=1,
                w=0,
            )
        ]

        rows = evaluate_dataset(samples, lambda _: '{"final_ground_truth_state": {"A": 4}}')

        self.assertTrue(rows[0]["exact_match"])

    def test_evaluate_dataset_preserves_original_sample_context(self):
        sample = sample_with_context()

        rows = evaluate_dataset([sample], lambda _: '{"z": 8, "d": 1, "k": 0, "w": "blue"}')

        for key in (
            "prompt",
            "steps",
            "trajectory",
            "transition_rule_metadata",
            "initial_state",
            "final_ground_truth_state",
            "shell",
        ):
            self.assertEqual(rows[0][key], sample[key])

    def test_one_step_symbolic_no_bounce_smoke_case(self):
        sample = sample_with_context(
            prompt=(
                "Track the bouncing state machine for the symbolic shell.\n"
                'Initial state JSON: {"d": 1, "k": 0, "w": "blue", "z": 5}\n'
                'Steps JSON:\n[{"index": 1, "step_size": 3, "text": "Apply one transition with step_size 3."}]'
            )
        )

        rows = evaluate_dataset([sample], lambda _: '{"z": 8, "d": 1, "k": 0, "w": "blue"}')

        self.assertTrue(rows[0]["exact_match"])
        self.assertEqual(rows[0]["prediction"], {"z": 8, "d": 1, "k": 0, "w": "blue"})

    def test_compute_metrics(self):
        rows = [
            {
                "exact_match": True,
                "json_valid": True,
                "prediction": {"z": 4, "d": 1, "k": 0, "w": "blue"},
                "final_ground_truth_state": {"z": 4, "d": 1, "k": 0, "w": "blue"},
            },
            {
                "exact_match": False,
                "json_valid": True,
                "prediction": {"z": 3, "d": 1, "k": 0, "w": "blue"},
                "final_ground_truth_state": {"z": 4, "d": -1, "k": 1, "w": "blue"},
            },
            {
                "exact_match": False,
                "json_valid": False,
                "prediction": None,
                "final_ground_truth_state": {"z": 4, "d": -1, "k": 1, "w": "red"},
            },
        ]

        metrics = compute_metrics(rows)

        self.assertAlmostEqual(metrics["exact_match"], 1 / 3)
        self.assertAlmostEqual(metrics["json_validity"], 2 / 3)
        self.assertAlmostEqual(metrics["z_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["d_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["k_accuracy"], 1 / 3)
        self.assertAlmostEqual(metrics["w_preservation"], 2 / 3)


if __name__ == "__main__":
    unittest.main()
