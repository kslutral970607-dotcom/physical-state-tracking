import unittest

from physical_state_tracking.dataset import generate_dataset
from physical_state_tracking.shells import SHELLS


REQUIRED_KEYS = {
    "prompt",
    "shell",
    "steps",
    "initial_state",
    "final_ground_truth_state",
    "trajectory",
    "transition_rule_metadata",
    "z",
    "d",
    "k",
    "w",
}


class DatasetTest(unittest.TestCase):
    def test_generates_required_schema_for_all_shells(self):
        samples = generate_dataset(n=len(SHELLS), seed=7, num_steps=3)

        self.assertEqual([sample["shell"] for sample in samples], list(SHELLS))
        for sample in samples:
            self.assertEqual(set(sample), REQUIRED_KEYS)
            self.assertEqual(set(sample["initial_state"]), {"z", "d", "k", "w"})
            self.assertEqual(set(sample["final_ground_truth_state"]), {"z", "d", "k", "w"})
            self.assertEqual(len(sample["steps"]), 3)
            self.assertEqual(len(sample["trajectory"]), 3)
            self.assertEqual(sample["initial_state"]["w"], sample["final_ground_truth_state"]["w"])
            self.assertNotIn('{"z": 4, "d": 1, "k": 0, "w": "blue"}', sample["prompt"])
            self.assertIn("Return only one valid JSON object.", sample["prompt"])
            self.assertIn('The object must contain exactly these keys: "z", "d", "k", "w".', sample["prompt"])
            self.assertIn('"z" must be the computed final integer position.', sample["prompt"])
            self.assertIn('"d" must be the computed final direction, either 1 or -1.', sample["prompt"])
            self.assertIn('"k" must be the computed final bounce count.', sample["prompt"])
            self.assertIn('"w" must be copied unchanged from the initial state.', sample["prompt"])
            self.assertIn("Do not include Markdown fences.", sample["prompt"])
            self.assertIn("Do not include explanation.", sample["prompt"])

    def test_generation_is_deterministic(self):
        first = generate_dataset(n=5, seed=13)
        second = generate_dataset(n=5, seed=13)

        self.assertEqual(first, second)

    def test_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            generate_dataset(n=1, d=0)
        with self.assertRaises(ValueError):
            generate_dataset(n=1, z=11)
        with self.assertRaises(ValueError):
            generate_dataset(n=1, num_steps=0)

    def test_can_pin_initial_state(self):
        sample = generate_dataset(n=1, seed=1, z=9, d=1, k=0, w="blue", num_steps=1)[0]

        self.assertEqual(sample["initial_state"], {"z": 9, "d": 1, "k": 0, "w": "blue"})
        self.assertEqual(sample["w"], "blue")


if __name__ == "__main__":
    unittest.main()
