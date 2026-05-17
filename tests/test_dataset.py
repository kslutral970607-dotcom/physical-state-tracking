import unittest

from physical_state_tracking.dataset import generate_dataset
from physical_state_tracking.shells import SHELLS


REQUIRED_KEYS = {
    "prompt",
    "shell",
    "steps",
    "initial_state",
    "final_ground_truth_state",
    "z",
    "d",
    "k",
    "w",
}


class DatasetTest(unittest.TestCase):
    def test_generates_required_schema_for_all_shells(self):
        samples = generate_dataset(n=len(SHELLS), seed=7, d=3, k=2, w=1)

        self.assertEqual([sample["shell"] for sample in samples], list(SHELLS))
        for sample in samples:
            self.assertEqual(set(sample), REQUIRED_KEYS)
            self.assertEqual(len(sample["steps"]), sample["d"])
            self.assertEqual(len(sample["initial_state"]), sample["k"])
            self.assertEqual(len(sample["final_ground_truth_state"]), sample["k"])
            self.assertIn("Return only the final state.", sample["prompt"])

    def test_generation_is_deterministic(self):
        first = generate_dataset(n=5, seed=13)
        second = generate_dataset(n=5, seed=13)

        self.assertEqual(first, second)

    def test_rejects_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            generate_dataset(n=1, d=0)
        with self.assertRaises(ValueError):
            generate_dataset(n=1, k=5)
        with self.assertRaises(ValueError):
            generate_dataset(n=1, w=-1)


if __name__ == "__main__":
    unittest.main()
