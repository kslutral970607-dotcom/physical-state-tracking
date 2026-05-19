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
    "prompt_variant",
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
            self.assertEqual(sample["prompt_variant"], "metadata_json")
            self.assertEqual(sample["initial_state"]["w"], sample["final_ground_truth_state"]["w"])
            self.assertNotIn('{"z": 4, "d": 1, "k": 0, "w": "blue"}', sample["prompt"])
            self.assertIn("Return only one valid JSON object.", sample["prompt"])
            self.assertIn("If a step does not reach or cross 0 or 10, update only z; d and k stay unchanged.", sample["prompt"])
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

    def test_diagnostic_forced_arguments(self):
        samples = generate_dataset(
            n=10,
            shells=["symbolic"],
            seed=0,
            z=5,
            d=1,
            k=0,
            w="blue",
            num_steps=1,
        )

        self.assertEqual(len(samples), 10)
        for sample in samples:
            self.assertEqual(sample["initial_state"], {"z": 5, "d": 1, "k": 0, "w": "blue"})
            self.assertEqual(len(sample["steps"]), 1)
            self.assertEqual(sample["shell"], "symbolic")
            step_size = sample["steps"][0]["step_size"]
            self.assertIn(step_size, {1, 2, 3, 4})
            self.assertEqual(
                sample["final_ground_truth_state"],
                {"z": 5 + step_size, "d": 1, "k": 0, "w": "blue"},
            )
            self.assertEqual(sample["trajectory"], [sample["final_ground_truth_state"]])
            self.assertEqual(sample["transition_rule_metadata"]["step_sizes"], [step_size])
            self.assertEqual(sample["z"], 5)
            self.assertEqual(sample["d"], 1)
            self.assertEqual(sample["k"], 0)
            self.assertEqual(sample["w"], "blue")

    def test_simple_plain_no_boundary_prompt_is_minimal(self):
        sample = generate_dataset(
            n=1,
            shells=["symbolic"],
            seed=5,
            z=5,
            d=1,
            k=0,
            w="blue",
            num_steps=1,
            prompt_variant="simple_plain",
        )[0]

        prompt = sample["prompt"]

        self.assertEqual(sample["prompt_variant"], "simple_plain")
        self.assertEqual(sample["initial_state"], {"z": 5, "d": 1, "k": 0, "w": "blue"})
        self.assertEqual(sample["final_ground_truth_state"], {"z": 8, "d": 1, "k": 0, "w": "blue"})
        self.assertIn("step_size = 3", prompt)
        self.assertIn("There is no boundary crossing in this case.", prompt)
        self.assertIn("z = z + step_size * d", prompt)
        self.assertIn("d stays the same", prompt)
        self.assertIn("k stays the same", prompt)
        self.assertIn("w stays the same", prompt)
        self.assertNotIn("Transition rule metadata JSON", prompt)
        self.assertNotIn("Steps JSON", prompt)
        self.assertNotIn("bouncing state machine", prompt)
        self.assertNotIn('{"z": 4, "d": 1, "k": 0, "w": "blue"}', prompt)

    def test_simple_plain_boundary_prompt_explains_reflection(self):
        sample = generate_dataset(
            n=1,
            shells=["symbolic"],
            seed=1,
            z=9,
            d=1,
            k=0,
            w="blue",
            num_steps=1,
            prompt_variant="simple_plain",
        )[0]

        prompt = sample["prompt"]

        self.assertIn("proposed z_next = 9 + 2 * 1 = 11", prompt)
        self.assertIn("The boundary is 0 or 10.", prompt)
        self.assertIn("crossing causes reflection", prompt)
        self.assertIn("d reverses", prompt)
        self.assertIn("k increases", prompt)
        self.assertNotIn("Transition rule metadata JSON", prompt)
        self.assertNotIn("Steps JSON", prompt)
        self.assertNotIn('{"z": 4, "d": 1, "k": 0, "w": "blue"}', prompt)


if __name__ == "__main__":
    unittest.main()
