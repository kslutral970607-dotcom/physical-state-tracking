import json
import unittest

from physical_state_tracking.dataset import generate_dataset
from physical_state_tracking.sft_data import SYSTEM_MESSAGE, format_sft_example


class SftDataTest(unittest.TestCase):
    def test_final_only_assistant_answer_is_valid_json(self):
        sample = generate_dataset(
            n=1,
            shells=["symbolic"],
            seed=5,
            z=5,
            d=1,
            k=0,
            w="blue",
            num_steps=1,
            prompt_variant="metadata_json",
        )[0]

        example = format_sft_example(sample, output_format="final_only")
        assistant = example["messages"][2]["content"]

        self.assertEqual(example["messages"][0], {"role": "system", "content": SYSTEM_MESSAGE})
        self.assertEqual(json.loads(assistant), sample["final_ground_truth_state"])
        self.assertEqual(assistant, '{"z": 8, "d": 1, "k": 0, "w": "blue"}')

    def test_no_answer_leakage_added_to_user_prompt(self):
        sample = generate_dataset(
            n=1,
            shells=["symbolic"],
            seed=5,
            z=5,
            d=1,
            k=0,
            w="blue",
            num_steps=1,
            prompt_variant="metadata_json",
        )[0]

        example = format_sft_example(sample, output_format="final_only")
        user_prompt = example["messages"][1]["content"]

        self.assertEqual(user_prompt, sample["prompt"])
        self.assertNotIn(example["messages"][2]["content"], user_prompt)
        self.assertNotIn("final_ground_truth_state", user_prompt)

    def test_metadata_preserves_final_ground_truth_state(self):
        sample = generate_dataset(
            n=1,
            shells=["symbolic"],
            seed=5,
            z=5,
            d=1,
            k=0,
            w="blue",
            num_steps=1,
        )[0]

        example = format_sft_example(sample, output_format="trajectory_then_final")

        self.assertEqual(example["metadata"]["final_ground_truth_state"], sample["final_ground_truth_state"])
        self.assertIn("Trajectory:", example["messages"][2]["content"])
        self.assertIn("Final:", example["messages"][2]["content"])


if __name__ == "__main__":
    unittest.main()
