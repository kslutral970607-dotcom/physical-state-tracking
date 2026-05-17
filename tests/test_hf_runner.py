import unittest

from physical_state_tracking.hf_runner import HuggingFaceRunner, SYSTEM_MESSAGE


class FakeTensor:
    def __init__(self, values):
        self.values = values
        self.shape = (1, len(values))

    def to(self, _device):
        return self


class FakeSequence:
    def __init__(self, values):
        self.values = values

    def __getitem__(self, item):
        if isinstance(item, slice):
            return self.values[item]
        return self.values[item]


class FakeParameter:
    device = "cpu"


class FakeModel:
    def __init__(self):
        self.generate_kwargs = None

    def parameters(self):
        return iter([FakeParameter()])

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        return [FakeSequence([101, 102, 201, 202])]


class FakeChatTokenizer:
    eos_token_id = 0
    chat_template = "chat-template"

    def __init__(self):
        self.messages = None
        self.add_generation_prompt = None
        self.decoded_ids = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.messages = messages
        self.add_generation_prompt = add_generation_prompt
        self.tokenize = tokenize
        return "<chat>prompt</chat>"

    def __call__(self, text, return_tensors):
        self.encoded_text = text
        self.return_tensors = return_tensors
        return {"input_ids": FakeTensor([101, 102])}

    def decode(self, ids, skip_special_tokens):
        self.decoded_ids = ids
        self.skip_special_tokens = skip_special_tokens
        return '{"z": 8, "d": 1, "k": 0, "w": "blue"}'


class HuggingFaceRunnerTest(unittest.TestCase):
    def test_chat_runner_uses_template_and_decodes_only_completion(self):
        runner = HuggingFaceRunner.__new__(HuggingFaceRunner)
        runner.model_name = "Qwen/Qwen2.5-1.5B-Instruct"
        runner.max_new_tokens = 96
        runner.device = -1
        runner._tokenizer = FakeChatTokenizer()
        runner._model = FakeModel()

        completion = runner.generate("dataset prompt")

        self.assertEqual(completion, '{"z": 8, "d": 1, "k": 0, "w": "blue"}')
        self.assertEqual(
            runner._tokenizer.messages,
            [
                {"role": "system", "content": SYSTEM_MESSAGE},
                {"role": "user", "content": "dataset prompt"},
            ],
        )
        self.assertTrue(runner._tokenizer.add_generation_prompt)
        self.assertEqual(runner._tokenizer.encoded_text, "<chat>prompt</chat>")
        self.assertEqual(runner._tokenizer.decoded_ids, [201, 202])
        self.assertEqual(runner._model.generate_kwargs["max_new_tokens"], 96)
        self.assertFalse(runner._model.generate_kwargs["do_sample"])
        self.assertNotIn("temperature", runner._model.generate_kwargs)
        self.assertNotIn("top_p", runner._model.generate_kwargs)
        self.assertNotIn("top_k", runner._model.generate_kwargs)

    def test_plain_runner_does_not_append_example_answer(self):
        runner = HuggingFaceRunner.__new__(HuggingFaceRunner)
        runner.model_name = "gpt2"
        runner._tokenizer = object()

        request = runner._format_prompt("dataset prompt")

        self.assertEqual(request, "dataset prompt")
        self.assertNotIn('{"z": 4, "d": 1, "k": 0, "w": "blue"}', request)


if __name__ == "__main__":
    unittest.main()
