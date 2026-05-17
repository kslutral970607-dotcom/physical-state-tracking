"""HuggingFace text-generation runner."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HuggingFaceRunner:
    model_name: str = "gpt2"
    max_new_tokens: int = 96
    device: int = -1

    def __post_init__(self) -> None:
        try:
            from transformers import pipeline
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace dependencies are missing. Install them with "
                "`pip install -r requirements.txt`, then rerun this script."
            ) from exc

        self._generator = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.model_name,
            device=self.device,
        )

    def generate(self, prompt: str) -> str:
        request = (
            f"{prompt}\n\n"
            "Answer as a JSON object containing only the final state, for example "
            '{"A": 1, "B": 2}.\n'
        )
        outputs = self._generator(
            request,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._generator.tokenizer.eos_token_id,
        )
        generated = outputs[0]["generated_text"]
        return generated[len(request) :].strip()
