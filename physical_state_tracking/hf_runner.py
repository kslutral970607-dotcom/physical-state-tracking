"""HuggingFace text-generation runner."""

from __future__ import annotations

from dataclasses import dataclass


SYSTEM_MESSAGE = (
    "You are a precise deterministic state-machine simulator. "
    "Compute exactly. Return only JSON."
)


@dataclass
class HuggingFaceRunner:
    model_name: str = "gpt2"
    max_new_tokens: int = 96
    device: int = -1

    def __post_init__(self) -> None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace dependencies are missing. Install them with "
                "`pip install -r requirements.txt`, then rerun this script."
            ) from exc

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        self._model = AutoModelForCausalLM.from_pretrained(self.model_name, trust_remote_code=True)
        if self.device >= 0:
            self._model.to(f"cuda:{self.device}")
        self._model.eval()

    def generate(self, prompt: str) -> str:
        request = self._format_prompt(prompt)
        inputs = self._tokenizer(request, return_tensors="pt")
        model_device = next(self._model.parameters()).device
        inputs = {key: value.to(model_device) for key, value in inputs.items()}
        input_token_count = inputs["input_ids"].shape[-1]
        output_ids = self._model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            do_sample=False,
            pad_token_id=self._tokenizer.eos_token_id,
        )
        completion_ids = output_ids[0][input_token_count:]
        return self._tokenizer.decode(completion_ids, skip_special_tokens=True).strip()

    def _format_prompt(self, prompt: str) -> str:
        if self._uses_chat_template():
            return self._tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        return prompt

    def _uses_chat_template(self) -> bool:
        apply_chat_template = getattr(self._tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            return False
        if getattr(self._tokenizer, "chat_template", None):
            return True
        model_name = self.model_name.lower()
        return any(marker in model_name for marker in ("instruct", "chat", "qwen"))
