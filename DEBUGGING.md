# Logic Audit Report

## What was checked

- `physical_state_tracking/state_machine.py`
  - Verified `z_next = z + step_size * d` for no-boundary movement.
  - Verified upper reflection: proposed `12` from `z=9,d=1,step_size=3` becomes `z=8,d=-1,k=1`.
  - Verified lower reflection: proposed `-2` from `z=1,d=-1,step_size=3` becomes `z=2,d=1,k=1`.
  - Verified `w` is copied unchanged.

- `physical_state_tracking/dataset.py` and `physical_state_tracking/shells.py`
  - Verified `--z`, `--d`, `--k`, and `--w` force the initial state through `generate_dataset`.
  - Verified `--num-steps` controls the number of steps.
  - Verified `--shells symbolic` restricts samples to the symbolic shell.
  - Verified `final_ground_truth_state` and `trajectory` are produced from `StateMachine.run`.
  - Confirmed top-level `z`, `d`, `k`, and `w` are initial-state aliases, not final values. They are potentially ambiguous for downstream consumers.

- `physical_state_tracking/behavior.py`
  - Verified parsed predictions are compared against `final_ground_truth_state`.
  - Verified per-variable metrics use `final_ground_truth_state`.
  - Verified normalized integer fields compare as integers and `w` compares as a string.
  - Verified `exact_match` rejects extra keys because full dict equality is used.

- `physical_state_tracking/hf_runner.py`
  - Verified Qwen/Instruct models use `tokenizer.apply_chat_template`.
  - Verified `add_generation_prompt=True`.
  - Verified the system message is included.
  - Verified only generated token IDs after the prompt length are decoded.
  - Verified deterministic generation uses `do_sample=False` and does not pass `temperature`, `top_p`, or `top_k`.

- `scripts/01_generate_data.py` and `scripts/02_run_behavior.py`
  - Verified script arguments are passed through to the dataset generator and runner.

## Bugs found

- Parser isolation bug:
  - `parse_json_object` previously returned the first parseable JSON-like object in a string.
  - If a caller accidentally supplied full text containing the original prompt plus assistant completion, the parser could return the prompt's initial-state JSON instead of the final answer.
  - Fixed by returning the final parseable JSON-like object. This also handles Markdown-fenced completions.

- Prompt clarity issue:
  - The metadata prompt described bouncing and preserving `w`, but did not explicitly say that no-boundary steps preserve `d` and `k`.
  - Added: `If a step does not reach or cross 0 or 10, update only z; d and k stay unchanged.`

## Bugs not found

- No bug found in the diagnostic no-boundary transition:
  - Initial `{"z": 5, "d": 1, "k": 0, "w": "blue"}`, `step_size=3` correctly produces `{"z": 8, "d": 1, "k": 0, "w": "blue"}`.

- No bug found in forced dataset arguments:
  - With `n=10`, `num_steps=1`, pinned initial state, and `shells=["symbolic"]`, every generated sample has the pinned initial state, exactly one step, symbolic shell, and final state `z = 5 + step_size`, `d=1`, `k=0`, `w=blue`.

- No bug found in HuggingFace manual generation for Qwen-style instruct models:
  - The runner already uses the chat template, includes the system message, uses `add_generation_prompt=True`, and decodes only completion tokens.

## Likely remaining cause of the diagnostic failure

Given this audit, the specific wrong output `{"z": 6, "d": -1, "k": 2, "w": "blue"}` is unlikely to be caused by the state transition or dataset generator.

The most likely remaining causes are:

1. Model ability or prompt-following failure on the default metadata-heavy prompt.
2. Prompt design: even though there is no gold-answer leakage, the phrase "bouncing state machine" and boundary metadata may over-activate bounce behavior in weak models.
3. Runner integration outside this code path, if any caller uses a pipeline or parses full prompt-plus-completion text instead of only assistant completion text.

The parser fix protects the evaluation path against the third case if full text accidentally reaches the parser.

## Verification

- `pytest -q` could not be run because `pytest` is not installed in the available Python runtime.
- Verified with:

```powershell
C:\Users\Kings\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests -v
```

Result: 26 tests passed.
