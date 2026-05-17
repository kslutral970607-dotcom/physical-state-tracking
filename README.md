# physical-state-tracking

Initial codebase for:

**When Physical Reasoning Becomes State Tracking: Causal Abstraction Tests in Language Models**

Phase 1 generates synthetic bouncing state-machine datasets across six prompt shells:

- `symbolic`
- `explicit_physics`
- `implicit_physics`
- `finance`
- `game`
- `adversarial`

Each JSONL sample contains:

- `prompt`
- `shell`
- `steps`
- `initial_state`: `{"z": ..., "d": ..., "k": ..., "w": ...}`
- `final_ground_truth_state`: `{"z": ..., "d": ..., "k": ..., "w": ...}`
- `trajectory`
- `transition_rule_metadata`
- `z`, `d`, `k`, `w`

The abstract state is:

- `z`: discrete position from 0 to 10
- `d`: direction, either `1` or `-1`
- `k`: boundary-crossing bounce count
- `w`: dummy variable, such as a color, that must remain unchanged

Each transition proposes `z_next = z + step_size * d`. If the proposal reaches or crosses boundary `0` or `10`, it reflects back into range, reverses `d`, and increments `k`. Otherwise, only `z` updates.

## Repository Layout

```text
physical_state_tracking/
  behavior.py
  dataset.py
  hf_runner.py
  shells.py
  state_machine.py
scripts/
  01_generate_data.py
  02_run_behavior.py
tests/
  test_behavior.py
  test_dataset.py
  test_state_machine.py
```

## Local Commands

Run from the repository root.

Install Phase 2 dependencies when you are ready to run HuggingFace models:

```powershell
python -m pip install -r requirements.txt
```

Generate Phase 1 data:

```powershell
python scripts/01_generate_data.py --n 60 --seed 0 --num-steps 6 --output data/phase1/synthetic_state_tracking.jsonl
```

Create the default Phase 2 dev file:

```powershell
New-Item -ItemType Directory -Force data/processed
Copy-Item data/phase1/synthetic_state_tracking.jsonl data/processed/dev.jsonl
```

Run GPT-2 small behavior evaluation:

```powershell
python scripts/02_run_behavior.py --model gpt2 --input data/processed/dev.jsonl --predictions outputs/behavior_predictions.jsonl --metrics outputs/behavior_metrics.csv
```

Run tests:

```powershell
python -m unittest discover -s tests
```

If `python` is not on PATH in Codex Desktop, use the bundled Python:

```powershell
C:\Users\Kings\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m pip install -r requirements.txt
C:\Users\Kings\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/01_generate_data.py --n 60 --seed 0 --num-steps 6 --output data/phase1/synthetic_state_tracking.jsonl
New-Item -ItemType Directory -Force data/processed
Copy-Item data/phase1/synthetic_state_tracking.jsonl data/processed/dev.jsonl
C:\Users\Kings\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe scripts/02_run_behavior.py --model gpt2 --input data/processed/dev.jsonl --predictions outputs/behavior_predictions.jsonl --metrics outputs/behavior_metrics.csv
C:\Users\Kings\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe -m unittest discover -s tests
```

Example with all control variables set:

```powershell
python scripts/01_generate_data.py --n 120 --seed 42 --z 9 --d 1 --k 0 --w blue --num-steps 8 --output data/phase1/z9_d1_k0_wblue.jsonl
```

## AutoDL Commands

Run from the repository root on AutoDL.

```bash
python -m pip install -r requirements.txt
python scripts/01_generate_data.py --n 60 --seed 0 --num-steps 6 --output data/phase1/synthetic_state_tracking.jsonl
mkdir -p data/processed
cp data/phase1/synthetic_state_tracking.jsonl data/processed/dev.jsonl
python scripts/02_run_behavior.py --model gpt2 --input data/processed/dev.jsonl --predictions outputs/behavior_predictions.jsonl --metrics outputs/behavior_metrics.csv
python -m unittest discover -s tests
```

Larger example:

```bash
python scripts/01_generate_data.py --n 1000 --seed 42 --z 9 --d 1 --k 0 --w blue --num-steps 8 --output data/phase1/autodl_phase1.jsonl
```

No dependency installation is required for Phase 1. Phase 2 requires the packages in `requirements.txt`, but this repository does not install them automatically.

## Phase 2 Metrics

`outputs/behavior_predictions.jsonl` stores one row per sample with the raw model output, parsed JSON, normalized prediction, JSON validity, and exact-match result.

`outputs/behavior_metrics.csv` stores:

- `exact_match`: fraction of predictions matching `final_ground_truth_state`
- `z_accuracy`: fraction with correct final `z`
- `d_accuracy`: fraction with correct final `d`
- `k_accuracy`: fraction with correct final `k`
- `w_preservation`: fraction with correct unchanged `w`
- `json_validity`: fraction of outputs with a parseable JSON object
