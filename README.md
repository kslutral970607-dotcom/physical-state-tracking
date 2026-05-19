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
  06_prepare_sft_data.py
  07_train_lora_sft.py
  08_eval_lora_checkpoint.py
tests/
  test_behavior.py
  test_dataset.py
  test_sft_data.py
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

Generate a simple plain diagnostic set:

```powershell
python scripts/01_generate_data.py --n 4 --seed 5 --z 5 --d 1 --k 0 --w blue --num-steps 1 --shells symbolic --prompt-variant simple_plain --output data/phase1/simple_plain_symbolic_no_boundary.jsonl
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

## LoRA SFT Pipeline

The SFT path is separate from the zero-shot behavior runner. `scripts/02_run_behavior.py` remains unchanged for baseline evaluation.

Convert generated samples to chat-format SFT JSONL:

```powershell
python scripts/06_prepare_sft_data.py --input data/phase1/stage1_symbolic_no_boundary.jsonl --output data/sft/stage1_final_only.jsonl --format final_only
python scripts/06_prepare_sft_data.py --input data/phase1/stage1_symbolic_no_boundary.jsonl --output data/sft/stage1_trajectory_then_final.jsonl --format trajectory_then_final
```

Train a Qwen2.5-1.5B-Instruct LoRA adapter:

```powershell
python scripts/07_train_lora_sft.py --model Qwen/Qwen2.5-1.5B-Instruct --train-data data/sft/stage1_final_only.jsonl --eval-data data/sft/stage1_final_only.jsonl --output-dir outputs/lora_qwen_stage1 --max-steps 200 --learning-rate 2e-4 --lora-r 16 --lora-alpha 32 --batch-size 1 --gradient-accumulation-steps 8
```

Evaluate a trained LoRA checkpoint with the behavior metrics:

```powershell
python scripts/08_eval_lora_checkpoint.py --model Qwen/Qwen2.5-1.5B-Instruct --checkpoint outputs/lora_qwen_stage1 --input data/phase1/stage1_symbolic_no_boundary.jsonl --predictions outputs/lora_stage1_predictions.jsonl --metrics outputs/lora_stage1_metrics.csv --device 0 --max-new-tokens 128
```

## Curriculum Data Commands

Stage 1: 1-step no-boundary symbolic.

```powershell
python scripts/01_generate_data.py --n 1000 --seed 101 --z 5 --d 1 --k 0 --w blue --num-steps 1 --shells symbolic --prompt-variant metadata_json --output data/phase1/stage1_symbolic_no_boundary.jsonl
python scripts/06_prepare_sft_data.py --input data/phase1/stage1_symbolic_no_boundary.jsonl --output data/sft/stage1_symbolic_no_boundary_final_only.jsonl --format final_only
```

Stage 2: 1-step boundary symbolic.

```powershell
python scripts/01_generate_data.py --n 1000 --seed 102 --z 9 --d 1 --k 0 --w blue --num-steps 1 --shells symbolic --prompt-variant metadata_json --output data/phase1/stage2_symbolic_boundary.jsonl
python scripts/06_prepare_sft_data.py --input data/phase1/stage2_symbolic_boundary.jsonl --output data/sft/stage2_symbolic_boundary_final_only.jsonl --format final_only
```

Stage 3: 2-step symbolic mixed.

```powershell
python scripts/01_generate_data.py --n 2000 --seed 103 --num-steps 2 --shells symbolic --prompt-variant metadata_json --output data/phase1/stage3_symbolic_2step_mixed.jsonl
python scripts/06_prepare_sft_data.py --input data/phase1/stage3_symbolic_2step_mixed.jsonl --output data/sft/stage3_symbolic_2step_mixed_final_only.jsonl --format final_only
```

Stage 4: 4-step / 6-step symbolic mixed.

```powershell
python scripts/01_generate_data.py --n 3000 --seed 104 --num-steps 4 --shells symbolic --prompt-variant metadata_json --output data/phase1/stage4_symbolic_4step_mixed.jsonl
python scripts/01_generate_data.py --n 3000 --seed 105 --num-steps 6 --shells symbolic --prompt-variant metadata_json --output data/phase1/stage4_symbolic_6step_mixed.jsonl
python scripts/06_prepare_sft_data.py --input data/phase1/stage4_symbolic_4step_mixed.jsonl --output data/sft/stage4_symbolic_4step_mixed_final_only.jsonl --format final_only
python scripts/06_prepare_sft_data.py --input data/phase1/stage4_symbolic_6step_mixed.jsonl --output data/sft/stage4_symbolic_6step_mixed_final_only.jsonl --format final_only
```

Stage 5: semantic shells.

```powershell
python scripts/01_generate_data.py --n 6000 --seed 106 --num-steps 6 --shells explicit_physics implicit_physics finance game adversarial --prompt-variant metadata_json --output data/phase1/stage5_semantic_shells.jsonl
python scripts/06_prepare_sft_data.py --input data/phase1/stage5_semantic_shells.jsonl --output data/sft/stage5_semantic_shells_final_only.jsonl --format final_only
```

Example with all control variables set:

```powershell
python scripts/01_generate_data.py --n 120 --seed 42 --z 9 --d 1 --k 0 --w blue --num-steps 8 --output data/phase1/z9_d1_k0_wblue.jsonl
```

## MatPool Commands

Run from the repository root on MatPool.

```bash
python -m pip install -r requirements.txt
python scripts/01_generate_data.py --n 60 --seed 0 --num-steps 6 --output data/phase1/synthetic_state_tracking.jsonl
python scripts/01_generate_data.py --n 4 --seed 5 --z 5 --d 1 --k 0 --w blue --num-steps 1 --shells symbolic --prompt-variant simple_plain --output data/phase1/simple_plain_symbolic_no_boundary.jsonl
mkdir -p data/processed
cp data/phase1/simple_plain_symbolic_no_boundary.jsonl data/processed/dev.jsonl
python scripts/02_run_behavior.py --model gpt2 --input data/processed/dev.jsonl --predictions outputs/behavior_predictions.jsonl --metrics outputs/behavior_metrics.csv
python -m unittest discover -s tests
```

Qwen behavior example:

```bash
python scripts/02_run_behavior.py --model Qwen/Qwen2.5-7B-Instruct --input data/processed/dev.jsonl --predictions outputs/qwen_behavior_predictions.jsonl --metrics outputs/qwen_behavior_metrics.csv --device 0 --max-new-tokens 96
```

Larger MatPool data example:

```bash
python scripts/01_generate_data.py --n 1000 --seed 42 --z 9 --d 1 --k 0 --w blue --num-steps 8 --output data/phase1/matpool_phase1.jsonl
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
