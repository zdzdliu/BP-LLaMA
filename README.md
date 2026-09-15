# BP-LLaMA: Cuffless Blood Pressure Estimation with Large Language Models

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

This repository provides the official implementation of **BP-LLaMA**, a method for
**cuffless blood pressure (BP) estimation** that recasts BP prediction as a
**text-generation task** on a 4-bit quantized **Llama-3-8B** backbone, fine-tuned with
**QLoRA**.

Instead of directly regressing systolic/diastolic BP, the model generates **mean
arterial pressure (MAP)** and **pulse pressure (PP)** in natural language:

```
Predicted_map: 88.0 mmHg
Predicted_pp: 39.0 mmHg
```

SBP and DBP are then recovered via the standard physiological relations
`MAP = DBP + PP/3` and `PP = SBP − DBP`:

```
SBP = MAP + 2/3 * PP
DBP = MAP − 1/3 * PP
```

---

## Overview

| Component | Description |
|---|---|
| Backbone | Llama-3-8B (4-bit, via Unsloth) |
| Fine-tuning | QLoRA (`r=16`, `alpha=16`, all 7 linear projections) |
| Input | Instruction prompt containing a user profile and 32 physiological features (9 cardiac output + 20 systemic vascular resistance + 3 arterial stiffness) |
| Output | Generated text `Predicted_map / Predicted_pp` |
| Loss | Completion-only causal-LM loss (or full-sequence loss) |
| Calibration | Post-hoc `0.3 × prediction + 0.7 × calibration BP` |

The key idea is that **baseline/calibration BP is not fed into the model**. The model
predicts absolute MAP/PP from physiological features and user context; an external
linear blend with each subject's calibration BP then yields the final SBP/DBP. This
avoids shortcut learning where the model simply copies the calibration BP.

---

## Repository Layout

```
BP-LLaMA/
├── train.py            # Training + inference + evaluation script
├── requirements.txt
├── dataset/
│   ├── train_dataset.json   # Example training split
│   ├── valid_dataset.json   # Example validation split
│   └── test_dataset.json    # Example test split
└── README.md
```

---

## Installation

We recommend a dedicated conda/virtual environment. The code has been tested with
`PyTorch 2.x`, `transformers 4.56.2`, `trl 0.23`, and `unsloth 2025.10.8`.

```bash
conda create -n bpllama python=3.10 -y
conda activate bpllama

# Install PyTorch matching your CUDA version, e.g. CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu121

# Install remaining dependencies
pip install -r requirements.txt
```

> **Note on Unsloth**: `unsloth` constrains the `transformers` version
> (`>=4.51.3, <=4.56.2`). Installing `unsloth==2025.10.8` from PyPI resolves a
> compatible `transformers` automatically.

---

## Data Preparation

The model consumes **instruction-style JSON** files. Each file is a JSON list of
records:

```json
[
  {
    "text": "### Instruction: ... ### Input: ... ### Response:\nPredicted_map: 62.5 mmHg, Predicted_pp: 24.0 mmHg.",
    "refsbp": 110.0,
    "refdbp": 74.0,
    "basesbp": 116.0,
    "basedbp": 71.0
  }
]
```

| Field | Type | Description |
|---|---|---|
| `text` | string | Instruction prompt + `### Response:` completion (contains `Predicted_map / Predicted_pp`). |
| `refsbp` | number | Reference (ground-truth) systolic BP (mmHg). |
| `refdbp` | number | Reference diastolic BP (mmHg). |
| `basesbp` | number | Calibration (baseline) systolic BP (mmHg). |
| `basedbp` | number | Calibration diastolic BP (mmHg). |

The `text` field is the only field used during training; `refsbp`/`refdbp` and
`basesbp`/`basedbp` are used for evaluation and post-hoc calibration.

Three files are expected under `dataset/` (example placeholders are provided):

- `train_dataset.json`
- `valid_dataset.json`
- `test_dataset.json`

The `text` field follows an Alpaca-style three-part format:

```
### Instruction:
<role description and output format>

### Input:
<BP domain knowledge + user profile + three physiological feature lists>

### Response:
Predicted_map: X.X mmHg, Predicted_pp: Y.Y mmHg.
```

The three physiological feature groups are:

- **Cardiac output features** (9 values)
- **Systemic vascular resistance features** (20 values)
- **Arterial stiffness features** (3 values)

The medical source data (ECG/PPG recordings) is **not** included in this repository;
replace the example JSONs with your own preprocessed data.

---

## Training

With the three dataset files in place, simply run:

```bash
python train.py \
    --model_dir /path/to/llama-3-8b-bnb-4bit \
    --prompt_type with_knowledge_and_user_info \
    --loss_mode completion
```

By default it reads `dataset/train_dataset.json`, `dataset/valid_dataset.json`, and
`dataset/test_dataset.json`. To use different paths:

```bash
python train.py \
    --model_dir /path/to/llama-3-8b-bnb-4bit \
    --train dataset/train_dataset.json \
    --valid dataset/valid_dataset.json \
    --test dataset/test_dataset.json
```

### Key options

| Option | Default | Description |
|---|---|---|
| `--train` / `--valid` / `--test` | `dataset/*.json` | Paths to the three splits. |
| `--prompt_type` | `with_knowledge_and_user_info` | `basic` / `with_knowledge` / `with_knowledge_and_user_info` |
| `--loss_mode` | `completion` | `completion` (answer-only) or `full` (whole-sequence) |
| `--model_dir` | HF repo | Path to a local Llama-3-8B (bnb 4-bit) checkpoint. |
| `--lr` | `2e-4` | Learning rate |
| `--epochs` | `3` | Number of epochs |
| `--lora_r` / `--lora_alpha` | `16` / `16` | LoRA rank / alpha |
| `--batch_size` / `--grad_accum` | `4` / `4` | Per-device batch size / gradient accumulation |
| `--limit_test N` | `0` | Run only the first N test samples (smoke test) |
| `--skip_train` | `False` | Load an existing adapter and only run inference |

---

## Inference

To re-run inference with a previously trained adapter:

```bash
python train.py \
    --model_dir /path/to/llama-3-8b-bnb-4bit \
    --test dataset/test_dataset.json \
    --output_dir outputs/test_dataset_with_knowledge_and_user_info \
    --skip_train
```

---

## Outputs

For each run, the following files are written to `outputs/<teststem>_<prompt_type>/`:

- `<teststem>_<prompt_type>.csv` — per-sample SBP/DBP predictions and references.
- `test_predictions.json` — generated text and parsed MAP/PP for every sample.
- `test_metrics.json` — raw / calibrated / baseline-only metrics (MAE, RMSE, ME, SD,
  Pearson R, BHS grade, AAMI pass/fail).
- `train_result.json` — training metrics and full log history.
- LoRA adapter and tokenizer.

---

## Evaluation

Metrics are computed for **SBP, DBP, MAP, and PP** under three conditions:

- **raw** — the model's direct prediction;
- **calibrated** — `0.3 × prediction + 0.7 × calibration BP`;
- **baseline-only** — the calibration BP itself (the reference for "does the model
  beat the baseline?").

Standard cuffless-BP criteria are also reported: **BHS grades** (percentage of errors
within 5/10/15 mmHg) and the **AAMI/ANSI criterion** (`|ME| ≤ 5` and `SD ≤ 8`).

---

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{...,
  title  = {Cuffless Blood Pressure Estimation via Large Language Models},
  author = {...},
  journal= {...},
  year   = {...}
}
```

---

## License

This project is released under the [MIT License](LICENSE).
