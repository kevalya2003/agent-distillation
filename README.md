# Agent Trajectory Distillation

An end-to-end ML project for teaching a small open model to imitate successful agent
tool-use trajectories produced by a stronger teacher.

The pipeline:

1. ingests append-only JSONL trajectories from `../coding-agent`;
2. rejects failed, incomplete, duplicate, or inefficient runs;
3. converts successful actions into conversational tool-use examples;
4. creates a deterministic task-grouped train/evaluation split;
5. fine-tunes a small model with 4-bit QLoRA; and
6. measures held-out tool-call validity, tool selection, and argument accuracy.

This keeps the SWE and ML projects independent: the coding agent is useful by itself,
while this project focuses on data quality, fine-tuning, ablations, and evaluation.

## Why this is an ML project

Prompting a hosted model is not model training. This repository demonstrates:

- rejection-sampled data curation;
- leakage-resistant dataset splitting;
- QLoRA fine-tuning with PEFT;
- experiment manifests containing versions, configuration, and data hashes;
- held-out metrics; and
- ablations over filtering, data volume, and LoRA configuration.

No result numbers are pre-filled. Publish only measurements from completed runs.

## Data preparation (runs locally)

Python 3.8+ is enough for data preparation and scoring:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"

agent-distill prepare `
  --input ..\coding-agent\runs `
  --output data\processed `
  --eval-ratio 0.2 `
  --seed 42
```

Outputs:

```text
data/processed/
  train.jsonl
  eval.jsonl
  dataset_report.json
```

`dataset_report.json` records accepted and rejected counts, rejection reasons, tool
frequencies, and split sizes. Attempts for the same normalized issue are assigned to one
task group, so they cannot leak across train and evaluation splits.

By default, “successful” means the coding agent submitted after its configured tests
passed. For benchmark-quality data, export the task IDs that passed an external or hidden
test harness and add `--verified-ids passing_ids.json`. This prevents locally plausible
but actually incorrect patches from becoming teacher data.

## QLoRA training (Google Colab or an NVIDIA GPU)

The data pipeline and tests run on CPU. Training uses the optional GPU dependencies:

```bash
pip install -e ".[train]"
agent-distill train --config configs/qwen25_coder_3b_qlora.yaml
```

The default is `Qwen/Qwen2.5-Coder-3B-Instruct`, small enough for a free Colab GPU with
4-bit quantization. Follow `notebooks/COLAB.md` for the hosted workflow.

The output directory contains only the LoRA adapter, tokenizer files, trainer state, and
`training_manifest.json`; base-model weights are not copied into the repository.

## Held-out evaluation

Generate the first tool decision for each held-out conversation:

```bash
agent-distill generate \
  --config configs/qwen25_coder_3b_qlora.yaml \
  --adapter outputs/qwen25-coder-3b-tool-use \
  --references data/processed/eval.jsonl \
  --output outputs/predictions.jsonl

agent-distill score \
  --references data/processed/eval.jsonl \
  --predictions outputs/predictions.jsonl \
  --output outputs/metrics.json
```

Reported metrics:

- **parse rate** — generated calls containing valid structured JSON;
- **tool-name accuracy** — correct first tool selected;
- **argument-key F1** — correct shape of the tool arguments; and
- **exact-call accuracy** — exact tool name and argument object.

These offline metrics do not replace end-to-end task success. The final evaluation should
also serve the adapter behind an OpenAI-compatible endpoint, plug it into the coding
agent, and report issue resolve rate on the same fixed benchmark subset.

## Recommended experiments

Run these as controlled ablations:

1. unfiltered trajectories versus success-only rejection sampling;
2. all successful trajectories versus an efficient-step filter;
3. 25%, 50%, and 100% of the training set;
4. base model versus SFT adapter; and
5. LoRA rank 8 versus 16.

For every run, retain the manifest and evaluate against the unchanged held-out split.

## Suggested resume bullet

After replacing placeholders with real measurements:

> Built a rejection-sampled trajectory distillation pipeline for agentic tool use and
> fine-tuned a 3B open model with QLoRA. Improved held-out tool-selection accuracy from
> **X%** to **Y%**, recovered **Z%** of teacher task success, and reduced cost per task by
> **N%**; ablations isolated the effects of success filtering and training-set size.

