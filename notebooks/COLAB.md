# Colab runbook

Create a new Google Colab notebook, select a T4 GPU runtime, and run the following cells.

## 1. Clone and install

Replace the URL after pushing this folder to its own GitHub repository.

```python
REPO_URL = "https://github.com/YOUR_USERNAME/agent-distillation.git"
!git clone "$REPO_URL"
%cd agent-distillation
%pip install -e ".[train]"
```

Restart the runtime if Colab asks after installing GPU packages, then return to the
repository directory.

## 2. Upload teacher trajectories

Upload either individual `.jsonl` files or a ZIP containing them.

```python
from google.colab import files
from pathlib import Path
import zipfile

raw = Path("data/raw")
raw.mkdir(parents=True, exist_ok=True)
uploaded = files.upload()
for name, payload in uploaded.items():
    target = raw / name
    target.write_bytes(payload)
    if target.suffix.lower() == ".zip":
        with zipfile.ZipFile(target) as archive:
            archive.extractall(raw)
```

## 3. Prepare the fixed split

```python
!agent-distill prepare \
  --input data/raw \
  --output data/processed \
  --eval-ratio 0.2 \
  --seed 42
```

Inspect `data/processed/dataset_report.json`. Do not train if there are fewer than two
accepted trajectory IDs or if the evaluation split is empty.

## 4. Train the QLoRA adapter

```python
!nvidia-smi
!agent-distill train --config configs/qwen25_coder_3b_qlora.yaml
```

## 5. Generate and score held-out calls

```python
!agent-distill generate \
  --config configs/qwen25_coder_3b_qlora.yaml \
  --adapter outputs/qwen25-coder-3b-tool-use \
  --references data/processed/eval.jsonl \
  --output outputs/predictions.jsonl

!agent-distill score \
  --references data/processed/eval.jsonl \
  --predictions outputs/predictions.jsonl \
  --output outputs/metrics.json
```

## 6. Download reproducible artifacts

```python
import shutil
from google.colab import files

shutil.make_archive(
    "agent-distillation-artifacts",
    "zip",
    root_dir="outputs/qwen25-coder-3b-tool-use",
)
files.download("agent-distillation-artifacts.zip")
files.download("outputs/metrics.json")
files.download("data/processed/dataset_report.json")
```

