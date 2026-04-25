"""One-off generator for Colab_Maize_Pipeline_Clean.ipynb — run: python notebooks/_gen_colab_clean.py"""
import json
from pathlib import Path

cells = []


def md(s):
    lines = s.strip().split("\n")
    cells.append({"cell_type": "markdown", "metadata": {}, "source": [ln + "\n" for ln in lines]})


def code(s):
    lines = s.strip().split("\n")
    cells.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [ln + "\n" for ln in lines],
        }
    )


# --- Rich intro: structure, safety, no interference of markdown with code
md(
    r"""
# Maize leaf disease — Colab pipeline (clean)

This notebook runs the **MSc project code** from [**smart-agriculture**](https://github.com/FRIEZEWANDABWA/smart-agriculture) on Google Colab: download data → build splits → prepare images → train a CNN → evaluate on the test set.

---

### How this notebook is built (read once)

| Cell type | Role |
|-----------|------|
| **Gray markdown cells** | Explanations only. They are **not** Python. They do **not** run as code and **cannot** break your program. |
| **Code cells** | Actual commands. Only these execute when you press Run. |

**Always run code cells in the order they appear** (or use **Runtime → Run all** from the top). Skipping a step, or running training before the data-prep steps finish, is what usually causes errors.

---

### What you should see at the end

- A trained model file: `artifacts/checkpoints/exp01_cnn_raw/best_model.keras`
- Printed **confusion matrix** and **classification report** (accuracy, F1 per class) on the held-out **raw** test folder

---

### Phases (follow this order)

| # | Section | You change settings here? |
|---|---------|---------------------------|
| 1 | **Paths** — `USE_DRIVE`, `PROJECT_CANDIDATES` | Yes, if your Drive folder name differs |
| 2 | **Mount Drive** | Only if `USE_DRIVE = True` |
| 3 | **Set working folder** — `cd` into the repo | Usually no |
| 4 | **Install packages** (TensorFlow, OpenCV, …) | No |
| 5 | **Download Kaggle dataset** | No |
| 6 | **Build train/val/test CSVs** | No (same `seed=42` for reproducibility) |
| 7 | **Verify** CSV file paths on this machine | No |
| 8 | **Preprocess** → `data/processed/for_keras/` | `SKIP_PRECOMPUTE` if already done for *these* CSVs |
| 9 | **Copy raw** → `data/processed/for_keras_raw/` | `SKIP_RAW_MATERIALIZE` if already done |
| 10 | **Train** CNN on raw | `RUN_TRAIN`; long runtime (GPU) |
| 11 | **Evaluate** on test | No (must match the experiment name you trained) |

> **Important:** If your `train.csv` was created on **Windows** and you open the notebook on **Colab**, the paths inside the CSV may point to `G:\...` and will **not** work. The cells in sections 5–7 fix that by re-downloading and re-building splits *on Colab*.

**Kaggle dataset:** `smaranjitghose/corn-or-maize-leaf-disease-dataset` — class images must live under the subfolder **`data/`** (this notebook sets `RAW_ROOT` to that path automatically).
"""
)

md(
    r"""
## Phase 1 — Paths and flags

**Why:** Colab must know where your project lives (Google Drive) or that it should **clone** the GitHub repo into `/content`.

**You edit here:**
- `USE_DRIVE = True` — set **`False`** if you do *not* use Google Drive; the next phases will clone the repo and use `/content/...` only.
- `PROJECT_CANDIDATES` — first path that **exists** and contains `src/train/train.py` wins. Put your own Drive path at the top if needed.

**Expected result:** After running the code cell, nothing else happens yet; variables are set for the next cell.
"""
)

code(
    r"""
# --- Phase 1: configuration (edit if needed) ---
USE_DRIVE = True  # False = no Drive; repo will be cloned from GitHub into /content

PROJECT_CANDIDATES = [
    "/content/drive/MyDrive/final masters project/maize_disease_msc",
    "/content/drive/MyDrive/Masters in AI/Maize disease project/maize_disease_msc",
]
REPO_URL = "https://github.com/FRIEZEWANDABWA/smart-agriculture.git"
CLONE_DIR = "/content/maize_disease_msc_smart_agriculture"
"""
)

md(
    r"""
## Phase 2 — Mount Google Drive (optional)

**Why:** If your `maize_disease_msc` folder is synced under **My Drive**, Colab needs permission to read/write it (checkpoints, processed images).

**When to skip:** Set `USE_DRIVE = False` in Phase 1. Then this cell only prints a skip message (no sign-in window).

**Expected result:** If mounting: Google account prompt, then a line like *Mounted at /content/drive*. If skipping: *Skipping Drive mount*.

> Do **not** run training writing to a path you have not mounted yet; mount first, then resolve the project.
"""
)

code(
    r"""
# --- Phase 2: mount Drive (respects USE_DRIVE) ---
if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
else:
    print("Skipping Drive mount (USE_DRIVE = False).")
"""
)

md(
    r"""
## Phase 3 — Open the project folder (`cd` + `PYTHONPATH`)

**Why:** All commands (`python -m src....`) must run with the **repository root** as the current working directory, so `src` and `configs` resolve correctly.

**What it does:** Tries each `PROJECT_CANDIDATE` in order. If none exists (e.g. first time, no sync), it **clones** `REPO_URL` to `CLONE_DIR` and uses that.

**Expected result:** Prints *Using Drive project: …* or *Cloning from GitHub…* then *Using cloned project: …*, and *cwd =* a path ending in `maize_disease_msc` (or the clone directory).

**If it fails:** Check that Phase 1 paths match your Drive, or that `git` clone is allowed; fix `PROJECT_CANDIDATES` and re-run from Phase 1.
"""
)

code(
    r"""
# --- Phase 3: resolve repo and chdir ---
from pathlib import Path
import os
import shutil
import subprocess
import sys


def is_repo(p: Path) -> bool:
    return p.is_dir() and (p / "src" / "train" / "train.py").is_file()


PROJECT = None
for s in PROJECT_CANDIDATES:
    cand = Path(s)
    if is_repo(cand):
        PROJECT = cand.resolve()
        print("Using Drive project:", PROJECT)
        break

if PROJECT is None:
    print("Cloning from GitHub into", CLONE_DIR, "…")
    p = Path(CLONE_DIR)
    if p.exists():
        shutil.rmtree(p)
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, str(p)], check=True)
    PROJECT = p.resolve()
    print("Using cloned project:", PROJECT)

os.chdir(PROJECT)
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))
print("cwd =", Path.cwd())
"""
)

md(
    r"""
## Phase 4 — Install Python dependencies

**Why:** Colab’s default image may not have your exact `tensorflow` / `opencv` / `kagglehub` versions. This installs the same stack as the project’s `requirements.txt` plus `kagglehub` for the dataset.

**Expected result:** Ends with *pip OK* and no error tracebacks.

> If you **re-run** this after TensorFlow is already loaded, you might see warnings. A **Runtime → Restart runtime** after pip, then re-run from Phase 1, is the cleanest approach for long training runs.
"""
)

code(
    r"""
# --- Phase 4: dependencies ---
import subprocess, sys

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "kagglehub",
        "tensorflow>=2.15,<2.19",
        "opencv-python-headless>=4.8",
        "numpy>=1.24",
        "PyYAML>=6.0",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "tqdm>=4.66",
    ]
)
print("pip OK")
"""
)

md(
    r"""
## Phase 5 — Download the maize image dataset

**Why:** The thesis uses a public Kaggle set of maize/corn leaves; we need a local folder of images before splitting.

**What happens:** `kagglehub` downloads the dataset. We set `RAW_ROOT = <download> / "data"` because the class folders (`Common_Rust`, `Blight`, `Healthy`, `Gray_Leaf_Spot`) are under `data/`.

**Expected result:** Lines printing `RAW_ROOT = ...` and `classes: [...]` with **four** class names.

> First run can take several minutes (download size). Do not run the next phase until this finishes.
"""
)

code(
    r"""
# --- Phase 5: Kaggle data → RAW_ROOT ---
import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("smaranjitghose/corn-or-maize-leaf-disease-dataset")
RAW_ROOT = Path(path) / "data"
print("RAW_ROOT =", RAW_ROOT)
print("classes :", sorted(p.name for p in RAW_ROOT.iterdir() if p.is_dir()))
"""
)

md(
    r"""
## Phase 6 — Stratified train / val / test splits (CSV files)

**Why:** We need reproducible, stratified file lists (not random ad-hoc folders) for fair train–val–test and for the thesis.

**What happens:** Writes `data/interim/splits/train.csv`, `val.csv`, `test.csv` with columns `filepath, label`.

**Expected result:** Program prints *Classes: 4* (or the number of classes found), and counts for Train / Val / Test, plus *Wrote CSVs to...*

**If it fails:** `raw_root` wrong — confirm Phase 5 and that `RAW_ROOT` points to the directory that directly contains the class subfolders.
"""
)

code(
    r"""
# --- Phase 6: build_splits ---
import subprocess, sys
from pathlib import Path

out_dir = Path("data/interim/splits")
subprocess.run(
    [
        sys.executable,
        "-m",
        "src.data.build_splits",
        "--raw-root",
        str(RAW_ROOT),
        "--out-dir",
        str(out_dir),
        "--seed",
        "42",
    ],
    check=True,
)
"""
)

md(
    r"""
## Phase 7 — Sanity-check that CSV paths work *here*

**Why:** The CSVs store full paths. If you moved work from a PC, paths may still be Windows. Training would then fail to open images. This cell checks the first 20 rows per split.

**Expected result:** *Spot-check: 60 paths, 0 missing* and *OK.* (numbers may differ slightly).

**If it fails:** It stops with a message. **Re-run Phase 5 and Phase 6** on this Colab session (do not use old CSVs from another computer without rebuilding).
"""
)

code(
    r"""
# --- Phase 7: verify CSV paths on this machine ---
from pathlib import Path
import csv

splits_dir = Path("data/interim/splits")
missing = 0
checked = 0
for name in ("train", "val", "test"):
    csv_path = splits_dir / f"{name}.csv"
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for i, row in enumerate(r):
            if i >= 20:
                break
            p = Path(row["filepath"])
            checked += 1
            if not p.is_file():
                missing += 1
                if missing <= 5:
                    print("MISSING:", p)

print(f"Spot-check: {checked} paths, {missing} missing (first 20 rows per split).")
if missing:
    raise SystemExit(
        "CSV paths invalid here — re-run the kagglehub cell and build_splits on this Colab runtime."
    )
print("OK.")
"""
)

md(
    r"""
## Phase 8 — Precompute *preprocessed* images (for “pre” experiments)

**Why:** The thesis compares **raw** vs **preprocessed** (HSV mask, CLAHE, etc.). This step writes `*_pre.png` files into `data/processed/for_keras/{train,val,test}/`.

**Time:** Can be long (CPU, many images). Safe to run once per fresh split.

**Set `SKIP_PRECOMPUTE = True` only if** you already generated these folders in this same project **for the same** `train/val/test` CSVs (e.g. resumed session or synced Drive). Otherwise keep **`False`**.

**Expected result:** *Precompute done.* and progress bars for each split.
"""
)

code(
    r"""
# --- Phase 8: precompute preprocessed (toggle SKIP_PRECOMPUTE) ---
SKIP_PRECOMPUTE = False

import subprocess, sys
from pathlib import Path

if SKIP_PRECOMPUTE:
    print("Skipped precompute.")
else:
    for split in ("train", "val", "test"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.data.precompute_preprocessed",
                "--split-csv",
                str(Path("data/interim/splits") / f"{split}.csv"),
                "--out-root",
                "data/processed/for_keras",
                "--split-name",
                split,
            ],
            check=True,
        )
    print("Precompute done.")
"""
)

md(
    r"""
## Phase 9 — Copy *raw* files into Keras-style folders (same split)

**Why:** So **raw** experiments use the **same** images as the preprocessed branch, in `train/val/test` layout. Output: `data/processed/for_keras_raw/{train,val,test}/<class>/...`

**Set `SKIP_RAW_MATERIALIZE = True` only if** that tree is already complete for the current CSVs.

**Expected result:** *Raw materialize done.* with tqdm copy progress.
"""
)

code(
    r"""
# --- Phase 9: materialize raw splits (toggle SKIP_RAW_MATERIALIZE) ---
SKIP_RAW_MATERIALIZE = False

import subprocess, sys
from pathlib import Path

if SKIP_RAW_MATERIALIZE:
    print("Skipped raw materialize.")
else:
    for split in ("train", "val", "test"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "src.data.materialize_raw_splits",
                "--split-csv",
                str(Path("data/interim/splits") / f"{split}.csv"),
                "--out-root",
                "data/processed/for_keras_raw",
                "--split-name",
                split,
            ],
            check=True,
        )
    print("Raw materialize done.")
"""
)

md(
    r"""
## Phase 10 — Train: CNN on **raw** images (`exp01_cnn_raw`)

**Why:** Baseline in the 2×2 (CNN/ResNet × raw/pre) plan — this cell trains the **raw** path with the custom CNN from the repo.

**Saves to:** `artifacts/checkpoints/exp01_cnn_raw/best_model.keras` (and `training_log.csv`, `final_model.keras`).

**Set `RUN_TRAIN = False` only to** re-run evaluation only (after a successful train), or to save time while debugging.

**Expected result:** Keras `Epoch` logs, early stopping if applicable, then *Saved:* with path to `best_model.keras`.  
**Time:** Long on GPU; much longer on CPU. Enable **Runtime → Change runtime type → GPU** when possible.
"""
)

code(
    r"""
# --- Phase 10: training (toggle RUN_TRAIN) ---
RUN_TRAIN = True
EXPERIMENT = "exp01_cnn_raw"

import subprocess, sys
from pathlib import Path

ckpt = Path("artifacts/checkpoints") / EXPERIMENT / "best_model.keras"
if not RUN_TRAIN:
    print("Skipping train. Expect:", ckpt.resolve())
else:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "src.train.train",
            "--train-dir",
            "data/processed/for_keras_raw/train",
            "--val-dir",
            "data/processed/for_keras_raw/val",
            "--architecture",
            "cnn",
            "--experiment",
            EXPERIMENT,
            "--augment",
        ],
        check=True,
    )
    print("Saved:", ckpt.resolve())
"""
)

md(
    r"""
## Phase 11 — Evaluate on the held-out **raw** test set

**Why:** Report final accuracy, confusion matrix, and per-class metrics for the thesis.

**Input:** The same `EXPERIMENT` name you trained (`exp01_cnn_raw`) and the **raw** test folder that matches the training condition.

**Expected result:** Printed **confusion matrix** and **sklearn `classification_report`** (precision, recall, F1 per class).

**If it fails** with *Missing model*: run Phase 10 with `RUN_TRAIN = True`, or point `--model` to a checkpoint that actually exists on this disk.
"""
)

code(
    r"""
# --- Phase 11: evaluate (CNN, raw test) ---
import subprocess, sys
from pathlib import Path

EXPERIMENT = "exp01_cnn_raw"
model_path = Path("artifacts/checkpoints") / EXPERIMENT / "best_model.keras"
if not model_path.is_file():
    raise SystemExit(f"Missing: {model_path.resolve()}")

subprocess.run(
    [
        sys.executable,
        "-m",
        "src.eval.evaluate",
        "--model",
        str(model_path),
        "--test-dir",
        "data/processed/for_keras_raw/test",
        "--architecture",
        "cnn",
    ],
    check=True,
)
"""
)

md(
    r"""
## Next experiments (same code patterns)

| Goal | `train` / `val` dirs | `test` dir | Suggested `--experiment` |
|------|----------------------|------------|-------------------------|
| CNN, **preprocessed** | `data/processed/for_keras/train` & `val` | `data/processed/for_keras/test` | e.g. `exp02_cnn_pre` |
| ResNet50, preprocessed | same as above | same | e.g. `exp04_resnet_pre` (add `--architecture resnet50 --resnet-unfreeze 30` to the train command) |

**Rule:** *Architecture* and *raw vs pre* must match between training and `src.eval.evaluate`.

---

### Notebook hygiene (optional)

- **Runtime → Run all** after a full restart = safest way to run top-to-bottom.
- Keep **explanation cells** (markdown) — they are only documentation and do not affect execution of **code** cells.
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {
            "provenance": [],
            "toc_visible": True,
        },
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent / "Colab_Maize_Pipeline_Clean.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out)
