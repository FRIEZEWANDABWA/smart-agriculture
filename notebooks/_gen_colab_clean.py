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


md(
    r"""
# Maize disease — clean Colab pipeline

**GitHub:** [FRIEZEWANDABWA/smart-agriculture](https://github.com/FRIEZEWANDABWA/smart-agriculture.git)  
**Kaggle dataset:** `smaranjitghose/corn-or-maize-leaf-disease-dataset` → use subfolder **`data/`** as `RAW_ROOT`.

**Order:** (1) set `USE_DRIVE` and paths → (2) mount (skipped if `USE_DRIVE = False`) → (3) **Set `PROJECT` + `cd`** → dependencies → dataset → splits → …  

If you do **not** use Drive, set **`USE_DRIVE = False`**; the repo clones from GitHub into `/content/…`.

If split CSVs were built on another computer, **re-run dataset download + build_splits on Colab** so file paths in the CSVs point to files this machine can read.
"""
)

code(
    r"""
# --- Paths (edit lists / flags for your setup) ---
USE_DRIVE = True  # False = skip mount; project will clone from GitHub into /content

PROJECT_CANDIDATES = [
    "/content/drive/MyDrive/final masters project/maize_disease_msc",
    "/content/drive/MyDrive/Masters in AI/Maize disease project/maize_disease_msc",
]
REPO_URL = "https://github.com/FRIEZEWANDABWA/smart-agriculture.git"
CLONE_DIR = "/content/maize_disease_msc_smart_agriculture"
"""
)

code(
    r"""
# --- Mount Google Drive (only if USE_DRIVE is True) ---
if USE_DRIVE:
    from google.colab import drive

    drive.mount("/content/drive")
else:
    print("Skipping Drive mount (USE_DRIVE = False).")
"""
)

code(
    r"""
# --- Resolve project folder: Drive clone first, else git clone into /content ---
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

code(
    r"""
# --- Dependencies ---
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
### Download dataset and set `RAW_ROOT`

`RAW_ROOT` **must** be `Path(kagglehub_download) / "data"` for this dataset.
"""
)

code(
    r"""
import kagglehub
from pathlib import Path

path = kagglehub.dataset_download("smaranjitghose/corn-or-maize-leaf-disease-dataset")
RAW_ROOT = Path(path) / "data"
print("RAW_ROOT =", RAW_ROOT)
print("classes :", sorted(p.name for p in RAW_ROOT.iterdir() if p.is_dir()))
"""
)

md("### Stratified splits (`train.csv`, `val.csv`, `test.csv`)")

code(
    r"""
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

code(
    r"""
# Verify CSV paths exist on THIS runtime (avoids Windows paths inside CSVs)
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
### Precompute preprocessed images (`data/processed/for_keras/`)

Set `SKIP_PRECOMPUTE = True` if this tree is already complete for the current CSVs.
"""
)

code(
    r"""
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

md("### Materialize **raw** copies (`data/processed/for_keras_raw/`)")

code(
    r"""
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
### Train — CNN on **raw** (`exp01_cnn_raw`)

Saves `artifacts/checkpoints/exp01_cnn_raw/best_model.keras`.
"""
)

code(
    r"""
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

md("### Evaluate — **raw** test folder + CNN")

code(
    r"""
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
### Optional next runs (same notebook)

| Goal | Train dirs | Test dir | experiment |
|------|------------|----------|--------------|
| CNN preprocessed | `data/processed/for_keras/train` & `val` | `data/processed/for_keras/test` | `exp02_cnn_pre` |
| ResNet50 preprocessed | same + `--architecture resnet50 --resnet-unfreeze 30` | same | e.g. `exp04_resnet_pre` |

Always match **architecture** and **preprocessing** (raw vs `_pre.png` tree) between train and evaluate.
"""
)

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    },
    "cells": cells,
}

out = Path(__file__).resolve().parent / "Colab_Maize_Pipeline_Clean.ipynb"
out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("Wrote", out)
