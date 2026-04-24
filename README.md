# Maize disease detection (MSc) — code

This folder is the implementation for **Enhancing Maize Disease Detection Using Image Preprocessing and Deep Learning Models**.

## Prerequisites

- Python 3.10+ (3.11 recommended)
- Windows: use **PowerShell**, run commands from this directory (`maize_disease_msc`)

## Setup

```powershell
cd "C:\Master in AI\Masters Project for Agriculture\maize_disease_msc"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data layout (PlantVillage maize classes)

Put images under `data\raw\plantvillage\<ClassName>\*.jpg` (one folder per class), for example:

- `Common_Rust`
- `Gray_Leaf_Spot`
- `Healthy`
- `Northern_Leaf_Blight`

(Use the exact folder names that exist in your PlantVillage export.)

## 1) Create stratified splits (CSV lists)

```powershell
python -m src.data.build_splits `
  --raw-root ".\data\raw\plantvillage" `
  --out-dir ".\data\interim\splits" `
  --seed 42
```

This writes `train.csv`, `val.csv`, `test.csv` with columns `filepath,label`.

## 2) Precompute preprocessed images (for “preprocessed” experiments)

Run once per split (adjust paths if needed):

```powershell
python -m src.data.precompute_preprocessed `
  --split-csv ".\data\interim\splits\train.csv" `
  --out-root ".\data\processed\for_keras" `
  --split-name train

python -m src.data.precompute_preprocessed `
  --split-csv ".\data\interim\splits\val.csv" `
  --out-root ".\data\processed\for_keras" `
  --split-name val

python -m src.data.precompute_preprocessed `
  --split-csv ".\data\interim\splits\test.csv" `
  --out-root ".\data\processed\for_keras" `
  --split-name test
```

## 3) Materialize **raw** Keras folders from the same CSV splits (baseline)

This copies originals into a parallel tree so **raw** and **preprocessed** experiments use the **same** train/val/test images.

```powershell
python -m src.data.materialize_raw_splits `
  --split-csv ".\data\interim\splits\train.csv" `
  --out-root ".\data\processed\for_keras_raw" `
  --split-name train

python -m src.data.materialize_raw_splits `
  --split-csv ".\data\interim\splits\val.csv" `
  --out-root ".\data\processed\for_keras_raw" `
  --split-name val

python -m src.data.materialize_raw_splits `
  --split-csv ".\data\interim\splits\test.csv" `
  --out-root ".\data\processed\for_keras_raw" `
  --split-name test
```

## 4) Train (raw vs preprocessed)

The Keras loader resizes to 224×224. Raw vs preprocessed differs only by which folder tree you point to.

**Example — CNN on raw train/val:**

```powershell
python -m src.train.train `
  --train-dir ".\data\processed\for_keras_raw\train" `
  --val-dir ".\data\processed\for_keras_raw\val" `
  --architecture cnn `
  --experiment exp01_cnn_raw `
  --augment
```

**Example — CNN on preprocessed train/val:**

```powershell
python -m src.train.train `
  --train-dir ".\data\processed\for_keras\train" `
  --val-dir ".\data\processed\for_keras\val" `
  --architecture cnn `
  --experiment exp02_cnn_pre `
  --augment
```

**Example — ResNet50 on preprocessed:**

```powershell
python -m src.train.train `
  --train-dir ".\data\processed\for_keras\train" `
  --val-dir ".\data\processed\for_keras\val" `
  --architecture resnet50 `
  --experiment exp04_resnet_pre `
  --resnet-unfreeze 30 `
  --augment
```

Checkpoints and logs: `artifacts\checkpoints\<experiment>\`.

## 5) Evaluate on held-out test folder

```powershell
python -m src.eval.evaluate `
  --model ".\artifacts\checkpoints\exp02_cnn_pre\best_model.keras" `
  --test-dir ".\data\processed\for_keras\test" `
  --architecture cnn
```

For **raw** test evaluation, point `--test-dir` to `.\data\processed\for_keras_raw\test` and use `--architecture` matching the trained model.

## Configuration

- `configs\data.yaml` — preprocessing parameters (HSV range, CLAHE, blur).
- `configs\train.yaml` — batch size, epochs, learning rate, early stopping.

## Next implementation pieces (tell me which you want first)

- **Synthetic corruption** suite + robustness evaluation tables.
- **Grad-CAM** notebook or script for XAI figures.
- **McNemar / bootstrap** statistical comparison between paired raw vs preprocessed predictions.
