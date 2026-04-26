from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

from ..preprocess.pipeline import preprocess_bgr_to_rgb


def main() -> None:
    p = argparse.ArgumentParser(description="Automated Farm Photo Analyzer")
    p.add_argument("--image", type=Path, required=True, help="Path to raw farm photo")
    p.add_argument("--model", type=Path, required=True, help="Path to best_model.keras")
    p.add_argument("--architecture", choices=["cnn", "resnet50"], required=True)
    p.add_argument("--skip-preprocess", action="store_true", help="Skip OpenCV pipeline")
    args = p.parse_args()

    if not args.image.is_file():
        raise SystemExit(f"Image not found: {args.image.resolve()}")
    if not args.model.is_file():
        raise SystemExit(f"Model not found: {args.model.resolve()}")

    print("[1] Loading raw photograph...")
    bgr = cv2.imread(str(args.image))
    if bgr is None:
        raise SystemExit("Failed to read image.")

    print(f"[2] Processing pipeline: {'SKIPPED' if args.skip_preprocess else 'OpenCV Active (HSV Mask -> CLAHE -> Resize)'}")
    if not args.skip_preprocess:
        rgb = preprocess_bgr_to_rgb(bgr)
    else:
        resized = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    batch = np.expand_dims(rgb, axis=0).astype(np.float32)

    if args.architecture == "resnet50":
        batch = resnet50_preprocess_input(batch)

    print(f"[3] Booting Deep Learning Architecture ({args.architecture})...")
    model = keras.models.load_model(args.model)
    
    class_names_path = args.model.parent / "class_names.txt"
    if class_names_path.is_file():
        classes = class_names_path.read_text(encoding="utf-8").strip().split("\n")
    else:
        classes = ["Blight", "Common_Rust", "Gray_Leaf_Spot", "Healthy"]

    print("[4] Generating Medical Prediction...")
    preds = model.predict(batch, verbose=0)[0]
    best_idx = int(np.argmax(preds))
    best_class = classes[best_idx]
    confidence = preds[best_idx] * 100

    print("\n" + "="*50)
    print(" [AUTOMATED MAIZE DIAGNOSIS RESULT] ")
    print("="*50)
    print(f"  Prediction : {best_class}")
    print(f"  Confidence : {confidence:.2f}%")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()
