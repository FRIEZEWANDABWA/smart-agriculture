from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a saved Keras model on a test directory.")
    p.add_argument("--model", type=Path, required=True, help="Path to .keras model")
    p.add_argument("--test-dir", type=Path, required=True, help="test/<class>/images")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--architecture", choices=["cnn", "resnet50"], required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    model = keras.models.load_model(args.model)

    if args.architecture == "resnet50":
        from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

        def rn_norm(x, y):
            return resnet50_preprocess_input(x), y
    else:

        def rn_norm(x, y):
            return x, y

    ds = keras.utils.image_dataset_from_directory(
        args.test_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False,
    )
    class_names = ds.class_names
    ds = ds.map(rn_norm, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    y_true = []
    y_pred = []
    for xb, yb in ds:
        pb = model.predict(xb, verbose=0)
        y_true.append(np.argmax(yb.numpy(), axis=1))
        y_pred.append(np.argmax(pb, axis=1))
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    cr = classification_report(y_true, y_pred, target_names=class_names, digits=4)

    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)
    print()
    print(cr)

    report_path = args.model.parent / "eval_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("Confusion matrix (rows=true, cols=pred):\n")
        f.write(f"{cm}\n\n")
        f.write(cr)
        f.write("\n")
    print(f"\nEvaluation report saved to {report_path}")


if __name__ == "__main__":
    main()
