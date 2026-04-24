from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet50_preprocess_input

from ..models.cnn_baseline import build_cnn_baseline
from ..models.resnet50_tl import build_resnet50
from ..utils.config import load_yaml, repo_root


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CNN or ResNet50 on folder-organized dataset (class subfolders).")
    p.add_argument("--train-dir", type=Path, required=True, help="train/<class>/images")
    p.add_argument("--val-dir", type=Path, required=True, help="val/<class>/images")
    p.add_argument("--architecture", choices=["cnn", "resnet50"], required=True)
    p.add_argument("--train-yaml", type=Path, default=None, help="Optional train.yaml override")
    p.add_argument("--experiment", type=str, default="exp", help="Checkpoint subfolder name")
    p.add_argument("--augment", action="store_true", help="Light Keras augmentation on training set")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--resnet-unfreeze", type=int, default=30, help="Unfreeze last N base layers when architecture=resnet50")
    return p.parse_args()


def augmentation_model(image_size: int) -> keras.Sequential:
    return keras.Sequential(
        [
            layers.RandomFlip("horizontal"),
            layers.RandomRotation(0.08),
            layers.RandomZoom(0.08),
            layers.RandomContrast(0.1),
        ],
        name="augment",
    )


def main() -> None:
    args = parse_args()
    root = repo_root()
    train_cfg_path = args.train_yaml or (root / "configs" / "train.yaml")
    train_cfg = load_yaml(train_cfg_path)

    batch_size = int(train_cfg.get("batch_size", 32))
    epochs = int(train_cfg.get("epochs", 40))
    lr = float(train_cfg.get("learning_rate", 1e-4))
    patience = int(train_cfg.get("early_stopping_patience", 6))
    ckpt_spec = Path(train_cfg.get("checkpoint_dir", "artifacts/checkpoints"))
    ckpt_root = ckpt_spec.resolve() if ckpt_spec.is_absolute() else (root / ckpt_spec).resolve()
    ckpt_dir = ckpt_root / args.experiment
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = keras.utils.image_dataset_from_directory(
        args.train_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42,
    )
    val_ds = keras.utils.image_dataset_from_directory(
        args.val_dir,
        image_size=(args.image_size, args.image_size),
        batch_size=batch_size,
        label_mode="categorical",
        shuffle=False,
    )

    class_names = train_ds.class_names
    num_classes = len(class_names)
    if val_ds.class_names != class_names:
        raise SystemExit("train_dir and val_dir must contain the same set of class folder names")

    aug = augmentation_model(args.image_size) if args.augment else None

    def apply_aug(images, labels):
        if aug is None:
            return images, labels
        return aug(images, training=True), labels

    train_ds = train_ds.map(apply_aug, num_parallel_calls=tf.data.AUTOTUNE)

    if args.architecture == "resnet50":

        def rn_norm(images, labels):
            return resnet50_preprocess_input(images), labels

        train_ds = train_ds.map(rn_norm, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.map(rn_norm, num_parallel_calls=tf.data.AUTOTUNE)

    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    if args.architecture == "cnn":
        model = build_cnn_baseline(num_classes=num_classes, image_size=args.image_size)
    else:
        model = build_resnet50(
            num_classes=num_classes,
            image_size=args.image_size,
            trainable_base_layers=args.resnet_unfreeze,
        )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.0),
        metrics=[keras.metrics.CategoricalAccuracy(name="accuracy")],
    )

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            filepath=str(ckpt_dir / "best_model.keras"),
            monitor="val_loss",
            save_best_only=True,
        ),
        keras.callbacks.CSVLogger(str(ckpt_dir / "training_log.csv")),
    ]

    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)
    model.save(ckpt_dir / "final_model.keras")
    with (ckpt_dir / "class_names.txt").open("w", encoding="utf-8") as f:
        f.write("\n".join(class_names))
    print(f"Saved checkpoints and logs under {ckpt_dir}")


if __name__ == "__main__":
    main()
