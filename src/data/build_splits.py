from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from sklearn.model_selection import train_test_split


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images_by_class(raw_root: Path) -> dict[str, list[Path]]:
    by_class: dict[str, list[Path]] = defaultdict(list)
    if not raw_root.is_dir():
        raise FileNotFoundError(f"raw_root not found: {raw_root}")
    for class_dir in sorted(p for p in raw_root.iterdir() if p.is_dir()):
        label = class_dir.name
        for p in class_dir.rglob("*"):
            if p.suffix.lower() in IMAGE_EXTS and p.is_file():
                by_class[label].append(p)
    return dict(by_class)


def write_split_csv(rows: list[tuple[str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filepath", "label"])
        for fp, lab in rows:
            w.writerow([str(fp), lab])


def main() -> None:
    ap = argparse.ArgumentParser(description="Stratified train/val/test split file lists from folder dataset.")
    ap.add_argument("--raw-root", type=Path, required=True, help="Root with one folder per class")
    ap.add_argument("--out-dir", type=Path, required=True, help="Where to write train.csv val.csv test.csv")
    ap.add_argument("--train", type=float, default=0.70)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if abs(args.train + args.val + args.test - 1.0) > 1e-6:
        raise SystemExit("train + val + test must sum to 1.0")

    random.seed(args.seed)
    by_class = list_images_by_class(args.raw_root)
    if not by_class:
        raise SystemExit(f"No class folders/images under {args.raw_root}")

    all_rows: list[tuple[str, str]] = []
    for label, paths in by_class.items():
        if not paths:
            continue
        for p in paths:
            all_rows.append((str(p.resolve()), label))

    labels = [lab for _, lab in all_rows]
    paths = [fp for fp, _ in all_rows]

    train_p, temp_p, train_y, temp_y = train_test_split(
        paths,
        labels,
        test_size=(args.val + args.test),
        stratify=labels,
        random_state=args.seed,
    )
    rel_test = args.test / (args.val + args.test)
    val_p, test_p, val_y, test_y = train_test_split(
        temp_p,
        temp_y,
        test_size=rel_test,
        stratify=temp_y,
        random_state=args.seed,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_split_csv(list(zip(train_p, train_y)), args.out_dir / "train.csv")
    write_split_csv(list(zip(val_p, val_y)), args.out_dir / "val.csv")
    write_split_csv(list(zip(test_p, test_y)), args.out_dir / "test.csv")

    print(f"Classes: {len(by_class)}")
    print(f"Train: {len(train_p)}  Val: {len(val_p)}  Test: {len(test_p)}")
    print(f"Wrote CSVs to {args.out_dir}")


if __name__ == "__main__":
    main()
