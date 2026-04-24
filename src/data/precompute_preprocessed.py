from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
from tqdm import tqdm

from ..preprocess.pipeline import PreprocessConfig, preprocess_bgr_to_rgb


def read_split_csv(csv_path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((Path(row["filepath"]), row["label"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute preprocessed images from a split CSV.")
    ap.add_argument("--split-csv", type=Path, required=True, help="train.csv val.csv or test.csv")
    ap.add_argument(
        "--out-root",
        type=Path,
        required=True,
        help="Output root; images go to <out-root>/<split-name>/<label>/",
    )
    ap.add_argument(
        "--split-name",
        type=str,
        default="train",
        help="Subfolder under out-root, e.g. train, val, test",
    )
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--clahe-clip", type=float, default=2.0)
    ap.add_argument("--clahe-grid", type=int, default=8)
    ap.add_argument("--gaussian-ksize", type=int, default=3)
    args = ap.parse_args()

    cfg = PreprocessConfig(
        size=args.size,
        clahe_clip=args.clahe_clip,
        clahe_grid=args.clahe_grid,
        gaussian_ksize=args.gaussian_ksize,
    )

    rows = read_split_csv(args.split_csv)
    split_dir = args.out_root / args.split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for src, label in tqdm(rows, desc="preprocess"):
        img_bgr = cv2.imread(str(src), cv2.IMREAD_COLOR)
        if img_bgr is None:
            continue
        out_rgb = preprocess_bgr_to_rgb(img_bgr, cfg)
        class_dir = split_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        out_path = class_dir / (src.stem + "_pre.png")
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_path), out_bgr)

    print(f"Done. Wrote under {split_dir}")


if __name__ == "__main__":
    main()
