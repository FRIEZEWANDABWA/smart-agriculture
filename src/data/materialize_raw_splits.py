from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

from tqdm import tqdm


def read_split_csv(csv_path: Path) -> list[tuple[Path, str]]:
    rows: list[tuple[Path, str]] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append((Path(row["filepath"]), row["label"]))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Copy original images from split CSV into <out-root>/<split-name>/<label>/ for Keras training."
    )
    ap.add_argument("--split-csv", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--split-name", type=str, required=True, choices=["train", "val", "test"])
    args = ap.parse_args()

    rows = read_split_csv(args.split_csv)
    split_dir = args.out_root / args.split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    for src, label in tqdm(rows, desc=f"copy {args.split_name}"):
        if not src.is_file():
            continue
        class_dir = split_dir / label
        class_dir.mkdir(parents=True, exist_ok=True)
        dest = class_dir / src.name
        if dest.exists():
            continue
        shutil.copy2(src, dest)

    print(f"Done. Copied to {split_dir}")


if __name__ == "__main__":
    main()
