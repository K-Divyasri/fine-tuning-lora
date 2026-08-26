"""Write the synthetic support-ticket dataset to CSV. Run this once, first.

    python generate_data.py

It writes three files to ./data: train.csv, val.csv, test.csv -- the same
deterministic split the package builds in memory (dataset.make_splits), just saved
to disk so notebooks and labs can load a CSV like a real dataset instead of
regenerating it every time.
"""

from __future__ import annotations

import csv
from pathlib import Path

from finetune_lab.dataset import make_splits

DATA_DIR = Path(__file__).parent / "data"


def _write_csv(path: Path, texts: list[str], labels: list[int]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["text", "label"])
        writer.writerows(zip(texts, labels))


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train, val, test = make_splits()
    _write_csv(DATA_DIR / "train.csv", train.texts, train.labels)
    _write_csv(DATA_DIR / "val.csv", val.texts, val.labels)
    _write_csv(DATA_DIR / "test.csv", test.texts, test.labels)
    print(f"wrote {DATA_DIR/'train.csv'}  ({len(train)} rows)")
    print(f"wrote {DATA_DIR/'val.csv'}  ({len(val)} rows)")
    print(f"wrote {DATA_DIR/'test.csv'}  ({len(test)} rows)")


if __name__ == "__main__":
    main()
