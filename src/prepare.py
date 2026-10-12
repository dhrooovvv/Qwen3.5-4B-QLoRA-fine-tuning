"""Prepare sequence-level splits and load verbatim training examples."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator

try:
    from config import load_config, set_seed
except ImportError:  # pragma: no cover
    from .config import load_config, set_seed


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read JSONL records without changing label values."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_sequences(dataset_path: str | Path) -> list[dict[str, Any]]:
    return json.loads((Path(dataset_path) / "sequences.json").read_text(encoding="utf-8"))


def load_labels(dataset_path: str | Path) -> dict[str, dict[str, Any]]:
    return {row["sequence_id"]: row for row in read_jsonl(Path(dataset_path) / "development" / "labels.jsonl")}


def create_split(dataset_path: str | Path, split_path: str | Path, seed: int,
                 train_count: int = 64, val_count: int = 16) -> dict[str, list[str]]:
    """Create a seeded split using sequence IDs only."""
    development = [row["sequence_id"] for row in load_sequences(dataset_path) if row["split"] == "development"]
    if len(development) != train_count + val_count:
        raise ValueError(f"Expected {train_count + val_count} development sequences, found {len(development)}")
    shuffled = development.copy()
    random.Random(seed).shuffle(shuffled)
    train_ids = shuffled[:train_count]
    val_ids = shuffled[train_count:]
    labels = load_labels(dataset_path)
    empty_page_ids = sorted(
        sequence_id for sequence_id in development
        if any(not page for page in labels[sequence_id]["pages"])
    )
    if not any(sequence_id in empty_page_ids for sequence_id in val_ids):
        empty_train_ids = [sequence_id for sequence_id in train_ids if sequence_id in empty_page_ids]
        if not empty_train_ids:
            raise AssertionError("Expected an empty-page sequence in the training split")
        # Preserve the seeded split except for one deterministic swap.
        moved_to_val = min(empty_train_ids)
        moved_to_train = max(val_ids)
        train_ids[train_ids.index(moved_to_val)] = moved_to_train
        val_ids[val_ids.index(moved_to_train)] = moved_to_val
    split = {"train": sorted(train_ids), "val": sorted(val_ids)}
    if set(split["train"]) & set(split["val"]):
        raise AssertionError("Sequence appears in both train and validation splits")
    output = Path(split_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(split, indent=2) + "\n", encoding="utf-8")
    return split


def load_examples(dataset_path: str | Path, split_ids: list[str]) -> Iterator[dict[str, Any]]:
    """Yield paths and compact target JSON for requested development sequences."""
    root = Path(dataset_path)
    sequence_map = {row["sequence_id"]: row for row in load_sequences(root)}
    labels = load_labels(root)
    for sequence_id in split_ids:
        sequence = sequence_map[sequence_id]
        target = labels.get(sequence_id)
        if target is None:
            raise ValueError(f"No development labels for sequence {sequence_id}")
        if len(sequence["images"]) != 3 or len(target["pages"]) != 3:
            raise ValueError(f"Sequence {sequence_id} does not contain exactly three pages")
        yield {
            "sequence_id": sequence_id,
            "image_paths": [root / path for path in sequence["images"]],
            "target_json": json.dumps({"pages": target["pages"]}, ensure_ascii=False, separators=(",", ":")),
        }


def _median(values: list[int]) -> float:
    values = sorted(values)
    if not values:
        return 0
    middle = len(values) // 2
    return float(values[middle]) if len(values) % 2 else (values[middle - 1] + values[middle]) / 2


def composition(split_ids: list[str], labels: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pages = [page for sequence_id in split_ids for page in labels[sequence_id]["pages"]]
    lines = [line for page in pages for line in page]
    speaker_counts = [len({line["speaker"] for page in labels[sid]["pages"] for line in page}) for sid in split_ids]
    return {
        "sequences": len(split_ids),
        "lines": len(lines),
        "empty_pages": sum(not page for page in pages),
        "narration_lines": sum(line["speaker"] == "NARRATION" for line in lines),
        "unknown_lines": sum(line["speaker"] == "UNKNOWN" for line in lines),
        "speakers_per_sequence": {"min": min(speaker_counts, default=0), "median": _median(speaker_counts), "max": max(speaker_counts, default=0)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(profile=args.profile)
    seed = config.random_seed if args.seed is None else args.seed
    set_seed(seed)
    split = create_split(config.dataset_path, config.split_path, seed)
    labels = load_labels(config.dataset_path)
    print(json.dumps({name: composition(ids, labels) for name, ids in split.items()}, indent=2))
    print(f"overlap={set(split['train']) & set(split['val'])}")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
