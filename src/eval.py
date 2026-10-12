"""Run the official dataset scorer and save its result with an experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import load_config, save_resolved_config


def run_scorer(predictions: Path, references: Path, output: Path, dataset_path: Path) -> None:
    """Delegate scoring to the provided scorer without recreating its logic."""

    command = [sys.executable, str(dataset_path / "score.py"),
               "--references", str(references),
               "--predictions", str(predictions), "--output", str(output)]
    subprocess.run(command, check=True)


def write_filtered_references(predictions: Path, references: Path, destination: Path) -> None:
    """Write original reference rows in prediction order without editing the source."""
    prediction_rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines() if line.strip()]
    reference_rows = [json.loads(line) for line in references.read_text(encoding="utf-8").splitlines() if line.strip()]
    reference_map = {row["sequence_id"]: row for row in reference_rows}
    ids = [row["sequence_id"] for row in prediction_rows]
    missing = [sequence_id for sequence_id in ids if sequence_id not in reference_map]
    if missing:
        raise ValueError(f"Prediction IDs are absent from references: {missing}")
    destination.write_text(
        "".join(json.dumps(reference_map[sequence_id], ensure_ascii=False, separators=(",", ":")) + "\n" for sequence_id in ids),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", type=Path)
    parser.add_argument("--experiment", default="evaluation")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()

    config = load_config(profile=args.profile)
    result_dir = Path(config.output_dir) / args.experiment
    result_dir.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, result_dir)
    score_path = result_dir / "scores.json"
    references = Path(config.dataset_path) / "development" / "labels.jsonl"
    filtered_references = result_dir / "references_filtered.jsonl"
    write_filtered_references(args.predictions, references, filtered_references)
    run_scorer(args.predictions, filtered_references, score_path, Path(config.dataset_path))
    with score_path.open("r", encoding="utf-8") as handle:
        json.load(handle)


if __name__ == "__main__":
    main()
