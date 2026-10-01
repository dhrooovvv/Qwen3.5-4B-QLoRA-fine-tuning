"""Run the official dataset scorer and save its result with an experiment."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from config import load_config, save_resolved_config


def run_scorer(predictions: Path, output: Path, dataset_path: Path) -> None:
    """Delegate scoring to the provided scorer without recreating its logic."""

    command = [sys.executable, str(dataset_path / "score.py"),
               "--references", str(dataset_path / "development/labels.jsonl"),
               "--predictions", str(predictions), "--output", str(output)]
    subprocess.run(command, check=True)


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
    run_scorer(args.predictions, score_path, Path(config.dataset_path))
    with score_path.open("r", encoding="utf-8") as handle:
        json.load(handle)


if __name__ == "__main__":
    main()
