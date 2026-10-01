"""Inference skeleton for three consecutive manga page images."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict


class Line(TypedDict):
    speaker: str
    text: str


class SequencePrediction(TypedDict):
    sequence_id: str
    pages: list[list[Line]]


def predict(sequence_id: str, image_paths: list[str | Path]) -> SequencePrediction:
    """Define the future output contract; model inference is not implemented."""

    if len(image_paths) != 3:
        raise ValueError("Exactly three consecutive image paths are required.")
    return {"sequence_id": sequence_id, "pages": [[], [], []]}


def main() -> None:
    """TODO: parse three image paths and run the local model pipeline."""
    raise NotImplementedError("Inference is intentionally not implemented yet.")


if __name__ == "__main__":
    main()
