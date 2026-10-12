"""Shared multimodal prompt construction for training and inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMPT = (
    "The images are in page order. Extract included story text in reading order "
    "and return exactly {\"pages\":[[...],[...],[...]]}. Each line must have "
    "speaker and text. Use the labels character1 through character8, UNKNOWN, "
    "or NARRATION exactly as provided by the dataset. Preserve text, stutters, "
    "repeats, and meaningful punctuation; omit non-story text. "
    "Number characters character1, character2, ... in order of first appearance. "
    "A page with no story text is an empty list."
)


def build_messages(
    image_paths: list[str | Path], target_json: str | dict[str, Any] | None = None,
    prompt_text: str | None = None,
) -> list[dict[str, Any]]:
    """Build one user turn with all pages first, optionally followed by an answer."""

    if len(image_paths) != 3:
        raise ValueError("Each sequence must contain exactly three image paths")
    content: list[dict[str, str]] = [
        {"type": "image", "image": str(Path(path))} for path in image_paths
    ]
    content.append({"type": "text", "text": PROMPT if prompt_text is None else prompt_text})
    messages: list[dict[str, Any]] = [{"role": "user", "content": content}]
    if target_json is not None:
        if isinstance(target_json, dict):
            target_json = json.dumps(target_json, ensure_ascii=False, separators=(",", ":"))
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target_json}]})
    return messages
