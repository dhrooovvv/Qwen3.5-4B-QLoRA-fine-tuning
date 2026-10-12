"""Schema-only validation for generated predictions."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_prediction_text(raw_text: str) -> dict[str, Any]:
    """Parse one model response and always return exactly three page lists."""
    cleaned = raw_text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    candidates = [cleaned, re.sub(r",\s*([}\]])", r"\1", cleaned)]
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and isinstance(value.get("pages"), list):
            pages = value["pages"][:3]
            pages += [[] for _ in range(3 - len(pages))]
            value["pages"] = pages
            return value
    raise ValueError("Model response did not contain valid JSON with pages")


def validate_prediction(row: dict[str, Any], expected_pages: int = 3) -> list[str]:
    """Return validation issues without changing or heuristically editing a row."""
    issues: list[str] = []
    if set(row) != {"sequence_id", "pages"}:
        return ["expected exactly sequence_id and pages"]
    if not isinstance(row["sequence_id"], str) or not row["sequence_id"].strip():
        issues.append("sequence_id must be a non-empty string")
    pages = row["pages"]
    if not isinstance(pages, list) or len(pages) != expected_pages:
        return issues + [f"pages must contain exactly {expected_pages} lists"]
    for page_number, page in enumerate(pages, 1):
        if not isinstance(page, list):
            issues.append(f"page {page_number} is not a list")
            continue
        for line_number, line in enumerate(page, 1):
            if not isinstance(line, dict) or set(line) != {"speaker", "text"}:
                issues.append(f"page {page_number} line {line_number} has the wrong schema")
                continue
            speaker, text = line["speaker"], line["text"]
            if not isinstance(speaker, str) or not speaker.strip():
                issues.append(f"page {page_number} line {line_number} has an invalid speaker")
            if not isinstance(text, str):
                issues.append(f"page {page_number} line {line_number} has non-string text")
    return issues
