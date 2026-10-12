"""CPU-only token-length report using the selected model processor.

Running this module may download processor files through Transformers. Ask for
permission before running it; no processor is loaded during ordinary imports.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from pathlib import Path
from typing import Any

from PIL import Image

try:
    from config import load_config
    from prepare import load_examples
    from prompts import build_messages
except ImportError:  # pragma: no cover
    from .config import load_config
    from .prepare import load_examples
    from .prompts import build_messages


def percentile(values: list[int], fraction: float) -> float:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def report_lengths(processor: Any, examples: list[dict[str, Any]], max_pixels: list[int], max_lengths: list[int]) -> dict[str, Any]:
    """Measure target-only and complete multimodal token lengths on CPU."""
    target_lengths: list[int] = []
    total_by_pixels: dict[str, list[int]] = {str(value): [] for value in max_pixels}
    for example in examples:
        messages = build_messages(example["image_paths"], example["target_json"])
        for message in messages:
            for item in message["content"]:
                if item["type"] == "image":
                    item["image"] = Image.open(item["image"]).convert("RGB")
        target_lengths.append(len(processor.tokenizer(example["target_json"], add_special_tokens=False)["input_ids"]))
        for pixels in max_pixels:
            encoded = processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=False,
                return_tensors="pt", return_dict=True, enable_thinking=False,
                images_kwargs={"size": {"shortest_edge": 65_536, "longest_edge": pixels}},
            )
            total_by_pixels[str(pixels)].append(int(encoded["input_ids"].shape[-1]))
    return {
        "target_json_tokens": {
            "min": min(target_lengths, default=0),
            "median": statistics.median(target_lengths) if target_lengths else 0,
            "p95": percentile(target_lengths, 0.95),
            "max": max(target_lengths, default=0),
        },
        "total_tokens": {
            str(pixels): {
                "min": min(values, default=0),
                "median": statistics.median(values) if values else 0,
                "max": max(values, default=0),
                "over_max_length": {str(limit): sum(value > limit for value in values) / len(values) if values else 0 for limit in max_lengths},
            }
            for pixels, values in total_by_pixels.items()
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", default=None)
    parser.add_argument("--processor", default=None)
    parser.add_argument("--max-pixels", nargs="+", type=int, default=None)
    parser.add_argument("--max-length", nargs="+", type=int, default=None)
    args = parser.parse_args()
    config = load_config(profile=args.profile)
    split = json.loads(Path(config.split_path).read_text(encoding="utf-8"))
    candidates = args.max_pixels or [config.max_pixels, 700_000, 500_000]
    limits = args.max_length or [config.max_length]
    model_name = args.processor or config.hf_model_name
    os.environ["HF_HUB_OFFLINE"] = "1"
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(
        model_name,
        revision=config.hf_model_revision,
        local_files_only=True,
    )
    examples = list(load_examples(config.dataset_path, split["train"]))
    print(json.dumps(report_lengths(processor, examples, candidates, limits), indent=2))


if __name__ == "__main__":
    main()
