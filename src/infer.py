"""Offline, deterministic inference for the base model or one LoRA adapter."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


class Line(TypedDict):
    speaker: str
    text: str


class SequencePrediction(TypedDict):
    sequence_id: str
    pages: list[list[Line]]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _open_images(image_paths: list[str | Path]) -> list[Any]:
    from PIL import Image

    return [Image.open(path).convert("RGB") for path in image_paths]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("val", "train_subset", "test"), required=True)
    parser.add_argument("--adapter", required=True, help="none or a local adapter directory")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-pixels", type=int, required=True)
    parser.add_argument("--max-new-tokens", type=int, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def _load_examples(config: Any, split_name: str, limit: int | None) -> list[dict[str, Any]]:
    from prepare import load_examples

    if split_name == "test":
        image_root = ROOT / config.dataset_path / "test" / "images"
        examples = []
        for sequence_dir in sorted(image_root.iterdir()):
            if not sequence_dir.is_dir():
                continue
            image_paths = [sequence_dir / f"{page:02d}.png" for page in (1, 2, 3)]
            if not all(path.exists() for path in image_paths):
                raise ValueError(f"Test sequence does not contain 01.png, 02.png, and 03.png: {sequence_dir}")
            examples.append({"sequence_id": sequence_dir.name, "image_paths": image_paths, "target_json": None})
        return examples[:limit] if limit is not None else examples

    split = json.loads((ROOT / config.split_path).read_text(encoding="utf-8"))
    ids = list(split["train"] if split_name == "train_subset" else split[split_name])
    if split_name == "train_subset":
        ids = ids[:16]
    if limit is not None:
        ids = ids[:limit]
    return list(load_examples(ROOT / config.dataset_path, ids))


def _adapter_info(adapter: str, base_repo: str) -> tuple[Path | None, str | None]:
    if adapter == "none":
        return None, None
    path = Path(adapter).expanduser().resolve()
    config_path = path / "adapter_config.json"
    weights_path = path / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Adapter must contain adapter_config.json and adapter_model.safetensors: {path}")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    declared_base = adapter_config.get("base_model_name_or_path")
    print(json.dumps({"adapter": str(path), "base_model_name_or_path": declared_base}, sort_keys=True))
    if declared_base != base_repo:
        raise ValueError(f"Adapter base {declared_base!r} does not match configured base {base_repo!r}")
    return path, _sha256(weights_path)


def _load_model(config: Any, adapter_path: Path | None, max_pixels: int):
    os.environ["HF_HUB_OFFLINE"] = "1"
    import unsloth  # noqa: F401
    import torch
    from peft import PeftModel
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        model_name=config.hf_model_name,
        revision=config.hf_model_revision,
        local_files_only=True,
        load_in_4bit=True,
        max_seq_length=config.max_length,
        use_gradient_checkpointing="unsloth",
    )
    processor.image_processor.size["longest_edge"] = max_pixels
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), local_files_only=True)
    model.eval()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for src/infer.py")
    return model, processor


def generate_one(model: Any, processor: Any, image_paths: list[str | Path], prompt_text: str, max_new_tokens: int) -> dict[str, Any]:
    """Shared generation path for the CLI and the Gradio UI."""

    from prompts import build_messages
    from PIL import Image
    import torch

    messages = build_messages(image_paths, prompt_text=prompt_text)
    rendered_prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    inputs = processor(
        images=[Image.open(path).convert("RGB") for path in image_paths],
        text=rendered_prompt,
        return_tensors="pt",
        truncation=False,
    )
    merge = int(processor.image_processor.merge_size)
    grid = inputs.get("image_grid_thw")
    image_tokens = [int(row[1] * row[2] // (merge * merge)) for row in grid]
    input_tokens = int(inputs["input_ids"].shape[-1])
    inputs = {key: value.to("cuda") if hasattr(value, "to") else value for key, value in inputs.items()}
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens, use_cache=True)
    elapsed = time.perf_counter() - started
    new_token_count = int(generated.shape[-1] - input_tokens)
    raw = processor.decode(generated[0, input_tokens:], skip_special_tokens=True).strip()
    return {
        "raw": raw,
        "prompt": rendered_prompt,
        "input_tokens": input_tokens,
        "image_tokens": image_tokens,
        "new_tokens": new_token_count,
        "truncated": new_token_count >= max_new_tokens,
        "elapsed_seconds": elapsed,
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 1024**3, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    from config import load_config, save_resolved_config
    from postprocess import parse_prediction_text, validate_prediction
    from prompts import build_messages

    config = load_config(ROOT, profile="cuda_4060")
    if args.max_pixels <= 0 or args.max_new_tokens <= 0:
        raise ValueError("max-pixels and max-new-tokens must be positive")
    examples = _load_examples(config, args.split, args.limit)
    if not examples:
        raise ValueError("The selected split is empty")
    adapter_path, adapter_hash = _adapter_info(args.adapter, config.hf_model_name)
    model, processor = _load_model(config, adapter_path, args.max_pixels)
    import torch

    output_path = args.out.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_resolved_config(config, output_path.parent)
    first_prompt = None
    first_input_tokens = None
    first_image_tokens = None
    rows: list[SequencePrediction] = []
    parse_failures = 0
    truncated = 0
    line_counts: list[int] = []
    started = time.perf_counter()
    for example in examples:
        from prompts import PROMPT
        result = generate_one(model, processor, example["image_paths"], PROMPT, args.max_new_tokens)
        prompt_text = result["prompt"]
        image_tokens = result["image_tokens"]
        if first_prompt is None:
            first_prompt = prompt_text
            first_input_tokens = result["input_tokens"]
            first_image_tokens = image_tokens
            print("FIRST_PROMPT_START")
            print(first_prompt)
            print("FIRST_PROMPT_END")
            print(json.dumps({"first_input_tokens": first_input_tokens, "first_image_tokens": first_image_tokens}))
        if result["truncated"]:
            truncated += 1
        raw = result["raw"]
        try:
            parsed = parse_prediction_text(raw)
            row: SequencePrediction = {"sequence_id": example["sequence_id"], "pages": parsed["pages"]}
            issues = validate_prediction(row)
            if issues:
                raise ValueError("; ".join(issues))
        except (ValueError, TypeError, json.JSONDecodeError):
            parse_failures += 1
            row = {"sequence_id": example["sequence_id"], "pages": [[], [], []]}
        rows.append(row)
        line_counts.append(sum(len(page) for page in row["pages"]))
    output_path.write_text("".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows), encoding="utf-8")
    elapsed = time.perf_counter() - started
    peak_vram = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
    versions = {name: _version(name) for name in ("torch", "transformers", "unsloth", "peft", "bitsandbytes")}
    summary = {"sequences": len(rows), "parse_failures": parse_failures, "truncated": truncated, "mean_lines_per_page": round(sum(line_counts) / (3 * len(rows)), 4), "seconds_per_sequence": round(elapsed / len(rows), 4), "peak_vram_gb": peak_vram}
    metadata = {
        "experiment": args.experiment,
        "split": args.split,
        "adapter": args.adapter,
        "adapter_sha256": adapter_hash,
        "base_model": config.hf_model_name,
        "base_revision": config.hf_model_revision,
        "max_pixels": args.max_pixels,
        "max_new_tokens": args.max_new_tokens,
        "first_prompt": first_prompt,
        "first_input_tokens": first_input_tokens,
        "first_image_tokens": first_image_tokens,
        **summary,
        "versions": versions,
    }
    (output_path.parent / f"{output_path.stem}.metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return summary


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
