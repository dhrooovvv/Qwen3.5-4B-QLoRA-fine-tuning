"""Local Gradio UI for the trained Qwen3.5-4B manga adapter."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from config import load_config
from infer import generate_one
from postprocess import parse_prediction_text, validate_prediction
from prompts import PROMPT

CONFIG = load_config(ROOT, profile="cuda_4060")
DEFAULT_PROMPT = PROMPT
ADAPTER_DIR = ROOT / "weights" / "qwen35_4b_20261010_181416"
if not ADAPTER_DIR.exists():
    candidates = [path.parent for path in (ROOT / "weights").glob("*/adapter_config.json")]
    if not candidates:
        raise FileNotFoundError("No LoRA adapter found under weights/")
    ADAPTER_DIR = max(candidates, key=lambda path: path.stat().st_mtime)

_MODEL = None
_PROCESSOR = None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _adapter_sha256_and_validate() -> str:
    config_path = ADAPTER_DIR / "adapter_config.json"
    weights_path = ADAPTER_DIR / "adapter_model.safetensors"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Adapter must contain adapter_config.json and adapter_model.safetensors: {ADAPTER_DIR}")
    adapter_config = json.loads(config_path.read_text(encoding="utf-8"))
    declared = adapter_config.get("base_model_name_or_path")
    if declared != CONFIG.hf_model_name:
        raise ValueError(f"Adapter base {declared!r} does not match configured base {CONFIG.hf_model_name!r}")
    return _sha256(weights_path)


ADAPTER_SHA256 = _adapter_sha256_and_validate()
print(f"Adapter path: {ADAPTER_DIR}")
print(f"Adapter SHA-256: {ADAPTER_SHA256}")
print(f"Base revision: {CONFIG.hf_model_revision}")


def _load_model():
    """Load the 4-bit base and adapter once, before the UI starts."""
    global _MODEL, _PROCESSOR
    if _MODEL is not None:
        return _MODEL, _PROCESSOR
    os.environ["HF_HUB_OFFLINE"] = "1"
    import unsloth  # noqa: F401
    from peft import PeftModel
    from unsloth import FastVisionModel

    base_model, processor = FastVisionModel.from_pretrained(
        model_name=CONFIG.hf_model_name,
        revision=CONFIG.hf_model_revision,
        local_files_only=True,
        load_in_4bit=True,
        max_seq_length=CONFIG.max_length,
        use_gradient_checkpointing="unsloth",
    )
    model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), local_files_only=True)
    processor.image_processor.size["longest_edge"] = CONFIG.max_pixels
    FastVisionModel.for_inference(model)
    model.eval()
    _MODEL, _PROCESSOR = model, processor
    import torch
    reserved = torch.cuda.memory_reserved() / 1024**3 if torch.cuda.is_available() else 0.0
    print(f"VRAM after load: {reserved:.3f} GB")
    print(f"Loaded base: {CONFIG.hf_model_name}")
    print(f"Loaded adapter: {ADAPTER_DIR}")
    return _MODEL, _PROCESSOR


def _validation_labels() -> dict[str, Any]:
    from prepare import load_labels
    split = json.loads((ROOT / CONFIG.split_path).read_text(encoding="utf-8"))
    labels = load_labels(ROOT / CONFIG.dataset_path)
    return {sequence_id: labels[sequence_id] for sequence_id in split["val"] if sequence_id in labels}


VALIDATION_LABELS = _validation_labels()


def _empty_pages_table() -> list[list[Any]]:
    return [["Page 1", "", "", ""], ["Page 2", "", "", ""], ["Page 3", "", "", ""]]


def _speaker_html(speaker: str) -> str:
    palette = {
        "narration": "#2563eb",
        "unknown": "#6b7280",
    }
    if speaker.lower().startswith("character"):
        try:
            number = int(speaker.removeprefix("character"))
        except ValueError:
            number = 0
        colors = ["#b91c1c", "#047857", "#7c3aed", "#c2410c", "#0369a1", "#a21caf", "#4d7c0f", "#be123c"]
        color = colors[(number - 1) % len(colors)] if number else "#374151"
    else:
        color = palette.get(speaker.lower(), "#374151")
    return f'<span style="color:{color};font-weight:600">{html.escape(speaker)}</span>'


def _speaker_legend() -> str:
    entries = [
        ("character1–character8", "#b91c1c"),
        ("NARRATION", "#2563eb"),
        ("UNKNOWN", "#6b7280"),
    ]
    return "**Speaker colours:** " + " · ".join(
        f'<span style="color:{color};font-weight:600">{name}</span>' for name, color in entries
    )


def _pages_table(pages: list[list[dict[str, str]]]) -> list[list[Any]]:
    rows: list[list[Any]] = []
    for page_number, page in enumerate(pages, 1):
        if page_number > 1:
            rows.append([f"--- Page {page_number} ---", "", "", ""])
        if not page:
            rows.append([f"Page {page_number}", "", "", ""])
        else:
            for order, line in enumerate(page, 1):
                rows.append([f"Page {page_number}", str(order), _speaker_html(line.get("speaker", "")), line.get("text", "")])
    return rows


def _page_counts_markdown(pages: list[list[dict[str, str]]]) -> str:
    parts = []
    for page_number, page in enumerate(pages, 1):
        count = len(page)
        note = " — no story lines detected" if count == 0 else ""
        parts.append(f"Page {page_number}: {count} line(s){note}")
    return "**Lines per page:** " + "  \n".join(parts)


def _label_markdown(sequence_id: str | None) -> str:
    if not sequence_id or sequence_id not in VALIDATION_LABELS:
        return "**Validation label:** none selected"
    pages = VALIDATION_LABELS[sequence_id].get("pages", [[], [], []])
    text = json.dumps({"pages": pages}, ensure_ascii=False, indent=2)
    return "**Validation label (reference):**\n\n```json\n" + text + "\n```"


def _predict(page1: str | None, page2: str | None, page3: str | None, prompt: str, validation_id: str | None, predictor: Callable[..., dict[str, Any]]):
    image_paths = [page1, page2, page3]
    if any(path is None or not str(path).strip() for path in image_paths):
        return _empty_pages_table(), "**Error:** upload exactly 3 images (Page 1, Page 2, and Page 3).", "", "", _label_markdown(validation_id), ""
    if not prompt or not prompt.strip():
        return _empty_pages_table(), "**Error:** the prompt cannot be empty.", "", "", _label_markdown(validation_id), ""
    started = time.perf_counter()
    result: dict[str, Any] = {}
    try:
        result = predictor(image_paths, prompt)
        raw = result["raw"]
        parsed = parse_prediction_text(raw)
        row = {"sequence_id": "uploaded", "pages": parsed["pages"]}
        issues = validate_prediction(row)
        if issues:
            raise ValueError("; ".join(issues))
        badge = "**JSON parsing:** PASS"
        pages = parsed["pages"]
        error_text = ""
    except (ValueError, TypeError, json.JSONDecodeError) as error:
        raw = result.get("raw", "")
        badge = f"**JSON parsing:** FAILED — `{html.escape(str(error))}`"
        pages = [[], [], []]
        error_text = str(error)
    elapsed = result.get("elapsed_seconds", time.perf_counter() - started)
    peak = result.get("peak_vram_gb", 0.0)
    stats = f"Elapsed: {elapsed:.2f} s  \nPeak VRAM reserved: {peak:.3f} GB"
    if error_text:
        stats += f"  \nParse error: {error_text}"
    return _pages_table(pages), badge, raw, stats, _label_markdown(validation_id), _page_counts_markdown(pages)


def _real_predictor(image_paths: list[str], prompt: str) -> dict[str, Any]:
    return generate_one(_MODEL, _PROCESSOR, image_paths, prompt, CONFIG.max_new_tokens)


def _stub_predictor(image_paths: list[str], prompt: str) -> dict[str, Any]:
    return {"raw": json.dumps({"pages": [[{"speaker": "NARRATION", "text": "stub output"}], [], []]}), "elapsed_seconds": 0.0, "peak_vram_gb": 0.0}


def build_app(predictor: Callable[..., dict[str, Any]]):
    import gradio as gr
    choices = [("No validation label (test sequence)", "")] + [
        (f"validation only — {sequence_id}", sequence_id) for sequence_id in VALIDATION_LABELS
    ]
    with gr.Blocks(title="Manga OCR — fine-tuned Qwen3.5-4B", analytics_enabled=False) as demo:
        gr.Markdown("# Manga OCR — fine-tuned Qwen3.5-4B\nDemo only; test predictions come from src/infer.py.")
        with gr.Row():
            page1 = gr.Image(label="Page 1", type="filepath")
            page2 = gr.Image(label="Page 2", type="filepath")
            page3 = gr.Image(label="Page 3", type="filepath")
        prompt = gr.Textbox(label="Prompt (read-only)", value=DEFAULT_PROMPT, lines=8, interactive=False)
        validation_id = gr.Dropdown(
            choices=choices,
            value="",
            label="validation only (optional)",
            allow_custom_value=False,
        )
        run_button = gr.Button("Run fine-tuned model", variant="primary")
        gr.Markdown("Upload exactly three pages. Outputs are read-only and are not saved by this UI.")
        label_output = gr.Markdown(label="Reference label")
        badge = gr.Markdown()
        gr.Markdown(_speaker_legend())
        table = gr.Dataframe(
            headers=["page", "reading order", "speaker", "text"],
            datatype=["str", "str", "html", "str"],
            label="Prediction",
            wrap=True,
            line_breaks=True,
            column_widths=["8%", "12%", "15%", "65%"],
            max_height=560,
            interactive=False,
        )
        page_counts = gr.Markdown()
        with gr.Accordion("Raw model output", open=False):
            raw_output = gr.Code(language="json", label="Raw output")
        stats = gr.Markdown()
        gr.Markdown(f"Adapter: `{ADAPTER_DIR}`  \nAdapter SHA-256: `{ADAPTER_SHA256}`  \nBase revision: `{CONFIG.hf_model_revision}`")
        run_button.click(
            lambda p1, p2, p3, text, val: _predict(p1, p2, p3, text, val, predictor),
            inputs=[page1, page2, page3, prompt, validation_id],
            outputs=[table, badge, raw_output, stats, label_output, page_counts],
            concurrency_limit=1,
        )
    return demo


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-model", action="store_true", help="Start a CPU-only UI smoke test with a stub predictor.")
    args = parser.parse_args()
    if args.no_model:
        predictor = _stub_predictor
    else:
        _load_model()
        predictor = _real_predictor
    app = build_app(predictor)
    app.queue(default_concurrency_limit=1)
    app.launch(server_name="127.0.0.1", share=False)


if __name__ == "__main__":
    main()
