# Manga Task (Advanced)

A multimodal manga dialogue extraction system built with **Qwen3.5-4B Vision**. Given three manga pages, the fine-tuned model identifies dialogue, narration, sound effects, and speakers, then returns the result as structured JSON.

## Approach

- Three-page sequences are kept together during preparation and split at the sequence level.
- Qwen3.5-4B Vision receives the three images and a shared extraction prompt.
- The model is adapted with 4-bit QLoRA/LoRA rather than trained from scratch. Lightweight adapter layers learn the manga-specific output format while the base model remains mostly frozen.
- Supervised fine-tuning targets JSON containing three page lists, with each line represented by a `speaker` and `text` field.
- Generated responses are parsed and schema-validated before evaluation or display.

### CUDA machine

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu124 -r requirements-train.txt
python -m src.check_env
```

### Mac Apple Silicon

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-mac.txt
python -m src.check_env
```

The pinned dependencies are recorded in `requirements-train.lock.txt`.

## Folder overview

- `configs/`: base and machine profiles.
- `data/`: reproducible sequence split metadata.
- `src/`: preparation, configuration, inference, evaluation, and post-processing.
- `weights/`: LoRA adapter configuration; large adapter weights are distributed separately.
- `docs/`: AI-use and external-resource records.
- `app_gradio.py`: optional local Gradio interface for exactly three pages.

## Optional GUI

The local Gradio demo accepts exactly three uploaded pages and runs the saved adapter through `src/infer.py`.

From the project root, run:

```powershell
.venv312\Scripts\python.exe app_gradio.py
```

The GUI loads the cached 4-bit base model and adapter at startup. Close notebooks or other processes that hold the model before launching it; an RTX 4060 with 8 GB VRAM may not have enough free memory otherwise. Uploads are kept in Gradio's temporary folder, normally `C:\Users\<username>\AppData\Local\Temp\gradio`; clear that folder after stopping the app if needed.

For a CPU-only UI smoke test that does not load the model, run `.venv312\Scripts\python.exe app_gradio.py --no-model`.

## Setup

The project was developed and tested with Python 3.12, CUDA PyTorch, Transformers, TRL, PEFT, bitsandbytes, and Unsloth.

```powershell
py -3.12 -m venv .venv312
uv pip install --python .venv312\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
uv pip install --python .venv312\Scripts\python.exe -r requirements-train.lock.txt
python -m src.check_env
```

The base model and processor must be available locally because inference uses offline loading. The training workflow is documented in `notebook/train_qwen35_4b.ipynb`.

## Reproducibility and references

Experiment settings, model revisions, environment details, and external resources are recorded in [`docs/CITATIONS.md`](docs/CITATIONS.md). AI-assisted project contributions are documented in [`docs/AI_USAGE.md`](docs/AI_USAGE.md).
