# Manga Task (Advanced)

## Project setup

The project targets Python 3.11 because it is supported by the pinned PyTorch,
Transformers, and Qwen2.5-VL dependencies on both machines in scope.
`transformers==4.49.0` is pinned to keep the Qwen2.5-VL interfaces stable for
this scaffold.

### CUDA machine

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cu124 -r requirements-train.txt
python -m src.check_env
```

All package versions are pinned in the two requirements files. The CUDA file
uses the PyTorch cu124 wheel index; the Mac file uses the standard index and
the MPS backend when available.

## Folder overview

- `configs/`: base and machine profiles.
- `data/`: committed split placeholder; the future generated split goes here.
- `src/`: configuration, environment, preparation, training, inference, and evaluation entry points.
- `dataset/`: official downloaded dataset, intentionally gitignored.
- `logs/`, `results/`, `weights/`: future deliverables and artifacts.
- `docs/`: AI-use and citation records.

## Optional GUI

The local Gradio demo accepts exactly three uploaded pages and runs the saved adapter through `src/infer.py`.

From the project root, run:

```powershell
.venv312\Scripts\python.exe app_gradio.py
```

The GUI loads the cached 4-bit base model and adapter at startup. Close notebooks or other processes that hold the model before launching it; an RTX 4060 with 8 GB VRAM may not have enough free memory otherwise. Uploads are kept in Gradio's temporary folder, normally `C:\Users\<username>\AppData\Local\Temp\gradio`; clear that folder after stopping the app if needed.

For a CPU-only UI smoke test that does not load the model, run `.venv312\Scripts\python.exe app_gradio.py --no-model`.

## TODO

- Inspect the supplied dataset schema.
- Implement sequence-level preparation.
- Train and evaluate an adapted local model.
- Add the final approach, experiments, results, and next steps in this README.
- Add base-versus-fine-tuned comparison notebook and review its outputs.
