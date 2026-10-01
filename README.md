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

### Mac Apple Silicon

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -r requirements-mac.txt
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

## TODO

- Inspect the supplied dataset schema.
- Implement sequence-level preparation.
- Train and evaluate an adapted local model.
- Add the final approach, experiments, results, and next steps in this README.
