# Citations

Record every external model, dataset, codebase, and license used by the project here.

| Resource | Version or revision | License | Link | How it is used |
|---|---|---|---|---|
| Unsloth Qwen3.5 4B Vision notebook | Reference copy in `notebook/Qwen3_5_(4B)_Vision.ipynb` | LGPL-3.0 | https://github.com/unslothai/notebooks/blob/main/Qwen3_5_(4B)_Vision.ipynb | Training/model-loading structure adapted for this project; installation, LaTeX data, decoding, and trainer settings were replaced. |
| Gradio | 6.30.0 | Installed package metadata does not declare a license (`License` field and license classifiers were empty) | https://pypi.org/project/gradio/ | Optional local upload-and-predict GUI in `app_gradio.py`. |

## Environment (measured)

Measured on 2026-10-04 from `D:\byte task2 manga adv\.venv312` using `importlib.metadata`, PyTorch, `nvidia-smi`, and the Windows platform APIs. No WSL installation is present, so this is native Windows.

| Component | Measured value |
|---|---|
| Python | 3.12.10 |
| torch | 2.8.0+cu128 |
| torchvision | 0.23.0+cu128 |
| CUDA runtime | 12.8 |
| cuDNN | 91002 |
| unsloth | 2026.9.14 |
| unsloth_zoo | 2026.9.9 |
| transformers | 5.2.0 |
| tokenizers | 0.22.2 |
| trl | 0.22.2 |
| peft | 0.21.2 |
| accelerate | 1.15.0 |
| bitsandbytes | 0.50.2 |
| xformers | 0.0.32.post2 |
| triton-windows | 3.4.0.post21 |
| torchao | 0.18.0 |
| datasets | 4.3.0 |
| huggingface_hub | 1.33.0 |
| GPU | NVIDIA GeForce RTX 4060 |
| GPU memory | 8188 MiB |
| NVIDIA driver | 591.86 |
| OS | Windows 11, build 26200; native Windows, no WSL |
| First processor report | `unsloth/Qwen3.5-4B`, revision `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636` |
| Current model and processor repo | `Qwen/Qwen3.5-4B` |
| Current model and processor revision | `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` read from the local `refs/main` file |
| Model weights | Not downloaded; only processor/tokenizer files were downloaded |

## Deviations from the Unsloth notebook install

- **VERIFIED** — The notebook's Colab install cell was not run; the environment was built from the terminal.
- **UNVERIFIED** — `torch` was set to `2.8.0+cu128` from the PyTorch cu128 index after an unsloth install had pulled `2.12.1+cpu`.
- **UNVERIFIED** — `xformers==0.0.32.post2` was installed with `--no-deps` because unsloth declares an xformers requirement on Windows.
- **UNVERIFIED** — `triton-windows` was pinned below 3.5 (`3.4.0.post21`) to match torch 2.8; the first unsloth-installed version was `3.8.0.post29`.
- **VERIFIED** — `causal_conv1d` is not installed; the notebook's prebuilt wheel is `linux_x86_64` only, so the transformers torch implementation is the intended fallback.
- **VERIFIED** — `flash-linear-attention`, `fla-core`, and `torchcodec` are not installed.
- **UNVERIFIED** — `trl` was pinned to `0.22.2` after unsloth had installed `0.24.0`.
- **UNVERIFIED** — Installed unsloth and unsloth_zoo versions are newer than the version banner saved in the notebook output.
- **UNVERIFIED** — “Failed to hardlink” warnings from uv are harmless and caused by the cache and project being on different drives.

## Reproducing the environment

Run these commands in order from the project root, targeting `.venv312` throughout:

1. `py -3.12 -m venv .venv312`
2. `uv pip install --python .venv312\Scripts\python.exe --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0`
3. `uv pip install --python .venv312\Scripts\python.exe unsloth`
4. `uv pip install --python .venv312\Scripts\python.exe "triton-windows<3.5"`
5. `uv pip install --python .venv312\Scripts\python.exe --no-deps "transformers==5.2.0" "tokenizers>=0.22.0,<=0.23.0" "trl==0.22.2" unsloth unsloth_zoo "torchao>=0.16.0"`
6. `uv pip install --python .venv312\Scripts\python.exe --no-deps xformers==0.0.32.post2`
7. `uv pip install --python .venv312\Scripts\python.exe --no-deps -r requirements-train.lock.txt`
8. `uv pip check --python .venv312\Scripts\python.exe`

The CUDA index must be used for the torch step. Installing the lock file
directly from the default index can replace CUDA torch with a CPU build.

## External code and models

- [Unsloth `Qwen3_5_(4B)_Vision.ipynb`](https://github.com/unslothai/notebooks/blob/main/Qwen3_5_(4B)_Vision.ipynb), LGPL-3.0; copied unchanged to `docs/reference/`.
- [Unsloth `Qwen3_VL_(8B)_Vision.ipynb`](https://github.com/unslothai/notebooks/blob/main/Qwen3_VL_(8B)_Vision.ipynb), LGPL-3.0; cited as an external reference. It was not present in the project's `notebook/` directory, so no local copy or hash was created.
- [Model `Qwen/Qwen3.5-4B`](https://huggingface.co/Qwen/Qwen3.5-4B), license: Apache-2.0 as read from the cached snapshot's `LICENSE` file. The cached model snapshot was downloaded on 2026-10-03.
- The first token report used the `unsloth/Qwen3.5-4B` processor at revision `3764fa359b9082ea5a1e4a5e3ac3aaf6e9671636`. It was re-run with the official `Qwen/Qwen3.5-4B` processor at revision `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a`. The compared processor files differ in SHA-256, but the image-token counts were identical.
- Libraries used include [PyTorch](https://pytorch.org/), [Transformers](https://github.com/huggingface/transformers), [TRL](https://github.com/huggingface/trl), [Unsloth](https://github.com/unslothai/unsloth), [PEFT](https://github.com/huggingface/peft), [bitsandbytes](https://github.com/bitsandbytes-foundation/bitsandbytes), [xformers](https://github.com/facebookresearch/xformers), [datasets](https://github.com/huggingface/datasets), and the other packages recorded in `requirements-train.lock.txt`.
- The BYTE ML Task dataset was provided by the club.
- `LaTeX_OCR` is **not used**.

## Reference notebook hashes

| File | SHA-256 |
|---|---|
| `docs/reference/Qwen3_5_(4B)_Vision.ipynb` | `03F9E4C20E50B11C115A58C89569C5337D339795E24CCB709CB2BE821B8C49FB` |
