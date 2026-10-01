"""Report runtime and accelerator information without loading a model."""

from __future__ import annotations

import importlib.metadata
import platform


def version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "not installed"


def main() -> None:
    print(f"python: {platform.python_version()}")
    print(f"torch: {version('torch')}")
    try:
        import torch
        print(f"cuda_available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(f"gpu: {torch.cuda.get_device_name(0)}")
            print(f"vram_total_bytes: {total}")
            print(f"vram_free_bytes: {free}")
        print(f"mps_available: {torch.backends.mps.is_available()}")
    except (ImportError, RuntimeError) as exc:
        print(f"accelerator_probe: unavailable ({exc})")
    for package in ("transformers", "peft", "trl", "accelerate", "bitsandbytes", "mlx-vlm"):
        print(f"{package}: {version(package)}")


if __name__ == "__main__":
    main()
