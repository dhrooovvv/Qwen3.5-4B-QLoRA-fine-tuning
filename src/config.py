"""Load one resolved experiment configuration from layered YAML files."""

from __future__ import annotations

import random
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Config:
    """All run-level settings shared by preparation, training, and inference."""

    dataset_path: str
    split_path: str
    hf_model_name: str
    hf_model_revision: str
    output_dir: str
    adapter_output_dir: str
    max_pixels: int
    max_length: int
    num_epochs: int
    lora_r: int
    lora_alpha: int
    learning_rate: float
    lora_dropout: float
    per_device_batch: int
    grad_accum: int
    load_in_4bit: bool
    finetune_vision_layers: bool
    max_new_tokens: int
    random_seed: int
    device: str
    profile: str
    experiment_name: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle) or {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def _read_cached_revision(model_name: str) -> str:
    """Read the cached model's main revision without contacting Hugging Face."""

    cache = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))
    repo_dir = cache / f"models--{model_name.replace('/', '--')}"
    ref_path = repo_dir / "refs" / "main"
    if not ref_path.exists():
        raise FileNotFoundError(f"Cached model revision file not found: {ref_path}")
    revision = ref_path.read_text(encoding="utf-8").strip()
    if not revision:
        raise ValueError(f"Cached model revision file is empty: {ref_path}")
    return revision


def load_config(root: str | Path = ".", profile: str | None = None,
                experiment: str | Path | None = None) -> Config:
    """Load base, machine-profile, then optional experiment YAML settings."""

    root_path = Path(root)
    values = _read_yaml(root_path / "configs/base.yaml")
    selected_profile = profile or values.get("profile", "base")
    if selected_profile != "base":
        values.update(_read_yaml(root_path / f"configs/{selected_profile}.yaml"))
    if experiment is not None:
        experiment_path = Path(experiment)
        if not experiment_path.is_absolute():
            experiment_path = root_path / experiment_path
        values.update(_read_yaml(experiment_path))
    values["hf_model_revision"] = _read_cached_revision(values["hf_model_name"])
    return Config(**{field: values[field] for field in Config.__annotations__})


def set_seed(seed: int) -> None:
    """Seed Python and available numerical libraries for deterministic setup."""

    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def save_resolved_config(config: Config, destination: str | Path) -> Path:
    """Write the resolved settings into an experiment result directory."""

    destination_path = Path(destination)
    destination_path.mkdir(parents=True, exist_ok=True)
    output_path = destination_path / "config.resolved.yaml"
    with output_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.as_dict(), handle, sort_keys=False)
    return output_path
