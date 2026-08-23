"""Lưu và nạp artifact GRU có version cùng metadata checksum."""

from __future__ import annotations

import hashlib
from pathlib import Path

import torch

from src.models.rnn import ReturnGRU


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_artifact(bundle: dict, path: str | Path) -> str:
    artifact_path = Path(path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, artifact_path)
    return sha256(artifact_path)


def load_artifact(path: str | Path) -> tuple[dict, ReturnGRU]:
    bundle = torch.load(Path(path), map_location="cpu", weights_only=False)
    model = ReturnGRU(len(bundle["feature_columns"]), bundle["hidden_size"])
    model.load_state_dict(bundle["model_state"])
    model.eval()
    return bundle, model
