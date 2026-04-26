"""
YAML config loader with `extends:` base-config support.

Example:
    # configs/e2_inverse_N5000.yaml
    extends: "base.yaml"
    data:
      n_observations: 5000
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Lists are replaced, not concatenated."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    """Load a YAML config, resolving the optional `extends` field."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    extends = cfg.pop("extends", None)
    if extends:
        base_path = (path.parent / extends).resolve()
        base_cfg = load_config(base_path)
        cfg = _deep_merge(base_cfg, cfg)

    return cfg


def save_config_snapshot(cfg: dict, path: str | Path) -> None:
    """Dump merged config to disk for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
