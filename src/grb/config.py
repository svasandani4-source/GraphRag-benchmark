from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    """Thin dict wrapper with attribute access and path helpers."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Config(value) if isinstance(value, dict) else value

    @property
    def artifacts(self) -> Path:
        p = Path(self["artifacts_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def results(self) -> Path:
        p = Path(self["results_dir"])
        p.mkdir(parents=True, exist_ok=True)
        return p


def load_config(path: str | Path) -> Config:
    with open(path) as f:
        return Config(yaml.safe_load(f))
