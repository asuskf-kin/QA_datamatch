"""Carga y validación mínima de config.yaml."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "config.yaml"


class Config(dict):
    """Dict con helpers para rutas y fecha de corrida."""

    def path(self, key: str) -> Path:
        p = Path(self["paths"][key])
        return p if p.is_absolute() else PROJECT_ROOT / p

    def path_or_none(self, key: str) -> Path | None:
        """Ruta opcional de la sección analysis/paths (null -> None)."""
        value = self.get("analysis", {}).get(key) or self["paths"].get(key)
        if not value:
            return None
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p

    @property
    def run_date(self) -> str:
        return str(self["project"].get("run_date") or date.today().strftime("%Y%m%d"))

    @property
    def tag(self) -> str:
        """Prefijo de archivos de salida, p.ej. Golfo_20261013."""
        return f"{self['project']['client']}_{self.run_date}"


def load_config(path: str | Path | None = None) -> Config:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path, encoding="utf-8") as f:
        raw: dict[str, Any] = yaml.safe_load(f)

    for section in ("project", "paths", "thresholds", "name_match", "columns", "outputs", "analysis"):
        if section not in raw:
            raise KeyError(f"Falta la sección '{section}' en {path}")

    thr = raw["thresholds"]
    if not 0 <= thr["match"] <= 1 or not 0 <= thr["super_match"] <= 1:
        raise ValueError("Los thresholds deben estar entre 0 y 1")
    if thr["super_match"] < thr["match"]:
        raise ValueError("thresholds.super_match no puede ser menor que thresholds.match")

    return Config(raw)
