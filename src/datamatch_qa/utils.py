"""Normalización de texto y distancia geográfica (vectorizada)."""

from __future__ import annotations

import re
import unicodedata

import numpy as np
import pandas as pd

EARTH_RADIUS_M = 6_371_000


def normalize_name(s) -> str:
    """minúsculas, sin acentos, & -> y, sin puntuación, espacios colapsados."""
    if pd.isna(s):
        return ""
    s = unicodedata.normalize("NFKD", str(s).strip().lower())
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.replace("&", " y ")
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Distancia en metros entre arrays de coordenadas (grados)."""
    lat1, lon1, lat2, lon2 = (np.radians(np.asarray(x, dtype=float)) for x in (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))
