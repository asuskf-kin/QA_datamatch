"""Lectura / escritura de tablas (csv, xlsx, parquet)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # utf-8-sig elimina el BOM (el 'ï»¿' que aparecía en la primera columna)
        df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    elif suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path)
    elif suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Formato no soportado: {path}")
    df.columns = df.columns.str.strip()
    print(f"Leído {path.name}: {len(df):,} filas, {df.shape[1]} columnas")
    return df


def write_csv(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"Guardado {path.name}: {len(df):,} filas")
    return path
