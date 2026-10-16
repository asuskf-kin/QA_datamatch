"""Rescate de no_match por nombre normalizado + distancia (notebook 1)."""

from __future__ import annotations

import pandas as pd

from .utils import haversine_m, normalize_name


def find_name_matches(
    no_match: pd.DataFrame,
    clients: pd.DataFrame,
    client_cols: dict,
    max_distance_m: float,
    unique_pos: bool = True,
) -> pd.DataFrame:
    """Devuelve dataplor_id -> cliente POS con mismo nombre normalizado dentro del radio.

    Columnas de salida: dataplor_id, new_pos_id, new_pos_name, new_pos_latitude,
    new_pos_longitude, name_match_distance_m.
    """
    cli = pd.DataFrame({
        "new_pos_id": clients[client_cols["id"]],
        "new_pos_name": clients[client_cols["name"]],
        "new_pos_latitude": pd.to_numeric(clients[client_cols["latitude"]], errors="coerce"),
        "new_pos_longitude": pd.to_numeric(clients[client_cols["longitude"]], errors="coerce"),
    })
    cli["name_norm"] = cli["new_pos_name"].map(normalize_name)
    cli = cli[(cli["name_norm"] != "") & cli["new_pos_latitude"].notna() & cli["new_pos_longitude"].notna()]

    dp = no_match[["dataplor_id", "dataplor_name", "dataplor_latitude", "dataplor_longitude"]].copy()
    dp["dataplor_latitude"] = pd.to_numeric(dp["dataplor_latitude"], errors="coerce")
    dp["dataplor_longitude"] = pd.to_numeric(dp["dataplor_longitude"], errors="coerce")
    dp["name_norm"] = dp["dataplor_name"].map(normalize_name)
    dp = dp[(dp["name_norm"] != "") & dp["dataplor_latitude"].notna() & dp["dataplor_longitude"].notna()]

    cand = dp.merge(cli, on="name_norm", how="inner")
    if cand.empty:
        return cand.assign(name_match_distance_m=pd.Series(dtype=float))[_OUT_COLS]

    cand["name_match_distance_m"] = haversine_m(
        cand["dataplor_latitude"], cand["dataplor_longitude"],
        cand["new_pos_latitude"], cand["new_pos_longitude"],
    )
    cand = cand[cand["name_match_distance_m"] < max_distance_m].sort_values("name_match_distance_m")

    # El más cercano por dataplor_id y, opcionalmente, cada pos_id usado una sola vez
    cand = cand.drop_duplicates("dataplor_id", keep="first")
    if unique_pos:
        cand = cand.drop_duplicates("new_pos_id", keep="first")
    return cand[_OUT_COLS].reset_index(drop=True)


_OUT_COLS = [
    "dataplor_id", "new_pos_id", "new_pos_name", "new_pos_latitude",
    "new_pos_longitude", "name_match_distance_m",
]


def apply_name_matches(df: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    """Actualiza el registro POS completo (id, nombre, coords) para los rescatados."""
    if matches.empty:
        return df
    out = df.merge(matches, on="dataplor_id", how="left")
    hit = out["new_pos_id"].notna()
    for col in ("pos_id", "pos_name", "pos_latitude", "pos_longitude"):
        if col in out.columns:
            out[col] = out[col].astype(object)
        out.loc[hit, col] = out.loc[hit, f"new_{col}"]
    out.loc[hit, "match_no_match"] = "match"
    out.loc[hit, "match_source"] = "name_distance"
    # Los scores del matcher pertenecían a otro candidato: no aplican al rescatado
    score_cols = [c for c in ("distance_score", "name_score", "address_score", "phonetic_score", "total_score") if c in out]
    out.loc[hit, score_cols] = float("nan")
    return out.drop(columns=[c for c in matches.columns if c.startswith("new_")])
