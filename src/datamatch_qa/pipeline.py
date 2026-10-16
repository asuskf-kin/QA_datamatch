"""Orquestación: explore (elegir thresholds) y run (datamatch final + exportes)."""

from __future__ import annotations

import pandas as pd

from .config import Config
from .io import read_table, write_csv
from .name_match import apply_name_matches, find_name_matches
from .analysis import analyze
from .thresholds import (
    apply_thresholds, data_quality_warnings, summary_table, threshold_grid, write_html_report,
)


def load_datamatch(cfg: Config) -> pd.DataFrame:
    df = read_table(cfg.path("datamatch"))
    df = df.rename(columns=cfg["columns"]["rename"])
    score = cfg["columns"]["score"]
    if score not in df.columns:
        raise KeyError(f"No existe la columna de score '{score}'")
    df[score] = pd.to_numeric(df[score], errors="coerce")
    dup = df["dataplor_id"].duplicated().sum()
    if dup:
        print(f"⚠️  {dup:,} dataplor_id duplicados en la entrada")
    return df


def explore(cfg: Config) -> None:
    df = load_datamatch(cfg)
    ex = cfg["thresholds"]["explore"]
    grid = threshold_grid(ex["start"], ex["stop"], ex["step"])
    score = cfg["columns"]["score"]
    for w in data_quality_warnings(df, score):
        print(f"⚠️  {w}")
    print(summary_table(df, grid, score).to_string(index=False))

    extra = analyze(df, cfg, score)
    out = cfg.path("output_dir") / f"threshold_report_{cfg.tag}.html"
    write_html_report(
        df, grid, ex["sample_size"], cfg["columns"]["display"], out,
        title=f"Datamatch {cfg['project']['client']} ({cfg['project']['country']}) · Threshold Analysis",
        score_col=score, extra_html=extra,
    )


def run(cfg: Config) -> pd.DataFrame:
    thr = cfg["thresholds"]
    score = cfg["columns"]["score"]

    df = load_datamatch(cfg)
    df = apply_thresholds(df, thr["match"], thr["super_match"], score)
    print(f"\nThreshold match={thr['match']}  super_match={thr['super_match']}")
    print(df["match_no_match"].value_counts().to_string())

    nm = cfg["name_match"]
    if nm["enabled"]:
        clients = read_table(cfg.path("clients"))
        matches = find_name_matches(
            df[df["match_no_match"] == "no_match"], clients, cfg["columns"]["clients"],
            nm["max_distance_m"], nm.get("unique_pos", True),
        )
        print(f"\nRescatados por nombre + distancia (<{nm['max_distance_m']} m): {len(matches):,}")
        df = apply_name_matches(df, matches)

    _report(df)

    out_dir = cfg.path("output_dir")
    out = cfg["outputs"]
    final_cols = [c for c in out["final_columns"] if c in df.columns]
    final = df[final_cols]
    write_csv(final, out_dir / f"Datamatch_Final_{cfg.tag}.csv")
    if out.get("write_latlon"):
        write_csv(df[[c for c in out["latlon_columns"] if c in df.columns]],
                  out_dir / f"Datamatch_Final_LatLon_{cfg.tag}.csv")
    if out.get("write_split"):
        write_csv(final[final["match_no_match"] == "match"], out_dir / f"match_{cfg.tag}.csv")
        write_csv(final[final["match_no_match"] == "no_match"], out_dir / f"no_match_{cfg.tag}.csv")
    return final


def _report(df: pd.DataFrame) -> None:
    n = len(df)
    print("\nResultado final:")
    for label, count in df["match_no_match"].value_counts().items():
        print(f"  {label:<9}: {count:>8,}  ({count / n:.1%})")
    print(f"  super_match: {int(df['super_match'].sum()):>6,}  ({df['super_match'].mean():.1%})")
    if "match_source" in df:
        print(df["match_source"].value_counts().rename("match_source").to_string())
    matched = df[df["match_no_match"] == "match"]
    dup_pos = matched["pos_id"].dropna().duplicated().sum()
    if dup_pos:
        print(f"⚠️  {dup_pos:,} pos_id asignados a más de un dataplor_id")
