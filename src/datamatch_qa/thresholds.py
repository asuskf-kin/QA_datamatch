"""Threshold explorer (notebook 0) y flags match / super_match."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCORE_COLS = ["distance_score", "name_score", "address_score", "phonetic_score", "total_score"]


def threshold_grid(start: float, stop: float, step: float) -> list[float]:
    n = int(round((start - stop) / step)) + 1
    return [round(start - i * step, 2) for i in range(n)]


def summary_table(df: pd.DataFrame, thresholds: list[float], score_col: str = "total_score") -> pd.DataFrame:
    """Cuántos registros pasan cada threshold."""
    scores = df[score_col]
    rows = [
        {"threshold": t, "n_qualify": int((scores >= t).sum()), "pct_total": (scores >= t).mean() * 100}
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def marginal_samples(
    df: pd.DataFrame, thresholds: list[float], n: int, cols: list[str], score_col: str = "total_score"
) -> dict[float, pd.DataFrame]:
    """Para cada threshold, los n registros con menor score que aún lo superan."""
    cols = [c for c in cols if c in df.columns]
    scored = df[df[score_col].notna()]
    return {
        t: scored[scored[score_col] >= t].nsmallest(n, score_col)[cols].reset_index(drop=True)
        for t in thresholds
    }


def _style(sample: pd.DataFrame, vmin: float, vmax: float):
    fmt = {c: "{:.4f}" for c in SCORE_COLS if c in sample.columns}
    sub_scores = [c for c in SCORE_COLS[:-1] if c in sample.columns]
    sty = sample.style.format(fmt, na_rep="—")
    if "total_score" in sample.columns:
        sty = sty.background_gradient(subset=["total_score"], cmap="RdYlGn", vmin=vmin, vmax=vmax)
    if sub_scores:
        sty = sty.background_gradient(subset=sub_scores, cmap="RdYlGn", vmin=0, vmax=1)
    return sty.set_properties(**{"font-size": "11px", "padding": "4px 8px", "white-space": "nowrap"})


def data_quality_warnings(df: pd.DataFrame, score_col: str = "total_score") -> list[str]:
    """Avisos sobre componentes que no aportan (p.ej. address_score siempre 0 si el POS no trae dirección)."""
    scored = df[df[score_col].notna()]
    warns = []
    for c in SCORE_COLS[:-1]:
        if c not in scored.columns or scored.empty:
            continue
        v = scored[c].dropna()
        if v.empty or v.nunique() == 1:
            val = "vacío" if v.empty else f"siempre {v.iloc[0]:.2f}"
            warns.append(f"{c} {val}: no discrimina y el total_score queda acotado por su peso.")
    return warns


_CSS = """
:root{--bg:#fcfcfb;--ink:#0b0b0b;--muted:#52514e;--line:#dddcd7;--th:#1a1a2e}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--ink:#fff;--muted:#c3c2b7;--line:#3a3a38;--th:#2b2b45}}
body{font-family:system-ui,sans-serif;margin:24px;background:var(--bg);color:var(--ink)}
h3{border-bottom:3px solid #e94560;padding-bottom:4px;margin-top:32px}
th{background:var(--th);color:#fff;padding:6px 8px;font-size:11px}
td{border:1px solid var(--line);padding:4px 8px;font-size:12px}
table{border-collapse:collapse;margin:8px 0}
img{background:#fcfcfb;border-radius:6px;margin:12px 0}
.warn{background:#eda10022;border-left:4px solid #eda100;padding:8px 12px}
"""


def write_html_report(
    df: pd.DataFrame, thresholds: list[float], n: int, cols: list[str], out_path: Path, title: str,
    score_col: str = "total_score", extra_html: str = "",
) -> Path:
    scored = df[df[score_col].notna()]
    samples = marginal_samples(df, thresholds, n, cols, score_col)
    vmin, vmax = min(thresholds), max(thresholds)
    warns = data_quality_warnings(df, score_col)

    parts = [
        f"<html><head><meta charset='utf-8'><title>{title}</title><style>{_CSS}</style></head><body>",
        f"<h1>{title}</h1>",
        f"<p>Total filas: {len(df):,} &nbsp;|&nbsp; con score: {len(scored):,} &nbsp;|&nbsp; sin score: {len(df) - len(scored):,}</p>",
        *[f"<p class='warn'>⚠️ {w}</p>" for w in warns],
        "<pre>" + scored[score_col].describe().to_string() + "</pre>",
        extra_html,
        "<h2>Registros marginales por threshold</h2>",
        summary_table(df, thresholds, score_col).to_html(index=False, float_format="{:.1f}".format),
        "<p>Cada bloque muestra los registros <b>más marginales</b> que aún superan el threshold. "
        "Si parecen el mismo lugar, el threshold es seguro; si no, sube un escalón.</p>",
    ]
    for t, sample in samples.items():
        n_q = int((scored[score_col] >= t).sum())
        parts.append(f"<h3>Threshold &ge; {t:.2f} — {n_q:,} registros califican</h3>")
        parts.append(_style(sample, vmin, vmax).to_html() if len(sample) else "<p><i>Sin registros</i></p>")
    parts.append("</body></html>")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(parts), encoding="utf-8")
    print(f"Reporte HTML: {out_path}")
    return out_path


def apply_thresholds(df: pd.DataFrame, match_thr: float, super_thr: float, score_col: str = "total_score") -> pd.DataFrame:
    """match_no_match y super_match a partir del score (NaN cuenta como no match)."""
    out = df.copy()
    score = out[score_col]
    out["match_no_match"] = np.where(score >= match_thr, "match", "no_match")
    out["super_match"] = score >= super_thr  # NaN >= x -> False
    out["match_source"] = np.where(out["match_no_match"] == "match", "score", pd.NA)
    return out
