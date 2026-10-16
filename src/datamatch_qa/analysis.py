"""Gráficos y escenarios para elegir el threshold.

- Distribución del score (histograma + KDE) y valles como cortes naturales
- Curva de sensibilidad (matches vs threshold) y codo (kneedle)
- Scores componentes por bucket
- Muestra estratificada por bucket para etiquetar a mano
- Precisión estimada por threshold a partir de las etiquetas
- Tabla de escenarios con todos los candidatos
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.signal import find_peaks  # noqa: E402
from scipy.stats import gaussian_kde  # noqa: E402

COMPONENTS = ["distance_score", "name_score", "address_score", "phonetic_score"]

# Paleta categórica validada (slots 1-4) + tinta de texto/grilla
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

plt.rcParams.update({
    "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "legend.frameon": False,
})


# ── Cálculos ────────────────────────────────────────────────────────────────

def score_grid(scores: pd.Series, step: float = 0.01) -> np.ndarray:
    lo = max(0.0, np.floor(scores.min() / step) * step)
    hi = min(1.0, np.ceil(scores.max() / step) * step)
    return np.round(np.arange(lo, hi + step / 2, step), 4)


def kde_valleys(scores: pd.Series, grid: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """Densidad y mínimos locales (cortes naturales entre poblaciones)."""
    if scores.nunique() < 3:
        return np.zeros_like(grid), []
    density = gaussian_kde(scores)(grid)
    idx, _ = find_peaks(-density, prominence=density.max() * 0.05)
    return density, [float(grid[i]) for i in idx]


def sensitivity_curve(scores: pd.Series, grid: np.ndarray, n_total: int) -> pd.DataFrame:
    s = np.sort(scores.to_numpy())
    n_match = len(s) - np.searchsorted(s, grid, side="left")
    curve = pd.DataFrame({"threshold": grid, "n_match": n_match})
    curve["pct_total"] = curve["n_match"] / n_total * 100
    curve["lost_per_step"] = -curve["n_match"].diff().fillna(0)  # matches que se pierden al subir 1 paso
    return curve


def find_elbow(curve: pd.DataFrame) -> float | None:
    """Kneedle: punto de la curva más alejado de la recta entre extremos."""
    if len(curve) < 3:
        return None
    x = curve["threshold"].to_numpy(dtype=float)
    y = curve["n_match"].to_numpy(dtype=float)
    xn = (x - x.min()) / (np.ptp(x) or 1)
    yn = (y - y.min()) / (np.ptp(y) or 1)
    dist = np.abs(yn - (1 - xn))  # recta de (0,1) a (1,0)
    return float(x[int(np.argmax(dist))])


def bucket_of(scores: pd.Series, width: float) -> pd.Series:
    """Límite inferior del bucket (0.70, 0.75, ...)."""
    return (np.floor(scores / width + 1e-9) * width).round(4)


def stratified_sample(
    scored: pd.DataFrame, width: float, n: int, seed: int, cols: list[str], score_col: str
) -> pd.DataFrame:
    df = scored.assign(bucket=bucket_of(scored[score_col], width))
    sample = (
        df.sample(frac=1, random_state=seed)  # barajar y tomar n por bucket
        .groupby("bucket").head(n)
        .sort_values(["bucket", score_col])
    )
    keep = ["dataplor_id", "bucket"] + [c for c in cols if c in sample.columns and c != "dataplor_id"]
    return sample[keep].assign(is_match="").reset_index(drop=True)


def load_labels(path: Path) -> pd.DataFrame | None:
    if not path or not Path(path).exists():
        return None
    lab = pd.read_csv(path, encoding="utf-8-sig")
    yes = {"1", "true", "si", "sí", "yes", "y", "s", "match"}
    no = {"0", "false", "no", "n", "no_match"}
    v = lab["is_match"].astype(str).str.strip().str.lower()
    lab["is_match"] = np.where(v.isin(yes), 1.0, np.where(v.isin(no), 0.0, np.nan))
    lab = lab.dropna(subset=["is_match"])
    return lab[["dataplor_id", "is_match"]] if len(lab) else None


def precision_curve(
    scored: pd.DataFrame, labels: pd.DataFrame, grid: np.ndarray, width: float, score_col: str
) -> pd.DataFrame:
    """Precisión estimada para cada threshold = promedio de precisión por bucket
    ponderado por cuántos registros de cada bucket superan el threshold."""
    df = scored[["dataplor_id", score_col]].assign(bucket=bucket_of(scored[score_col], width))
    lab = labels.merge(df, on="dataplor_id", how="inner")
    p_bucket = lab.groupby("bucket")["is_match"].agg(["mean", "size"])

    rows = []
    for t in grid:
        above = df[df[score_col] >= t]
        counts = above["bucket"].value_counts()
        known = counts[counts.index.isin(p_bucket.index)]
        prec = (known * p_bucket.loc[known.index, "mean"]).sum() / known.sum() if known.sum() else np.nan
        rows.append({
            "threshold": t, "precision": prec,
            "coverage": known.sum() / len(above) if len(above) else np.nan,
            "n_labeled": int(lab[lab[score_col] >= t].shape[0]),
        })
    return pd.DataFrame(rows)


def lowest_threshold_with_precision(prec: pd.DataFrame, target: float) -> float | None:
    """Menor threshold desde el cual la precisión se mantiene >= target hacia arriba."""
    ok = prec.dropna(subset=["precision"]).sort_values("threshold", ascending=False)
    best = None
    for _, r in ok.iterrows():
        if r["precision"] < target:
            break
        best = float(r["threshold"])
    return best


def scenarios_table(
    scored: pd.DataFrame, n_total: int, candidates: list[tuple[str, float, str]],
    prec: pd.DataFrame | None, current_match: float, score_col: str,
) -> pd.DataFrame:
    s = scored[score_col]
    base = int((s >= current_match).sum())
    rows = []
    for name, t, why in candidates:
        if t is None:
            continue
        n = int((s >= t).sum())
        row = {
            "escenario": name, "threshold": round(t, 2), "n_match": n,
            "% total": n / n_total * 100, "% con score": n / len(s) * 100,
            "Δ vs actual": n - base, "criterio": why,
        }
        if prec is not None:
            p = prec.loc[(prec["threshold"] - t).abs().idxmin(), "precision"]
            row["precisión est."] = p
            row["falsos est."] = round(n * (1 - p)) if pd.notna(p) else np.nan
        rows.append(row)
    return pd.DataFrame(rows).drop_duplicates(["escenario", "threshold"]).sort_values("threshold", ascending=False)


# ── Gráficos ────────────────────────────────────────────────────────────────

def _vlines(ax, marks: dict[str, tuple[float, str]]):
    """Líneas verticales de thresholds con su etiqueta encima del área de datos."""
    for label, (x, color) in marks.items():
        if x is None:
            continue
        ax.axvline(x, color=color, lw=1.5, ls="--")
        ax.annotate(f"{label} {x:.2f}", (x, 1), xycoords=("data", "axes fraction"), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=8, color=INK)


def plot_distribution(scores, grid, density, valleys, marks):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    bins = np.arange(grid.min(), grid.max() + 0.0101, 0.01)
    ax.hist(scores, bins=bins, density=True, color=BLUE, alpha=0.55, edgecolor="#fcfcfb", linewidth=1)
    ax.plot(grid, density, color=BLUE, lw=2, label="densidad (KDE)")
    for v in valleys:
        ax.axvline(v, color=MUTED, lw=1, ls=":")
        ax.annotate(f"valle {v:.2f}", (v, 0.97), xycoords=("data", "axes fraction"), xytext=(4, 0),
                    textcoords="offset points", va="top", fontsize=8, color=MUTED)
    _vlines(ax, marks)
    ax.set_title("Distribución de total_score (valles punteados = cortes naturales)", pad=18)
    ax.set(
           xlabel="total_score", ylabel="densidad")
    ax.legend(loc="upper right")
    return fig


def plot_sensitivity(curve, elbow, marks):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(9, 5.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    a1.plot(curve["threshold"], curve["n_match"], color=BLUE, lw=2)
    if elbow is not None:
        y = curve.loc[(curve["threshold"] - elbow).abs().idxmin(), "n_match"]
        a1.plot([elbow], [y], "o", ms=8, color=BLUE, mec="#fcfcfb", mew=2)
        a1.annotate(f"codo {elbow:.2f}", (elbow, y), xytext=(8, 8), textcoords="offset points")
    _vlines(a1, marks)
    a1.set_title("Curva de sensibilidad: matches que quedan con cada threshold", pad=18)
    a1.set(ylabel="n matches")
    a2.bar(curve["threshold"], curve["lost_per_step"], width=0.008, color=BLUE)
    a2.set(xlabel="threshold", ylabel="perdidos por +0.01")
    fig.tight_layout()
    return fig


def plot_components(scored, width, score_col):
    cols = [c for c in COMPONENTS if c in scored.columns]
    if not cols:
        return None
    g = scored.assign(bucket=bucket_of(scored[score_col], width)).groupby("bucket")[cols].mean()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for c, color in zip(cols, (BLUE, ORANGE, AQUA, YELLOW)):
        ax.plot(g.index, g[c], color=color, lw=2, marker="o", ms=5, label=c)
    ax.set(title="Promedio de cada score componente por bucket de total_score",
           xlabel="bucket de total_score (límite inferior)", ylabel="score medio", ylim=(0, 1.02))
    ax.legend(ncol=4, loc="lower right")
    return fig


def plot_precision(prec, targets, marks):
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(prec["threshold"], prec["precision"], color=BLUE, lw=2, label="precisión estimada")
    for label, t in targets.items():
        ax.axhline(t, color=MUTED, lw=1, ls=":")
        ax.text(prec["threshold"].max(), t, f"objetivo {label} {t:.0%}", ha="right", va="bottom", fontsize=8, color=MUTED)
    ax.set_title("Precisión estimada (etiquetas manuales) vs threshold", pad=18)
    ax.set(xlabel="threshold",
           ylabel="precisión", ylim=(max(0, prec["precision"].min() - 0.05), 1.01))
    _vlines(ax, marks)
    ax.legend(loc="lower right")
    return fig


def fig_to_html(fig, out_png: Path) -> str:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return f"<img style='max-width:100%' src='data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}'>"


# ── Orquestación ────────────────────────────────────────────────────────────

def analyze(df: pd.DataFrame, cfg, score_col: str) -> str:
    """Genera gráficos, escenarios y muestra de revisión. Devuelve el bloque HTML."""
    an, thr = cfg["analysis"], cfg["thresholds"]
    out_dir = cfg.path("output_dir")
    charts = out_dir / f"charts_{cfg.tag}"
    scored = df[df[score_col].notna()]
    if scored.empty:
        return "<p><b>No hay registros con score.</b></p>"
    s = scored[score_col]

    grid = score_grid(s)
    density, valleys = kde_valleys(s, grid)
    curve = sensitivity_curve(s, grid, len(df))
    elbow = find_elbow(curve)

    labels = load_labels(cfg.path_or_none("labels"))
    prec = rec_match = rec_super = None
    if labels is not None:
        prec = precision_curve(scored, labels, grid, an["bucket_width"], score_col)
        rec_match = lowest_threshold_with_precision(prec, an["target_precision_match"])
        rec_super = lowest_threshold_with_precision(prec, an["target_precision_super"])

    candidates = [
        ("Actual match (config)", thr["match"], "thresholds.match"),
        ("Actual super_match (config)", thr["super_match"], "thresholds.super_match"),
        ("Codo de la curva", elbow, "máxima curvatura matches vs threshold"),
        *[(f"Valle KDE #{i + 1}", v, "mínimo de densidad entre poblaciones") for i, v in enumerate(valleys)],
        ("Precisión ≥ objetivo match", rec_match, f"precisión ≥ {an['target_precision_match']:.0%} (etiquetas)"),
        ("Precisión ≥ objetivo super", rec_super, f"precisión ≥ {an['target_precision_super']:.0%} (etiquetas)"),
    ]
    scen = scenarios_table(scored, len(df), candidates, prec, thr["match"], score_col)

    marks = {"match": (thr["match"], ORANGE), "super": (thr["super_match"], AQUA)}
    html = ["<h2>Escenarios para elegir el threshold</h2>",
            scen.to_html(index=False, float_format="{:.2f}".format, na_rep="—")]

    html.append(fig_to_html(plot_distribution(s, grid, density, valleys, marks), charts / "1_distribucion.png"))
    html.append(fig_to_html(plot_sensitivity(curve, elbow, marks), charts / "2_sensibilidad.png"))
    fig = plot_components(scored, an["bucket_width"], score_col)
    if fig is not None:
        html.append(fig_to_html(fig, charts / "3_componentes.png"))
    if prec is not None:
        targets = {"match": an["target_precision_match"], "super": an["target_precision_super"]}
        html.append(fig_to_html(plot_precision(prec, targets, marks), charts / "4_precision.png"))
    else:
        html.append("<p><i>Sin etiquetas manuales: la precisión no se estima. Etiqueta la muestra "
                    "<code>review_sample_*.csv</code> (columna <code>is_match</code> = 1/0) y apunta "
                    "<code>analysis.labels</code> a ese archivo.</i></p>")

    sample = stratified_sample(scored, an["bucket_width"], an["sample_per_bucket"], an["seed"],
                               cfg["columns"]["display"], score_col)
    sample_path = out_dir / f"review_sample_{cfg.tag}.csv"
    sample.to_csv(sample_path, index=False, encoding="utf-8-sig")

    print("\nEscenarios:")
    print(scen.to_string(index=False, float_format=lambda x: f"{x:.2f}"))
    print(f"\nGráficos en: {charts}")
    print(f"Muestra para etiquetar ({len(sample):,} filas): {sample_path}")
    return "\n".join(html)
