import numpy as np
import pandas as pd

from datamatch_qa.name_match import apply_name_matches, find_name_matches
from datamatch_qa.thresholds import apply_thresholds, threshold_grid
from datamatch_qa.utils import haversine_m, normalize_name

CLIENT_COLS = {"id": "pos_id", "name": "pos_name", "latitude": "pos_latitude", "longitude": "pos_longitude"}


def test_normalize_name():
    assert normalize_name("  Vig’s  Pollo! ") == "vig s pollo"
    assert normalize_name("Cremería Blanquita") == "cremeria blanquita"
    assert normalize_name(None) == ""


def test_haversine_known_distance():
    # 1 grado de latitud ≈ 111.2 km
    d = haversine_m([0.0], [0.0], [1.0], [0.0])[0]
    assert abs(d - 111_195) < 50


def test_threshold_grid():
    assert threshold_grid(0.95, 0.40, 0.05) == [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.4]


def test_apply_thresholds_nan_is_no_match():
    df = pd.DataFrame({"total_score": [0.9, 0.75, 0.5, np.nan]})
    out = apply_thresholds(df, 0.71, 0.80)
    assert out["match_no_match"].tolist() == ["match", "match", "no_match", "no_match"]
    assert out["super_match"].tolist() == [True, False, False, False]


def test_name_match_updates_full_pos_record():
    df = pd.DataFrame({
        "dataplor_id": ["d1", "d2", "d3"],
        "dataplor_name": ["Abarrotes Camila", "Tienda Lejos", "Otra"],
        "dataplor_latitude": [19.0, 19.0, 19.0],
        "dataplor_longitude": [-96.0, -96.0, -96.0],
        "pos_id": ["old", "old2", None],
        "pos_name": ["X", "Y", None],
        "pos_latitude": [0.0, 0.0, np.nan],
        "pos_longitude": [0.0, 0.0, np.nan],
        "total_score": [np.nan, np.nan, np.nan],
    })
    df = apply_thresholds(df, 0.71, 0.80)
    clients = pd.DataFrame({
        "pos_id": ["p1", "p2"],
        "pos_name": ["ABARROTES CAMILA", "TIENDA LEJOS"],
        "pos_latitude": [19.0003, 19.01],      # ~33 m / ~1.1 km
        "pos_longitude": [-96.0, -96.0],
    })
    m = find_name_matches(df, clients, CLIENT_COLS, max_distance_m=100)
    assert m["dataplor_id"].tolist() == ["d1"]

    out = apply_name_matches(df, m).set_index("dataplor_id")
    assert out.loc["d1", "pos_id"] == "p1"
    assert out.loc["d1", "pos_name"] == "ABARROTES CAMILA"
    assert out.loc["d1", "match_no_match"] == "match"
    assert out.loc["d1", "match_source"] == "name_distance"
    assert out.loc["d2", "match_no_match"] == "no_match"


def test_ampersand_equals_y():
    assert normalize_name("MAR & TIERRA") == normalize_name("Mar y Tierra")


def test_elbow_and_precision():
    from datamatch_qa.analysis import (
        find_elbow, lowest_threshold_with_precision, precision_curve, score_grid, sensitivity_curve,
    )
    rng = np.random.default_rng(1)
    s = pd.Series(np.r_[rng.uniform(0.70, 0.80, 300), rng.uniform(0.90, 1.0, 100)])
    scored = pd.DataFrame({"dataplor_id": [f"d{i}" for i in range(len(s))], "total_score": s})
    grid = score_grid(s)
    elbow = find_elbow(sensitivity_curve(s, grid, len(s)))
    assert 0.75 <= elbow <= 0.92

    # Etiquetas: todo >= 0.85 es match real; no hay registros entre 0.80 y 0.90
    labels = scored.assign(is_match=(s >= 0.85).astype(float))[["dataplor_id", "is_match"]]
    prec = precision_curve(scored, labels, grid, 0.05, "total_score")
    assert 0.80 <= lowest_threshold_with_precision(prec, 0.95) <= 0.90
