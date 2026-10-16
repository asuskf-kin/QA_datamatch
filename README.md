# QA Datamatch (Dataplor vs POS)

Reemplaza los 3 notebooks (`notebooks/`) por un pipeline configurable con `config/config.yaml`.

| Notebook original | Ahora |
|---|---|
| `0_..._Threshold_Analysis` (explorar thresholds, `super_match`) | `datamatch-qa explore` → reporte HTML · `super_match` en `run` |
| `1_... por nombre y distancia` (threshold + rescate por nombre ≤100 m) | `datamatch-qa run` |
| `2_... validation` (split match / no_match) | `datamatch-qa run` (`outputs.write_split`) |

## Estructura

```
config/config.yaml        # rutas, thresholds, columnas, salidas
src/datamatch_qa/
  config.py               # carga/valida el YAML
  io.py                   # leer csv/xlsx/parquet, escribir csv
  utils.py                # normalize_name, haversine vectorizado
  thresholds.py           # explorer HTML + flags match/super_match
  analysis.py             # gráficos, escenarios, muestra para etiquetar, precisión
  name_match.py           # rescate de no_match por nombre + distancia
  pipeline.py             # explore / run
  cli.py                  # entrypoint `datamatch-qa`
data/raw/                 # entradas (no se versionan)
data/output/              # salidas (no se versionan)
notebooks/                # notebooks originales (referencia)
tests/
```

## Uso

```bash
uv sync
```

1. Copia `final_output.csv` (salida del matcher) y la base de clientes a `data/raw/` y ajusta `paths` y `project.client` en `config/config.yaml`.
2. Explora thresholds y abre el HTML en `data/output/`:

```bash
uv run datamatch-qa explore
```

   El reporte trae:
   - **Tabla de escenarios**: threshold actual, codo de la curva, valles de la KDE y (con etiquetas) el menor threshold que cumple la precisión objetivo. Para cada uno: n matches, % y Δ vs el actual, precisión estimada y falsos positivos estimados.
   - **Gráficos** (también en `charts_<...>/`): distribución + KDE, curva de sensibilidad + codo, scores componentes por bucket, precisión vs threshold.
   - **Avisos de calidad** (p.ej. `address_score` siempre 0).
   - Registros marginales por threshold, para revisarlos a ojo.

   Para estimar la precisión: etiqueta `review_sample_<...>.csv` (muestra estratificada por bucket; columna `is_match` = 1/0), guárdalo en `data/raw/review_labels.csv`, apunta `analysis.labels` a ese archivo y vuelve a correr `explore`.

3. Fija `thresholds.match` y `thresholds.super_match` en el YAML y genera las salidas:

```bash
uv run datamatch-qa run
```

Equivalente con el `main.py` de la raíz: `uv run main.py explore` / `uv run main.py run`.

Override puntual sin editar el YAML: `uv run datamatch-qa run --match 0.75 --super-match 0.85`.  
Otro cliente/país: `uv run datamatch-qa run -c config/otro.yaml`.

## Salidas (`data/output/`)

- `Datamatch_Final_<cliente>_<fecha>.csv` — base completa con `match_no_match`, `super_match`, `match_source` (`score` | `name_distance`).
- `Datamatch_Final_LatLon_<cliente>_<fecha>.csv`
- `match_<...>.csv` / `no_match_<...>.csv`
- `threshold_report_<...>.html` (de `explore`)

## Tests

```bash
uv run pytest
```
