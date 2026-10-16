"""Punto de entrada del QA de Datamatch.

Uso:
    uv run main.py explore            # reporte HTML para elegir thresholds
    uv run main.py run                # datamatch final + exportes
    uv run main.py run -c config/otro.yaml --match 0.75 --super-match 0.85
"""

from datamatch_qa.cli import main

if __name__ == "__main__":
    main()
