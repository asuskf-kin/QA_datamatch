"""CLI: `datamatch-qa explore` y `datamatch-qa run`."""

from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG, load_config
from .pipeline import explore, run


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="datamatch-qa", description=__doc__)
    parser.add_argument("command", choices=["explore", "run"],
                        help="explore = reporte HTML de thresholds; run = datamatch final + exportes")
    parser.add_argument("-c", "--config", default=DEFAULT_CONFIG, help="ruta a config.yaml")
    parser.add_argument("--match", type=float, help="override de thresholds.match")
    parser.add_argument("--super-match", type=float, help="override de thresholds.super_match")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.match is not None:
        cfg["thresholds"]["match"] = args.match
    if args.super_match is not None:
        cfg["thresholds"]["super_match"] = args.super_match

    {"explore": explore, "run": run}[args.command](cfg)


if __name__ == "__main__":
    main()
