# -*- coding: utf-8 -*-

"""CASDET entry point.

Usage:
    python main.py --task {main,intervention,ablation,all} [options]"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from CASDET.experiments import run_experiment  # noqa: E402
from CASDET.utils import load_config, setup_logging  # noqa: E402

DEFAULT_CONFIGS = {
    "main": "log/configs/main.yaml",
    "intervention": "log/configs/intervention.yaml",
    "ablation": "log/configs/ablation.yaml",
}

ALL_TASKS = ["main", "intervention", "ablation"]

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detecting RAG hallucinations via cascade influence from "
                    "attention to feedforward networks")
    parser.add_argument(
        "--task", "-t", required=True,
        choices=list(DEFAULT_CONFIGS) + ["all"],
        help="main=main experiments / intervention=graded intervention / "
             "ablation=ablation studies / all=run all three in order")
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to a YAML configuration file "
             "(defaults to the file under log/configs/ for the selected task).")
    parser.add_argument(
        "--model", default=None,
        help="Run only the given model (e.g. llama2_7b). Useful for running "
             "different models in parallel on separate GPUs.")
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable verbose logging.")
    parser.add_argument(
        "--log-dir", default="log",
        help="Directory for the run log (default: log).")
    return parser.parse_args()

def run_one(task: str, config_path: Path,
            model: str | None, logger: logging.Logger) -> bool:
    cfg = load_config(config_path)
    if model:
        models = cfg.get("models", {})
        if model not in models:
            logger.error("Model %s is not in the configuration; available: %s",
                         model, sorted(models))
            return False
        cfg["_all_models"] = dict(models)
        cfg["models"] = {model: models[model]}
    logger.info("Task [%s] using configuration: %s", task, config_path)
    try:
        result = run_experiment(task, cfg)
        n_failed = 0
        if isinstance(result, dict):
            n_failed = (int(result.get("n_failed", 0))
                        + int(result.get("n_missing_own", 0)))
        if n_failed:
            logger.error("Task [%s] finished with %d failed combinations; "
                         "see the exception log above.", task, n_failed)
            return False
        logger.info("Task [%s] completed.", task)
        return True
    except Exception:  # noqa: BLE001 - log the full exception
        logger.exception("Task [%s] failed", task)
        return False

def main() -> None:
    args = parse_args()
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(log_dir / "run.log",
                           logging.DEBUG if args.debug else logging.INFO)

    tasks = ALL_TASKS if args.task == "all" else [args.task]
    ok = True
    for task in tasks:
        config_path = Path(args.config) if args.config else Path(
            DEFAULT_CONFIGS[task])
        if not config_path.exists():
            logger.error("Configuration file not found: %s", config_path)
            ok = False
            continue
        ok = run_one(task, config_path, args.model, logger) and ok
    logger.info("All tasks finished: %s",
                "success" if ok else "failures detected, see the log")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
