"""Sanity checks for FastaGO-ExplainableAI environment."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from .utils import load_config, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a quick environment health check.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    args = parser.parse_args()

    config = load_config(args.config)
    logs_dir = Path(config["paths"]["logs_dir"])
    setup_logging(logs_dir, config.get("pipeline", {}).get("log_level", "INFO"))

    errors = []

    def check_path(name: str, path: Path, required: bool = True):
        if not path.exists():
            msg = f"Missing {name}: {path}"
            logging.error(msg)
            if required:
                errors.append(msg)
        else:
            logging.info("Found %s: %s", name, path)

    check_path("config", args.config)
    check_path("Swiss-Prot FASTA", Path(config["paths"]["uniprot_fasta"]))
    check_path("DIAMOND DB", Path(config["paths"]["uniprot_dmnd"]))
    check_path("Pfam HMM", Path(config["paths"]["pfam_hmm"]))
    check_path("Filtered GOA GAF", Path(config["paths"]["goa_gaf"]))
    check_path("pfam2go", Path(config["paths"]["pfam2go"]))

    diamond = Path(config["tools"]["diamond"])
    check_path("DIAMOND executable", diamond)

    try:
        import pyhmmer  # noqa: F401
        logging.info("pyhmmer available")
    except ImportError:
        msg = "pyhmmer is not installed. Install via `pip install pyhmmer`"
        logging.error(msg)
        errors.append(msg)

    if errors:
        logging.error("Environment check failed with %d issue(s).", len(errors))
        sys.exit(1)

    logging.info("Environment looks good. You are ready to run the pipeline.")


if __name__ == "__main__":
    main()
