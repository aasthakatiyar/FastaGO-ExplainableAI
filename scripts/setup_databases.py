"""Prepare databases for use with FastaGO-ExplainableAI.

- Builds DIAMOND database from Swiss-Prot FASTA
- Validates required files are present
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
from pathlib import Path

from .utils import load_config, setup_logging


def run_diamond_makedb(diamond_exe: Path, fasta: Path, out_db: Path) -> None:
    out_db.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(diamond_exe), "makedb", "--in", str(fasta), "-d", str(out_db)]
    logging.info("Running DIAMOND makedb: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare database files for FastaGO-ExplainableAI.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--force", action="store_true", help="Rebuild DIAMOND DB even if it exists.")
    args = parser.parse_args()

    config = load_config(args.config)
    logs_dir = Path(config["paths"]["logs_dir"])
    setup_logging(logs_dir, config.get("pipeline", {}).get("log_level", "INFO"))

    diamond_exe = Path(config["tools"]["diamond"])
    if not diamond_exe.exists():
        raise FileNotFoundError(
            f"DIAMOND binary not found at {diamond_exe}. Place diamond.exe into the tools/ directory or update config."
        )

    fasta = Path(config["paths"]["uniprot_fasta"])
    if not fasta.exists():
        raise FileNotFoundError(f"Swiss-Prot FASTA not found at {fasta}. Run download_databases.py first.")

    dmnd = Path(config["paths"]["uniprot_dmnd"])
    if args.force or not dmnd.exists():
        run_diamond_makedb(diamond_exe, fasta, dmnd)
    else:
        logging.info("DIAMOND DB already exists at %s (use --force to rebuild).", dmnd)

    # Validate other required files
    pfam = Path(config["paths"]["pfam_hmm"])
    if not pfam.exists():
        raise FileNotFoundError(f"Pfam HMM file not found at {pfam}. Run download_databases.py first.")

    goa = Path(config["paths"]["goa_gaf"])
    if not goa.exists():
        raise FileNotFoundError(f"Filtered GOA GAF not found at {goa}. Run download_databases.py first.")

    logging.info("Database setup complete.")


if __name__ == "__main__":
    main()
