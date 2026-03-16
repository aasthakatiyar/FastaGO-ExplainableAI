"""Wrapper around PyHMMER to scan sequences against Pfam HMMs."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pyhmmer
from pyhmmer import easel

from .utils import load_config, setup_logging


def scan_pfam(seqs_fasta: Path, pfam_hmm: Path, evalue: float = 1e-5) -> dict[str, list[dict[str, str]]]:
    """Scan sequences against Pfam and return domain hits."""
    logging.info("Scanning %s against Pfam HMMs", seqs_fasta)
    hits_by_seq: dict[str, list[dict[str, str]]] = {}

    try:
        with easel.SequenceFile(str(seqs_fasta), digital=True) as seqfile:
            sequences = list(seqfile)
        logging.info("Loaded %d sequences", len(sequences))

        with pyhmmer.plan7.HMMFile(str(pfam_hmm)) as hmms:
            hmm_list = list(hmms)[:100]  # Limit to first 100 for testing
            logging.info("Loaded %d HMMs (limited to 100 for testing)", len(hmm_list))
            for results in pyhmmer.hmmscan(sequences, hmm_list):
                # results corresponds to one input sequence
                seq_name = results.query.name
                hits: list[dict[str, str]] = []
                for hit in results:
                    if hit.evalue <= evalue:
                        hits.append(
                            {
                                "pfam_id": hit.name,
                                "evalue": f"{hit.evalue:.2e}",
                                "score": f"{hit.score:.2f}",
                            }
                        )
                hits_by_seq[seq_name] = hits
                logging.info("Processed sequence %s, found %d hits", seq_name, len(hits))
        logging.info("Pfam scan completed")
    except Exception as e:
        logging.error("Error in Pfam scan: %s", e)
        raise
    return hits_by_seq


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan FASTA sequences against Pfam using PyHMMER.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="Input FASTA file.")
    parser.add_argument("--output", type=Path, default=Path("output/pfam_hits.json"))
    parser.add_argument("--evalue", type=float, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    logs_dir = Path(config["paths"]["logs_dir"])
    setup_logging(logs_dir, config.get("pipeline", {}).get("log_level", "INFO"))

    pfam_hmm = Path(config["paths"]["pfam_hmm"])
    if not pfam_hmm.exists():
        raise FileNotFoundError(f"Pfam HMM file not found at {pfam_hmm}. Run download_databases.py first.")

    evalue = args.evalue if args.evalue is not None else config.get("pipeline", {}).get("hmmer_evalue", 1e-5)
    hits = scan_pfam(args.input, pfam_hmm, evalue=evalue)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    import json

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(hits, f, indent=2)

    logging.info("Pfam scan complete. Results written to %s", args.output)


if __name__ == "__main__":
    main()
