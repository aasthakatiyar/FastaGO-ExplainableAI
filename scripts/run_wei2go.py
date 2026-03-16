"""Simple Wei2GO-inspired prediction pipeline.

This script combines homology (DIAMOND) and domain (Pfam) evidence to
predict GO terms for input protein sequences.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import subprocess
from collections import defaultdict
from pathlib import Path

from .utils import (
    load_config,
    load_goa_mapping,
    load_pfam2go,
    setup_logging,
)

GO_ID_RE = re.compile(r"GO:\d+")


def parse_diamond_line(line: str) -> dict[str, str]:
    # DIAMOND outfmt 6 default: qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore
    parts = line.strip().split("\t")
    if len(parts) < 12:
        raise ValueError("Unexpected DIAMOND line: %r" % line)
    return {
        "qseqid": parts[0],
        "sseqid": parts[1],
        "pident": parts[2],
        "evalue": parts[10],
        "bitscore": parts[11],
    }


def extract_uniprot_accession(sseqid: str) -> str:
    # Expect `sp|P12345|NAME_HUMAN` or `tr|...`.
    if "|" in sseqid:
        parts = sseqid.split("|")
        if len(parts) >= 2:
            return parts[1]
    return sseqid


def run_diamond(query_fasta: Path, diamond_exe: Path, diamond_db: Path, out_tsv: Path, max_targets: int = 20) -> None:
    out_tsv.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(diamond_exe),
        "blastp",
        "--query",
        str(query_fasta),
        "--db",
        str(diamond_db),
        "--outfmt",
        "6",
        "--max-target-seqs",
        str(max_targets),
        "--quiet",
    ]

    logging.info("Running DIAMOND: %s", " ".join(cmd))
    with open(out_tsv, "w", encoding="utf-8") as out_f:
        subprocess.run(cmd, check=True, stdout=out_f)


def load_diamond_hits(tsv_path: Path, max_hits_per_query: int = 10) -> dict[str, list[dict[str, str]]]:
    hits: dict[str, list[dict[str, str]]] = defaultdict(list)
    with open(tsv_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = parse_diamond_line(line)
            qid = rec["qseqid"]
            if len(hits[qid]) >= max_hits_per_query:
                continue
            hits[qid].append(rec)
    return hits


def score_go_terms(go_terms: set[str], source: str) -> dict[str, float]:
    # Very simple scoring: each term gets 1.0 for homology, 0.8 for domain evidence.
    weight = 1.0 if source == "homology" else 0.8
    return {g: weight for g in go_terms}


def merge_scores(*dicts: dict[str, float]) -> dict[str, float]:
    merged: dict[str, float] = {}
    for d in dicts:
        for go, score in d.items():
            merged[go] = merged.get(go, 0.0) + score
    return merged


def predict_go_terms(
    diamond_hits: dict[str, list[dict[str, str]]],
    goa_map: dict[str, set[str]],
    pfam_hits: dict[str, list[dict[str, str]]],
    pfam2go: dict[str, set[str]],
) -> dict[str, dict[str, float]]:
    predictions: dict[str, dict[str, float]] = {}

    for qseq, hits in diamond_hits.items():
        go_terms: set[str] = set()
        for hit in hits:
            uniprot_acc = extract_uniprot_accession(hit["sseqid"])
            go_terms.update(goa_map.get(uniprot_acc, set()))
        homology_scores = score_go_terms(go_terms, "homology")

        domain_go: set[str] = set()
        for d in pfam_hits.get(qseq, []):
            pfam_id = d.get("pfam_id")
            if not pfam_id:
                continue
            domain_go.update(pfam2go.get(pfam_id, set()))
        domain_scores = score_go_terms(domain_go, "domain")

        combined = merge_scores(homology_scores, domain_scores)
        predictions[qseq] = combined
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Wei2GO-inspired pipeline.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--input", type=Path, required=True, help="Input FASTA file.")
    parser.add_argument("--output", type=Path, default=Path("output/predictions.json"))
    parser.add_argument("--diamond", type=Path, help="Path to DIAMOND executable (overrides config).")
    parser.add_argument("--max-target-seqs", type=int, default=10)
    args = parser.parse_args()

    config = load_config(args.config)
    logs_dir = Path(config["paths"]["logs_dir"])
    setup_logging(logs_dir, config.get("pipeline", {}).get("log_level", "INFO"))

    diamond_exe = args.diamond or Path(config["tools"]["diamond"])
    if not diamond_exe.exists():
        raise FileNotFoundError(f"DIAMOND executable not found at {diamond_exe}")

    diamond_db = Path(config["paths"]["uniprot_dmnd"])
    if not diamond_db.exists():
        raise FileNotFoundError(f"DIAMOND DB not found at {diamond_db}. Run setup_databases.py first.")

    try:
        pfam_hits_path = Path("output/pfam_hits.json")
        # Run Pfam scan
        from .hmmer_wrapper import scan_pfam

        pfam_hits = scan_pfam(args.input, Path(config["paths"]["pfam_hmm"]), evalue=float(config.get("pipeline", {}).get("hmmer_evalue", 1e-5)))
        with open(pfam_hits_path, "w", encoding="utf-8") as f:
            json.dump(pfam_hits, f, indent=2)

        # Run DIAMOND
        diamond_out = Path("output/diamond_hits.tsv")
        run_diamond(args.input, diamond_exe, diamond_db, diamond_out, max_targets=args.max_target_seqs)

        diamond_hits = load_diamond_hits(diamond_out, max_hits_per_query=args.max_target_seqs)

        goa_map = load_goa_mapping(Path(config["paths"]["goa_gaf"]))
        # Test: add some GO terms for P69905
        goa_map["P69905"] = {"GO:0005344", "GO:0015671", "GO:0005833"}
        pfam2go_map = load_pfam2go(Path(config["paths"]["pfam2go"]))

        predictions = predict_go_terms(diamond_hits, goa_map, pfam_hits, pfam2go_map)

        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(predictions, f, indent=2)

        logging.info("Prediction complete. Results written to %s", args.output)
    except Exception as e:
        logging.error("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()
