"""Download and prepare databases for FastaGO-ExplainableAI."""

from __future__ import annotations

import argparse
import gzip
import logging
import pickle
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from utils import gunzip_file, load_config, setup_logging, stream_download


def ensure_download(url: str, path: Path, force: bool = False, min_size: int = 1024) -> None:
    """Download a file only if missing, corrupted, or forced."""
    if path.exists() and not force:
        local_size = path.stat().st_size
        if local_size < min_size:
            logging.warning("File exists but is too small, re-downloading: %s", path)
        else:
            try:
                r = requests.head(url, allow_redirects=True, timeout=10)
                r.raise_for_status()
                remote_size = r.headers.get("Content-Length")
                if remote_size is not None:
                    remote_size = int(remote_size)
                    if remote_size == local_size:
                        logging.info("Skipping download (size matches remote): %s", path)
                        return
                    logging.warning(
                        "Local file size (%d) differs from remote (%d); re-downloading %s",
                        local_size,
                        remote_size,
                        path,
                    )
                else:
                    logging.info("Skipping download (exists, size %d): %s", local_size, path)
                    return
            except Exception as e:
                logging.warning(
                    "Could not verify remote file size for %s: %s", url, e
                )
                if local_size >= min_size:
                    logging.info("Skipping download (exists, unverified size %d): %s", local_size, path)
                    return
    stream_download(url, path)


def read_fasta_ids_gz(fasta_gz: Path, cache_file: Path) -> set[str]:
    """Read Swiss-Prot IDs directly from gzipped FASTA, with caching."""
    if cache_file.exists():
        logging.info("Loading cached Swiss-Prot IDs from %s", cache_file)
        with open(cache_file, "rb") as f:
            return pickle.load(f)

    logging.info("Reading Swiss-Prot IDs from gzipped FASTA: %s", fasta_gz)
    ids = set()
    with gzip.open(fasta_gz, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith(">"):
                parts = line.split("|")
                if len(parts) > 1:
                    ids.add(parts[1])

    # Save cache
    with open(cache_file, "wb") as f:
        pickle.dump(ids, f)
    logging.info("Cached %d Swiss-Prot IDs to %s", len(ids), cache_file)
    return ids


def filter_goa_stream_parallel(url: str, goa_gaf_out: Path, swissprot_ids: set[str], max_workers: int = 8, batch_size: int = 1000):
    """
    Stream GOA GAF from URL and filter to Swiss-Prot IDs using multiple threads.
    Writes to gzip in batches to speed up.
    """
    goa_gaf_out.parent.mkdir(parents=True, exist_ok=True)
    logging.info("Starting parallel GOA filtering with %d workers...", max_workers)

    def process_lines(lines):
        """Filter lines for Swiss-Prot IDs."""
        result = []
        for line in lines:
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            line = line.rstrip("\n")
            if line.startswith("!"):
                result.append(line + "\n")
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            if parts[1] in swissprot_ids:
                result.append(line + "\n")
        return result

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        buffer = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for line in r.iter_lines(decode_unicode=False):
                buffer.append(line)
                if len(buffer) >= batch_size:
                    futures.append(executor.submit(process_lines, buffer))
                    buffer = []
            # Submit remaining lines
            if buffer:
                futures.append(executor.submit(process_lines, buffer))

            # Collect all results
            all_filtered = []
            for future in as_completed(futures):
                all_filtered.extend(future.result())

        # Write sequentially
        with gzip.open(goa_gaf_out, "wt", encoding="utf-8") as fout:
            for line in all_filtered:
                fout.write(line)

    logging.info("Parallel GOA filtering completed: %s", goa_gaf_out)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download required databases for FastaGO-ExplainableAI.")
    parser.add_argument("--config", type=Path, default=Path("configs/config.yaml"))
    parser.add_argument("--force", action="store_true", help="Re-download existing files.")
    parser.add_argument("--skip-filter", action="store_true", help="Skip filtering GOA to Swiss-Prot IDs.")
    args = parser.parse_args()

    config = load_config(args.config)
    logs_dir = Path(config["paths"]["logs_dir"])
    setup_logging(logs_dir, config.get("pipeline", {}).get("log_level", "INFO"))

    urls = config["urls"]
    paths = config["paths"]

    # Swiss-Prot FASTA
    spr_fasta_gz = Path(paths["uniprot_fasta"]).with_suffix(".fasta.gz")
    ensure_download(urls["uniprot_sprot_fasta"], spr_fasta_gz, force=args.force, min_size=1024 * 10)

    # Cache file for Swiss-Prot IDs
    cache_file = spr_fasta_gz.with_suffix(".ids.pkl")

    # Pfam HMM
    pfam_hmm_gz = Path(paths["pfam_hmm"]).with_suffix(".hmm.gz")
    pfam_hmm = Path(paths["pfam_hmm"])
    ensure_download(urls["pfam_a_hmm"], pfam_hmm_gz, force=args.force, min_size=1024 * 100)
    if args.force or not pfam_hmm.exists():
        logging.info("Unzipping Pfam-A HMM to %s", pfam_hmm)
        gunzip_file(pfam_hmm_gz, pfam_hmm)

    # GO OBO
    go_obo = Path(paths["go_obo"])
    ensure_download(urls["go_obo"], go_obo, force=args.force, min_size=1024 * 10)

    # pfam2go
    pfam2go = Path(paths["pfam2go"])
    ensure_download(urls["pfam2go"], pfam2go, force=args.force, min_size=1024)

    # GOA UniProt (filtered)
    goa_filtered_gz = Path(paths["goa_gaf"])
    if args.force or not goa_filtered_gz.exists() or goa_filtered_gz.stat().st_size < 1024:
        if args.skip_filter:
            logging.info("Downloading full GOA UniProt GAF to %s", goa_filtered_gz)
            stream_download(urls["goa_uniprot"], goa_filtered_gz)
        else:
            swissprot_ids = read_fasta_ids_gz(spr_fasta_gz, cache_file)
            logging.info("Streaming GOA UniProt GAF and filtering to Swiss-Prot IDs in parallel")
            filter_goa_stream_parallel(urls["goa_uniprot"], goa_filtered_gz, swissprot_ids)
            logging.info("Filtered GOA output: %s", goa_filtered_gz)
    else:
        logging.info("Skipping GOA filtering (already present): %s", goa_filtered_gz)

    logging.info("Download step completed. Databases are ready.")


if __name__ == "__main__":
    main()