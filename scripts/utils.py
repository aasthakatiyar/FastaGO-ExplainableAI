"""Common utilities for FastaGO-ExplainableAI scripts."""

from __future__ import annotations

import gzip
import json
import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import yaml
from rich.logging import RichHandler


def load_config(config_path: Path | str) -> Dict:
    """Load YAML config file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(logs_dir: Path, level: str = "INFO") -> None:
    """Configure logging."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_file = logs_dir / "pipeline.log"

    handlers = [RichHandler(rich_tracebacks=True), logging.FileHandler(log_file, encoding="utf-8")]
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=handlers,
    )


def ensure_dirs(paths: Iterable[Path]) -> None:
    for p in paths:
        p.mkdir(parents=True, exist_ok=True)


def is_command_available(cmd: str) -> bool:
    return shutil.which(cmd) is not None


def gzip_open(path: Path, mode: str = "rt", encoding: str = "utf-8"):
    # Wrapper to handle reading gz and plain text transparently
    if str(path).endswith(".gz"):
        return gzip.open(path, mode=mode, encoding=encoding)
    return open(path, mode=mode, encoding=encoding)


def read_fasta_ids(fasta_path: Path) -> set[str]:
    """Read UniProt accession IDs (sp|<accession>|) from a FASTA file."""
    ids: set[str] = set()
    with gzip_open(fasta_path, "rt") as f:
        for line in f:
            if not line.startswith(">"):
                continue
            # Expect header like: >sp|P12345|NAME_HUMAN ...
            parts = line.strip().split("|")
            if len(parts) >= 2 and parts[0].startswith(">"):
                acc = parts[1]
                ids.add(acc)
    return ids


def stream_download(url: str, dest_path: Path, chunk_size: int = 1024 * 1024) -> None:
    """Download a file with streaming and progress feedback."""
    import requests

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("Content-Length", 0))
        downloaded = 0
        with open(dest_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)
    logging.info("Downloaded %s (%d bytes)", dest_path, downloaded)


def gunzip_file(src: Path, dest: Path) -> None:
    """Uncompress a .gz file to dest. Overwrites if exists."""
    if not src.exists():
        raise FileNotFoundError(src)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)


def load_pfam2go(pfam2go_path: Path) -> dict[str, set[str]]:
    """Load pfam2go mapping file into dict pfam_id -> set(go_id)."""
    mapping: dict[str, set[str]] = {}
    with open(pfam2go_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.strip().split("\t")
            if len(parts) < 2:
                continue
            pfam_id = parts[0]
            go_id = parts[1]
            if pfam_id not in mapping:
                mapping[pfam_id] = set()
            mapping[pfam_id].add(go_id)
    return mapping


def load_goa_mapping(goa_gaf_path: Path, allowed_ids: Optional[set[str]] = None) -> dict[str, set[str]]:
    """Load UniProt -> GO mappings from a GAF file."""
    mapping: dict[str, set[str]] = {}
    open_fn = gzip.open if str(goa_gaf_path).endswith(".gz") else open
    with open_fn(goa_gaf_path, "rt", encoding="utf-8") as f:
        for line in f:
            if line.startswith("!" ) or not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            uni_id = parts[1]
            go_id = parts[4]
            # If allowed_ids is set, filter by it
            if allowed_ids is not None and uni_id not in allowed_ids:
                continue
            if uni_id not in mapping:
                mapping[uni_id] = set()
            mapping[uni_id].add(go_id)
    return mapping


def write_json(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
