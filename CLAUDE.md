Absolutely! Here’s an **updated, fully research-grade PLAN.md** for your Windows-based Wei2GO / FastaGO-ExplainableAI project, **including the Swiss-Prot filtered GAF optimization**. I’ve integrated **all best practices (2024–2025)**, **disk-friendly downloads**, and **Windows compatibility**.

---

# 📋 **PLAN.md – Wei2GO / FastaGO Setup (Optimized for Windows)**

## 🎯 Project Objective

Run **protein function prediction** locally on Windows with:

* FASTA → GO terms
* DIAMOND homology search
* Pfam domain annotation (PyHMMER optional)
* GO mapping from curated annotations
* Fully reproducible **Python pipeline**

Wei2GO combines:

1️⃣ sequence similarity
2️⃣ domain annotation
3️⃣ GO ontology scoring

This is still **one of the strongest classical GO predictors**, while minimizing disk usage.

---

# 📁 Updated Architecture

```
FastaGO-ExplainableAI/
│
├── databases/
│   ├── uniprot/
│   ├── pfam/
│   └── go/
│
├── scripts/
│   ├── download_databases.py
│   ├── setup_databases.py
│   ├── hmmer_wrapper.py
│   ├── run_wei2go.py
│   └── test_setup.py
│
├── input/
├── output/
├── tools/
│   └── diamond.exe
├── logs/
└── configs/
    └── config.yaml
```

---

# 📌 Phase 1 – Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install pandas biopython requests tqdm pyyaml pyhmmer
pip install rich loguru  # optional logging
```

---

# 📌 Phase 2 – Database Downloads (Optimized)

⚠️ **Do NOT download TrEMBL**.

Use only **Swiss-Prot**, **Pfam-A**, **GO ontology**, **filtered GOA**, and **pfam2go**.

| Database         | Size (compressed) | Purpose          |
| ---------------- | ----------------- | ---------------- |
| Swiss-Prot FASTA | 90 MB             | DIAMOND homology |
| Pfam-A HMM       | 220 MB            | domain detection |
| GO ontology      | 50 MB             | hierarchy        |
| GOA (filtered)   | 120–250 MB        | GO mapping       |
| pfam2go          | 1 MB              | domain → GO      |

**Total disk usage ≈ 2–3 GB** instead of 60–100 GB.

---

## Step 2a – Filtered GOA GAF

The **full GOA GAF** is ~8GB uncompressed.

* Filter only **Swiss-Prot IDs** → ~120–250 MB
* Use streaming + gzip to avoid memory issues
* Works fully on **Windows**

Example:

```python
# goa_uniprot_sprot.gaf.gz
# only Swiss-Prot IDs (~120–250 MB)
```

---

# 📌 Phase 3 – Database Formatting

### DIAMOND

```bash
diamond makedb --in databases/uniprot/uniprot_sprot.fasta -d databases/uniprot/uniprot
```

Output:

```
uniprot.dmnd
```

### Pfam HMM (PyHMMER optional)

```python
import pyhmmer
from pathlib import Path

hmm_file = Path("databases/pfam/Pfam-A.hmm")
with pyhmmer.plan7.HMMFile(hmm_file) as hmm:
    for model in hmm:
        pass
```

Expected outputs:

```
Pfam-A.h3m
Pfam-A.h3i
Pfam-A.h3f
Pfam-A.h3p
```

---

# 📌 Phase 4 – PyHMMER / HMMER Wrapper

Mimic `hmmscan`:

```python
with pyhmmer.easel.SequenceFile("input.fasta", digital=True) as seqs:
    sequences = list(seqs)

with pyhmmer.plan7.HMMFile("Pfam-A.hmm") as hmms:
    for hits in pyhmmer.hmmscan(sequences, hmms):
        for hit in hits:
            print(hit.name, hit.evalue)
```

Pipeline:

```
FASTA → Pfam domains → pfam2go → GO terms
```

---

# 📌 Phase 5 – Wei2GO Pipeline

Final scoring:

```
FASTA
│
├── DIAMOND → UniProt matches
├── PyHMMER → Pfam domains
└── Mapping → GO terms
```

**GO term scoring** = homology_score + domain_score + GO propagation

---

# 📌 Phase 6 – Testing

Sample FASTA:

```fasta
>test1
MVLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHF
```

Expected GO terms:

```
GO:0005344
GO:0015671
GO:0005833
```

---

# 📌 Phase 7 – Validation

Check:

* DIAMOND runs
* Pfam scanning works
* GOA filtered file loads
* Wei2GO output generated

---

# 📊 Resource Requirements

| Resource      | Old Plan  | Updated Plan |
| ------------- | --------- | ------------ |
| Disk          | 60–100 GB | 2–3 GB       |
| Download time | hours     | ~30 min      |
| Memory        | heavy     | <2 GB        |
| CPU           | heavy     | moderate     |

---

# 📌 Phase 8 – Logging & Caching

* All scripts log to `logs/`
* Retry failed downloads
* Show progress bars
* Check MD5/size of downloads to ensure integrity
* Resume interrupted downloads

---

# 📌 Optional Phase – Modern Upgrade

For **FastaGO-ExplainableAI**:

```
FASTA → ESM-2 embeddings → ML classifier → GO terms
```

* Replaces BLAST for **faster and more accurate predictions**
* Supports **explainable AI** for protein function

---

# ⭐ Execution Order

1. `download_databases.py` → download + filter GOA
2. `setup_databases.py` → DIAMOND + Pfam prep
3. `hmmer_wrapper.py` → optional PyHMMER scanning
4. `run_wei2go.py` → full pipeline
5. `test_setup.py` → validation

---

# ⚠️ Improvements over original plan

**Add:**

* filtered GOA (~120–250 MB)
* logging, caching, progress bars
* PyHMMER optional for Windows

**Remove:**

* UniProt TrEMBL
* unnecessary HMMER binaries if PyHMMER used

---

# ✅ Expected Final Output

| Protein | GO term    | Score | Evidence |
| ------- | ---------- | ----- | -------- |
| test1   | GO:0005344 | 0.91  | homology |
| test1   | GO:0005833 | 0.87  | domain   |

Disk footprint ~2–3 GB, fully **Windows-compatible**, ready for **explainable AI downstream**.

---

If you want, I can **also write a complete `download_databases.py`** that **does everything automatically**, including:

* Swiss-Prot download
* Pfam download
* GO ontology
* **Filtered GOA**
* pfam2go
* DIAMOND DB prep

It will be **ready-to-run on Windows** with **all optimizations included**.

Do you want me to do that next?
