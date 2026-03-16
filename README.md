# FastaGO-ExplainableAI

A Windows-friendly pipeline for protein function prediction via:

- **DIAMOND homology search (Swiss-Prot)**
- **Pfam domain detection (PyHMMER)**
- **GO annotation scoring (GOA + pfam2go)**

## Quick start

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 1) Download databases

```powershell
python -m scripts.download_databases
```

### 2) Build DIAMOND database

```powershell
python -m scripts.setup_databases
```

### 3) Test environment

```powershell
python -m scripts.test_setup
```

### 4) Run prediction

```powershell
python -m scripts.run_wei2go --input input/example.fasta --output output/predictions.json
```

## Input/Output

- Put query FASTA files in `input/`
- Outputs are written to `output/`

## Notes

- Make sure `tools/diamond.exe` is present (download from https://github.com/bbuchfink/diamond/releases)
- The pipeline uses a filtered GOA file containing only Swiss-Prot annotations to reduce disk usage.
