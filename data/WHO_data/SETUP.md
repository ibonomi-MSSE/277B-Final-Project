# Data Setup Guide

## Initial Setup

Before using the data_loader API, you need to prepare the cryptic consortium data files.

### Step 1: Download Raw Data

The raw CRyPTIC consortium data should already be in `data/cryptic_consortium_data/data/`:
- `MUTATIONS.parquet`
- `UKMYC_PHENOTYPES.parquet`
- `DRUG_CODES.csv.gz`

If not, run:
```bash
python data/cryptic_consortium_data/download_cryptic_dataset.py
```

### Step 2: Create Processed Files

Run these scripts in order:

```bash
# Create cryptic_consortium_data.parquet (~1GB, takes ~1-2 minutes)
python data/cryptic_consortium_data/create_cryptic_consortium_data.py

# Transform to WHO format (takes ~2-3 minutes, uses 10-30GB RAM)
python data/cryptic_consortium_data/transform.py
```

These create:
1. `data/cryptic_consortium_data.parquet` - Raw CRyPTIC data combined
2. `data/cryptic_consortium_data_filtered.parquet` - Filtered for modeling
3. `data/cryptic_consortium_to_who.parquet` - Transformed to match WHO format

### Step 3: Verify Setup

```bash
ls -lh data/*.parquet
```

You should see:
```
-rw-r--r--  ... data/cryptic_consortium_data.parquet           (~1GB)
-rw-r--r--  ... data/cryptic_consortium_data_filtered.parquet  (~hundreds of MB)
-rw-r--r--  ... data/cryptic_consortium_to_who.parquet         (~hundreds of MB)
```

## Using the Data Loader

Once setup is complete, you can use the data loader:

```python
from data.WHO_data.data_loader import get_who_dataframe

df, drug_lookup = get_who_dataframe({
    'drug': 'morgan_fingerprint',
    'mutation': 'extract_features',
    'target': 'ppv'
})
```

See [README.md](README.md) for full API documentation.

## Troubleshooting

### FileNotFoundError: cryptic_consortium_to_who.parquet

**Problem**: The transform script hasn't finished or failed.

**Solution**: 
1. Check if transform.py is still running: `ps aux | grep transform.py`
2. If stuck, kill it and run again with more memory:
   ```python
   # In transform.py, increase the limit:
   con.execute("SET memory_limit='30GB'")
   ```
3. Check the intermediate file exists: `ls -lh data/cryptic_consortium_data_filtered.parquet`

### MemoryError during transform

**Problem**: Not enough RAM for the transform step.

**Solution**:
1. Close other applications
2. Reduce DuckDB memory limit in transform.py
3. Or use the alternate approach: process in chunks

### Import errors

**Problem**: Cannot import data_loader module.

**Solution**: Run from project root:
```bash
cd /path/to/277B-Final-Project
python -c "from data.WHO_data.data_loader import get_who_dataframe; print('OK')"
```

## Performance Notes

- Initial data loading takes 30-60 seconds (reading parquet files, merging)
- Mutation encoding takes 20-40 seconds (regex parsing)  
- Drug encoding:
  - Morgan fingerprints: ~10 seconds
  - ChemBERTa: ~5 seconds (pre-computed)
  - One-hot: ~1 second
- Genomic positions: ~10 seconds

First run will be slower; subsequent runs benefit from OS file caching.

## File Sizes

Expected sizes after setup:
```
Raw data:
  MUTATIONS.parquet:          ~1.0 GB
  UKMYC_PHENOTYPES.parquet:   ~1.5 MB
  DRUG_CODES.csv.gz:          <1 MB

Processed data:
  cryptic_consortium_data.parquet:           ~983 MB
  cryptic_consortium_data_filtered.parquet:  ~100-500 MB
  cryptic_consortium_to_who.parquet:         ~50-200 MB

WHO data:
  WHO-UCN-TB-2023.6-eng_catalogue_master_file.txt:  ~36 MB
  WHO-UCN-TB-2023.7-eng_genomic_coordinates.txt:    ~7.5 MB
```
