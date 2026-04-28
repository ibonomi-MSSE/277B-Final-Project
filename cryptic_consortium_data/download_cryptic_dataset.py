#!/usr/bin/env python3
"""
Download CRyPTIC dataset files from Zenodo.
Usage:
    python download_data.py                  # downloads MUTATIONS.parquet only
    python download_data.py --all            # downloads all files
    python download_data.py --files MUTATIONS UKMYC_PHENOTYPES DRUG_CODES
"""

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD = "15680920"
BASE_URL = f"https://zenodo.org/records/{ZENODO_RECORD}/files"

# All files from the record with their MD5 checksums
FILES = {
    "MUTATIONS.parquet":                {"md5": "d5feeeae14304006ba67aaaef84cff03", "size": "1.1 GB"},
    "VARIANTS.parquet":                 {"md5": "45dcd2628e147f90391fb37cdebb845b", "size": "1.3 GB"},
    "UKMYC_PHENOTYPES.parquet":         {"md5": "020b6c0af6c05e19610a59f5ef97b832", "size": "1.6 MB"},
    "UKMYC_PLATES.parquet":             {"md5": "7b2880f45079c74e88aa4d12c1c18167", "size": "2.4 MB"},
    "UKMYC_GROWTH.parquet":             {"md5": "403ba6d904a3846b8d2535be2de4785a", "size": "15.1 MB"},
    "GENOMES.parquet":                  {"md5": "ebd82e85f71e36de5da10e776b6afe4e", "size": "2.6 MB"},
    "EFFECTS.parquet":                  {"md5": "15bc4e893c0dbcf417cd6e7c94e34517", "size": "5.9 MB"},
    "PREDICTIONS.parquet":              {"md5": "a9eff6b9c527b87a075f2973d052963c", "size": "2.4 MB"},
    "DST_MEASUREMENTS.parquet":         {"md5": "45b4501ea7c3925af565dbbc6188dec0", "size": "2.7 MB"},
    "DST_SAMPLES.parquet":              {"md5": "afa5d4d67e0e9ea9e4f325f06d815e9a", "size": "820.5 kB"},
    "BASHTHEBUG.parquet":               {"md5": "aa8b8eb4ac63d5473b2d8cc82a6f8a69", "size": "3.7 MB"},
    "BASHTHEBUG_CLASSIFICATIONS.parquet":{"md5": "079413ecb6bc172ab97a0445f8556db5", "size": "162.1 MB"},
    "PLATE_LAYOUT.parquet":             {"md5": "cb403c7517ec847467b7980cbc3e5389", "size": "5.9 kB"},
    "WGS_SAMPLES.parquet":              {"md5": "ea798f4cfc28525cf394ff9196c93021", "size": "9.4 MB"},
    "COUNTRIES_LOOKUP.csv.gz":          {"md5": "693cbffe95d305499779e09d7bb903e6", "size": "4.6 kB"},
    "DRUG_CODES.csv.gz":                {"md5": "923d3a193df21698bd6a00f857ab337e", "size": "385 B"},
    "SITES.csv.gz":                     {"md5": "c24c882c8988b5af9940232ada27fb60", "size": "1.3 kB"},
    "DATA_SCHEMA.pdf":                  {"md5": "8b11bfbb9255da6dfc1b8b7aede767d0", "size": "102.7 kB"},
    "RELEASE_NOTES.md":                 {"md5": "dfac1d2007ad30bb749f7bb3bcb9645b", "size": "18.8 kB"},
}

DATA_DIR = Path("data")


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def download_file(filename: str, info: dict) -> bool:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    dest = DATA_DIR / filename

    # Skip if already downloaded and valid
    if dest.exists():
        print(f"  Checking existing {filename}...")
        if md5_file(dest) == info["md5"]:
            print(f"  ✓ {filename} already downloaded and verified")
            return True
        else:
            print(f"  ✗ {filename} exists but MD5 mismatch — re-downloading")

    url = f"{BASE_URL}/{filename}?download=1"
    print(f"  Downloading {filename} ({info['size']})...")

    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(dest, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, unit_divisor=1024, leave=False
        ) as bar:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
                bar.update(len(chunk))

    print(f"  Verifying MD5...")
    actual = md5_file(dest)
    if actual != info["md5"]:
        print(f"  ✗ MD5 mismatch for {filename}!")
        print(f"    expected: {info['md5']}")
        print(f"    got:      {actual}")
        dest.unlink()
        return False

    print(f"  ✓ {filename} downloaded and verified")
    return True


def main():
    parser = argparse.ArgumentParser(description="Download CRyPTIC Zenodo dataset files")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Download all files")
    group.add_argument("--files", nargs="+", metavar="NAME",
                       help="File stems to download e.g. MUTATIONS UKMYC_PHENOTYPES")
    args = parser.parse_args()

    if args.all:
        to_download = FILES
    elif args.files:
        to_download = {}
        for stem in args.files:
            # Allow with or without extension
            match = next((k for k in FILES if k.startswith(stem)), None)
            if match:
                to_download[match] = FILES[match]
            else:
                print(f"Unknown file: {stem}. Available: {list(FILES.keys())}")
                sys.exit(1)
    else:
        # Default: just MUTATIONS
        to_download = {"MUTATIONS.parquet": FILES["MUTATIONS.parquet"]}

    print(f"Downloading {len(to_download)} file(s) to {DATA_DIR.resolve()}/\n")
    failures = []
    for filename, info in to_download.items():
        ok = download_file(filename, info)
        if not ok:
            failures.append(filename)

    print()
    if failures:
        print(f"Failed: {failures}")
        sys.exit(1)
    else:
        print("All done.")


if __name__ == "__main__":
    main()
