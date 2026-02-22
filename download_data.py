"""
DeepFinDLP - Dataset Download Script
Downloads and prepares the CIC-IDS2017 dataset.
Supports multiple download methods: direct HTTP, Kaggle API, or manual placement.
"""
import os
import sys
import glob
import subprocess
import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ── Mirror URLs (tried in order) ──
BASE_URLS = [
    "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs",
    "https://cse-cic-ids2017.s3.us-east-1.amazonaws.com/CSVs",
]

CSV_FILES = [
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]

# HTTP headers to mimic a browser (some servers block bare requests)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,*/*",
}

# Kaggle dataset identifier
KAGGLE_DATASET = "cicdataset/cicids2017"


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with progress bar."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  [SKIP] Already exists: {os.path.basename(dest_path)}")
        return True

    try:
        response = requests.get(url, stream=True, timeout=120, headers=HEADERS)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                      desc=f"  {os.path.basename(dest_path)[:50]}") as pbar:
                for chunk in response.iter_content(chunk_size=65536):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [WARN] {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def try_direct_download():
    """Try downloading from HTTP mirrors."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    success_count = 0

    for filename in CSV_FILES:
        dest = os.path.join(config.RAW_DATA_DIR, filename)
        if os.path.exists(dest) and os.path.getsize(dest) > 1000:
            print(f"  [SKIP] Already exists: {filename}")
            success_count += 1
            continue

        downloaded = False
        for base_url in BASE_URLS:
            url = f"{base_url}/{filename}"
            print(f"  Trying: {url}")
            if download_file(url, dest):
                downloaded = True
                break

        if downloaded:
            success_count += 1
        else:
            print(f"  [FAIL] Could not download: {filename}")

    return success_count


def try_kaggle_download():
    """Try downloading via Kaggle API / kagglehub."""
    print("\n  Attempting Kaggle download...")

    # Method 1: kagglehub (newer, simpler)
    try:
        import kagglehub
        print("  Using kagglehub...")
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        print(f"  Downloaded to: {path}")
        # Copy CSVs to our raw data directory
        _copy_csvs_from_dir(path)
        return True
    except ImportError:
        print("  kagglehub not installed.")
    except Exception as e:
        print(f"  kagglehub failed: {e}")

    # Method 2: kaggle CLI
    try:
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", KAGGLE_DATASET,
             "-p", config.RAW_DATA_DIR, "--unzip"],
            capture_output=True, text=True, timeout=600
        )
        if result.returncode == 0:
            print(f"  Kaggle CLI download successful!")
            return True
        else:
            print(f"  Kaggle CLI failed: {result.stderr}")
    except FileNotFoundError:
        print("  Kaggle CLI not installed.")
    except Exception as e:
        print(f"  Kaggle CLI error: {e}")

    return False


def _copy_csvs_from_dir(src_dir):
    """Recursively find and copy CSV files from a directory."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    import shutil
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".csv") and "ISCX" in f:
                src = os.path.join(root, f)
                dest = os.path.join(config.RAW_DATA_DIR, f)
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)
                    print(f"    Copied: {f}")


def download_dataset():
    """Download CIC-IDS2017 using the best available method."""
    print("=" * 70)
    print("DeepFinDLP - Downloading CIC-IDS2017 Dataset")
    print("=" * 70)

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    # Check if already downloaded
    existing = glob.glob(os.path.join(config.RAW_DATA_DIR, "*ISCX*.csv"))
    if len(existing) >= 8:
        print(f"  Dataset already downloaded ({len(existing)} CSV files found).")
        return True

    # Method 1: Direct HTTP download
    print("\n  [Method 1] Direct HTTP download...")
    count = try_direct_download()
    if count >= 8:
        print(f"\n  ✓ Downloaded {count}/8 files via HTTP.")
        return True

    # Method 2: Kaggle
    print(f"\n  HTTP got {count}/8 files. Trying Kaggle...")
    print("  [Method 2] Kaggle download...")
    if try_kaggle_download():
        existing = glob.glob(os.path.join(config.RAW_DATA_DIR, "*ISCX*.csv"))
        if len(existing) >= 8:
            print(f"\n  ✓ Downloaded via Kaggle ({len(existing)} files).")
            return True

    # Method 3: Manual instructions
    print("\n" + "=" * 70)
    print("  MANUAL DOWNLOAD REQUIRED")
    print("=" * 70)
    print("""
  The automated download failed. Please download manually:

  Option A — Kaggle (recommended):
    1. pip install kaggle
    2. Set up Kaggle API credentials (~/.kaggle/kaggle.json)
    3. kaggle datasets download -d cicdataset/cicids2017 -p data/raw --unzip

  Option B — Kaggle web:
    1. Visit: https://www.kaggle.com/datasets/cicdataset/cicids2017
    2. Download the dataset ZIP
    3. Extract all CSV files into: {raw_dir}

  Option C — UNB website:
    1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
    2. Download "GeneratedLabelledFlows" (CSV files)
    3. Extract into: {raw_dir}

  After placing CSV files, re-run:
    python download_data.py --skip-download
""".format(raw_dir=config.RAW_DATA_DIR))

    return False


def prepare_dataset():
    """Load all CSV files, clean, and save as Parquet."""
    print("\n" + "=" * 70)
    print("DeepFinDLP - Preparing Dataset")
    print("=" * 70)

    # Search for CSV files (handle nested directories from Kaggle)
    csv_files = []
    for root, dirs, files in os.walk(config.RAW_DATA_DIR):
        for f in sorted(files):
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    # Also check data/ directory directly (Kaggle sometimes nests differently)
    if not csv_files:
        for root, dirs, files in os.walk(config.DATA_DIR):
            for f in sorted(files):
                if f.endswith(".csv"):
                    csv_files.append(os.path.join(root, f))

    if not csv_files:
        print("[ERROR] No CSV files found. Please download the dataset first.")
        print("  Expected location:", config.RAW_DATA_DIR)
        return False

    print(f"Found {len(csv_files)} CSV files. Loading...")

    dfs = []
    for f in tqdm(csv_files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            dfs.append(df)
            print(f"  Loaded {os.path.basename(f)}: {len(df):,} rows, {len(df.columns)} cols")
        except Exception as e:
            print(f"  [WARN] Error loading {os.path.basename(f)}: {e}")

    if not dfs:
        print("[ERROR] No files could be loaded.")
        return False

    print("\nConcatenating all files...")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(df):,} rows, {len(df.columns)} columns")

    # Identify label column
    label_col = None
    for candidate in ["Label", " Label", "label"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        print("[ERROR] Could not find label column!")
        print(f"  Available columns: {list(df.columns)}")
        return False

    # Rename label column
    if label_col != "Label":
        df.rename(columns={label_col: "Label"}, inplace=True)

    # Strip label whitespace
    df["Label"] = df["Label"].astype(str).str.strip()

    # Print class distribution
    print("\nClass distribution:")
    class_counts = df["Label"].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count:,} ({100*count/len(df):.2f}%)")

    # Save as Parquet
    os.makedirs(os.path.dirname(config.PARQUET_FILE), exist_ok=True)
    print(f"\nSaving to Parquet: {config.PARQUET_FILE}")
    df.to_parquet(config.PARQUET_FILE, engine="pyarrow", index=False)
    file_size_mb = os.path.getsize(config.PARQUET_FILE) / (1024 * 1024)
    print(f"Saved! File size: {file_size_mb:.1f} MB")

    return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Download CIC-IDS2017 dataset")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip download, only prepare from existing CSVs")
    args = parser.parse_args()

    if not args.skip_download:
        download_dataset()

    prepare_dataset()
