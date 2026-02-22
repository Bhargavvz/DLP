"""
DeepFinDLP - Dataset Download Script
Downloads and prepares the CIC-IDS2017 dataset.
Downloads the MachineLearningCSV.zip from the official UNB server.
"""
import os
import sys
import glob
import shutil
import zipfile
import subprocess
import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# Official UNB download URL for the ML-ready CSVs (ZIP archive)
ZIP_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/MachineLearningCSV.zip"
ZIP_DEST = os.path.join(config.DATA_DIR, "MachineLearningCSV.zip")

# HTTP headers to mimic a browser
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "*/*",
}


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with progress bar."""
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000:
        print(f"  [SKIP] Already exists: {os.path.basename(dest_path)}")
        return True

    try:
        print(f"  Downloading from: {url}")
        response = requests.get(url, stream=True, timeout=300, headers=HEADERS)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                      desc=f"  Downloading") as pbar:
                for chunk in response.iter_content(chunk_size=131072):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [ERROR] {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def try_wget_download(url: str, dest_path: str) -> bool:
    """Fallback: try wget which handles redirects/headers better."""
    try:
        print("  Trying wget...")
        result = subprocess.run(
            ["wget", "-O", dest_path, "--no-check-certificate",
             "-U", "Mozilla/5.0", "-q", "--show-progress", url],
            timeout=600
        )
        return result.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000
    except Exception as e:
        print(f"  wget failed: {e}")
        return False


def try_curl_download(url: str, dest_path: str) -> bool:
    """Fallback: try curl."""
    try:
        print("  Trying curl...")
        result = subprocess.run(
            ["curl", "-L", "-o", dest_path, "-A", "Mozilla/5.0",
             "--progress-bar", url],
            timeout=600
        )
        return result.returncode == 0 and os.path.exists(dest_path) and os.path.getsize(dest_path) > 1000
    except Exception as e:
        print(f"  curl failed: {e}")
        return False


def extract_zip(zip_path: str, extract_to: str):
    """Extract ZIP and move CSVs to the raw data directory."""
    print(f"\n  Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)

    # Find all CSVs (may be in subdirectories)
    csv_files = []
    for root, dirs, files in os.walk(extract_to):
        for f in files:
            if f.endswith(".csv"):
                csv_files.append(os.path.join(root, f))

    # Move CSVs to raw data directory
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    for csv_path in csv_files:
        dest = os.path.join(config.RAW_DATA_DIR, os.path.basename(csv_path))
        if not os.path.exists(dest):
            shutil.move(csv_path, dest)
        print(f"    {os.path.basename(csv_path)}")

    print(f"  Extracted {len(csv_files)} CSV files.")
    return len(csv_files) > 0


def download_dataset():
    """Download CIC-IDS2017 using the best available method."""
    print("=" * 70)
    print("DeepFinDLP - Downloading CIC-IDS2017 Dataset")
    print("=" * 70)

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    # Check if already downloaded
    existing = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv"))
    if len(existing) >= 8:
        print(f"  Dataset already present ({len(existing)} CSV files found).")
        return True

    # ── Method 1: Download ZIP via requests ──
    print("\n  [Method 1] Downloading MachineLearningCSV.zip via requests...")
    success = download_file(ZIP_URL, ZIP_DEST)

    # ── Method 2: wget fallback ──
    if not success:
        print("\n  [Method 2] Trying wget...")
        success = try_wget_download(ZIP_URL, ZIP_DEST)

    # ── Method 3: curl fallback ──
    if not success:
        print("\n  [Method 3] Trying curl...")
        success = try_curl_download(ZIP_URL, ZIP_DEST)

    # Extract if downloaded
    if success and os.path.exists(ZIP_DEST):
        extracted = extract_zip(ZIP_DEST, config.DATA_DIR)
        if extracted:
            print("\n  ✓ Dataset downloaded and extracted successfully!")
            return True

    # ── Method 4: Manual instructions ──
    print("\n" + "=" * 70)
    print("  AUTOMATED DOWNLOAD FAILED — Manual Steps:")
    print("=" * 70)
    print(f"""
  Option A — wget (run this on your server):
    wget -O data/MachineLearningCSV.zip \\
      "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/MachineLearningCSV.zip"
    cd data && unzip MachineLearningCSV.zip -d raw/

  Option B — Kaggle:
    1. Get API key from https://www.kaggle.com/settings → Create New API Token
    2. mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/
    3. kaggle datasets download -d cicdataset/cicids2017 -p data/raw --unzip

  Option C — Browser download:
    1. Visit: https://www.unb.ca/cic/datasets/ids-2017.html
    2. Download "MachineLearningCSV.zip"
    3. Extract CSVs into: {config.RAW_DATA_DIR}

  Then re-run:
    python download_data.py --skip-download
""")
    return False


def prepare_dataset():
    """Load all CSV files, clean, and save as Parquet."""
    print("\n" + "=" * 70)
    print("DeepFinDLP - Preparing Dataset")
    print("=" * 70)

    # Search for CSV files recursively
    csv_files = []
    for search_dir in [config.RAW_DATA_DIR, config.DATA_DIR]:
        for root, dirs, files in os.walk(search_dir):
            for f in sorted(files):
                if f.endswith(".csv"):
                    full_path = os.path.join(root, f)
                    if full_path not in csv_files:
                        csv_files.append(full_path)

    if not csv_files:
        print("[ERROR] No CSV files found. Please download the dataset first.")
        print("  Expected location:", config.RAW_DATA_DIR)
        return False

    print(f"Found {len(csv_files)} CSV files. Loading...")

    dfs = []
    for f in tqdm(csv_files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
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

    if label_col != "Label":
        df.rename(columns={label_col: "Label"}, inplace=True)

    df["Label"] = df["Label"].astype(str).str.strip()

    print("\nClass distribution:")
    class_counts = df["Label"].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count:,} ({100*count/len(df):.2f}%)")

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
