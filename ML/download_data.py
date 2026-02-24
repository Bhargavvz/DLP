"""
DeepFinDLP - Dataset Download Script
Downloads CIC-IDS2017 dataset from Hugging Face (no API key needed).
Fallbacks: wget from UNB server, manual instructions.
"""
import os
import sys
import glob
import shutil
import zipfile
import subprocess
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# ── Hugging Face dataset (public, no auth needed) ──
HF_REPO = "c01dsnap/CIC-IDS2017"

# ── UNB direct download fallback ──
UNB_ZIP_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/MachineLearningCSV.zip"

# HTTP headers
HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
}


def try_huggingface_download():
    """Download from Hugging Face (no API key needed for public datasets)."""
    print("\n  [Method 1] Downloading from Hugging Face (c01dsnap/CIC-IDS2017)...")

    try:
        from huggingface_hub import snapshot_download
        print("  Using huggingface_hub.snapshot_download...")
        path = snapshot_download(
            repo_id=HF_REPO,
            repo_type="dataset",
            local_dir=os.path.join(config.DATA_DIR, "hf_download"),
        )
        print(f"  Downloaded to: {path}")
        _copy_csvs_from_dir(path)
        return True
    except ImportError:
        print("  huggingface_hub not installed. Installing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface_hub"],
                       capture_output=True)
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                repo_id=HF_REPO,
                repo_type="dataset",
                local_dir=os.path.join(config.DATA_DIR, "hf_download"),
            )
            print(f"  Downloaded to: {path}")
            _copy_csvs_from_dir(path)
            return True
        except Exception as e:
            print(f"  huggingface_hub install failed: {e}")
    except Exception as e:
        print(f"  Hugging Face download failed: {e}")

    return False


def try_wget_download():
    """Try wget from UNB server."""
    print("\n  [Method 2] Trying wget from UNB server...")
    zip_dest = os.path.join(config.DATA_DIR, "MachineLearningCSV.zip")

    try:
        result = subprocess.run(
            ["wget", "-O", zip_dest, "--no-check-certificate",
             "--user-agent=Mozilla/5.0", "-q", "--show-progress", UNB_ZIP_URL],
            timeout=600
        )
        if result.returncode == 0 and os.path.exists(zip_dest) and os.path.getsize(zip_dest) > 10000:
            _extract_zip(zip_dest)
            return True
        else:
            print("  wget download failed (403 or timeout).")
            if os.path.exists(zip_dest):
                os.remove(zip_dest)
    except Exception as e:
        print(f"  wget error: {e}")

    return False


def _extract_zip(zip_path):
    """Extract ZIP and move CSVs to raw data directory."""
    print(f"  Extracting {os.path.basename(zip_path)}...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(config.DATA_DIR)
    _copy_csvs_from_dir(config.DATA_DIR)


def _copy_csvs_from_dir(src_dir):
    """Find and copy CSV files to the raw data directory."""
    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    count = 0
    for root, dirs, files in os.walk(src_dir):
        for f in files:
            if f.endswith(".csv"):
                src = os.path.join(root, f)
                dest = os.path.join(config.RAW_DATA_DIR, f)
                if not os.path.exists(dest):
                    shutil.copy2(src, dest)
                    count += 1
                    print(f"    Copied: {f}")
    print(f"  Moved {count} CSV files to {config.RAW_DATA_DIR}")


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

    # Method 1: Hugging Face (best — no API key needed)
    if try_huggingface_download():
        existing = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv"))
        if existing:
            print(f"\n  ✓ Downloaded {len(existing)} CSV files from Hugging Face!")
            return True

    # Method 2: wget from UNB
    if try_wget_download():
        existing = glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv"))
        if existing:
            print(f"\n  ✓ Downloaded {len(existing)} CSV files from UNB!")
            return True

    # Method 3: Manual instructions
    print("\n" + "=" * 70)
    print("  AUTOMATED DOWNLOAD FAILED — Please download manually:")
    print("=" * 70)
    print(f"""
  Option A — Hugging Face CLI (easiest):
    pip install huggingface_hub
    python -c "from huggingface_hub import snapshot_download; \\
      snapshot_download('c01dsnap/CIC-IDS2017', repo_type='dataset', local_dir='data/hf')"
    cp data/hf/*.csv data/raw/

  Option B — Kaggle (needs API key from kaggle.com/settings):
    pip install kaggle
    mkdir -p ~/.kaggle
    echo '{{"username":"YOUR_USER","key":"YOUR_KEY"}}' > ~/.kaggle/kaggle.json
    kaggle datasets download -d cicdataset/cicids2017 -p data/raw --unzip

  Option C — Browser:
    1. Visit: https://www.kaggle.com/datasets/cicdataset/cicids2017
    2. Click "Download" → Extract CSVs into: {config.RAW_DATA_DIR}

  After placing CSV files, run:
    python download_data.py --skip-download
""")
    return False


def prepare_dataset():
    """Load all CSV files, clean, and save as Parquet."""
    print("\n" + "=" * 70)
    print("DeepFinDLP - Preparing Dataset")
    print("=" * 70)

    # Search for CSV files in data/raw/ only (avoid duplicates from hf_download)
    csv_files = []
    seen_names = set()
    for search_dir in [config.RAW_DATA_DIR]:
        if not os.path.exists(search_dir):
            continue
        for root, dirs, files in os.walk(search_dir):
            for f in sorted(files):
                if f.endswith(".csv") and f not in seen_names:
                    csv_files.append(os.path.join(root, f))
                    seen_names.add(f)

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
