"""
DeepFinDLP - Dataset Download Script
Downloads and prepares the CIC-IDS2017 dataset.
"""
import os
import sys
import glob
import requests
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


# CIC-IDS2017 CSV file URLs — Official UNB/CIC mirror
BASE_URL = "http://205.174.165.80/CICDataset/CIC-IDS-2017/Dataset/CIC-IDS-2017/CSVs"
DATASET_URLS = {
    "Monday-WorkingHours.pcap_ISCX.csv": f"{BASE_URL}/Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv": f"{BASE_URL}/Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv": f"{BASE_URL}/Wednesday-workingHours.pcap_ISCX.csv",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": f"{BASE_URL}/Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": f"{BASE_URL}/Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": f"{BASE_URL}/Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": f"{BASE_URL}/Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": f"{BASE_URL}/Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
}


def download_file(url: str, dest_path: str) -> bool:
    """Download a file with progress bar."""
    if os.path.exists(dest_path):
        print(f"  [SKIP] Already exists: {os.path.basename(dest_path)}")
        return True

    try:
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))

        with open(dest_path, "wb") as f:
            with tqdm(total=total_size, unit="B", unit_scale=True,
                      desc=f"  {os.path.basename(dest_path)[:50]}") as pbar:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
                    pbar.update(len(chunk))
        return True
    except Exception as e:
        print(f"  [ERROR] Failed to download {url}: {e}")
        if os.path.exists(dest_path):
            os.remove(dest_path)
        return False


def download_dataset():
    """Download all CIC-IDS2017 CSV files."""
    print("=" * 70)
    print("DeepFinDLP - Downloading CIC-IDS2017 Dataset")
    print("=" * 70)

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    success_count = 0

    for filename, url in DATASET_URLS.items():
        dest = os.path.join(config.RAW_DATA_DIR, filename)
        if download_file(url, dest):
            success_count += 1

    print(f"\nDownloaded {success_count}/{len(DATASET_URLS)} files.")
    return success_count == len(DATASET_URLS)


def prepare_dataset():
    """Load all CSV files, clean, and save as Parquet."""
    print("\n" + "=" * 70)
    print("DeepFinDLP - Preparing Dataset")
    print("=" * 70)

    csv_files = sorted(glob.glob(os.path.join(config.RAW_DATA_DIR, "*.csv")))
    if not csv_files:
        print("[ERROR] No CSV files found. Please download the dataset first.")
        print("  You can manually download CIC-IDS2017 from:")
        print("  https://www.unb.ca/cic/datasets/ids-2017.html")
        print("  Place the CSV files in:", config.RAW_DATA_DIR)
        return False

    print(f"Found {len(csv_files)} CSV files. Loading...")

    dfs = []
    for f in tqdm(csv_files, desc="Loading CSVs"):
        try:
            df = pd.read_csv(f, encoding="utf-8", low_memory=False)
            # Strip whitespace from column names
            df.columns = df.columns.str.strip()
            dfs.append(df)
            print(f"  Loaded {os.path.basename(f)}: {len(df)} rows, {len(df.columns)} cols")
        except Exception as e:
            print(f"  [WARN] Error loading {os.path.basename(f)}: {e}")

    if not dfs:
        print("[ERROR] No files could be loaded.")
        return False

    print("\nConcatenating all files...")
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total: {len(df)} rows, {len(df.columns)} columns")

    # Identify label column
    label_col = None
    for candidate in ["Label", " Label", "label"]:
        if candidate in df.columns:
            label_col = candidate
            break
    if label_col is None:
        print("[ERROR] Could not find label column!")
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
