"""
Download Raissi et al. (2019) cylinder wake ground truth data.

Source: https://github.com/maziarraissi/PINNs/blob/master/main/Data/cylinder_nektar_wake.mat

Usage:
    python scripts/download_data.py
"""

import hashlib
import sys
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

DATA_URL = (
    "https://github.com/maziarraissi/PINNs/raw/master/main/Data/"
    "cylinder_nektar_wake.mat"
)
TARGET_DIR = Path(__file__).parent.parent / "data"
TARGET_FILE = TARGET_DIR / "cylinder_nektar_wake.mat"
EXPECTED_SIZE_MB_MIN = 20  # sanity check; actual ~30 MB


def download(url: str, target: Path) -> None:
    """Download a file with a simple progress indicator."""
    print(f"Downloading from: {url}")
    print(f"Target:           {target}")
    print()

    try:
        with urlopen(url, timeout=60) as response:
            total_bytes = int(response.headers.get("Content-Length", 0))
            total_mb = total_bytes / (1024 * 1024)
            print(f"File size: {total_mb:.1f} MB")

            downloaded = 0
            chunk_size = 1024 * 64  # 64 KB
            with open(target, "wb") as f:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_bytes > 0:
                        pct = 100 * downloaded / total_bytes
                        mb = downloaded / (1024 * 1024)
                        print(
                            f"\r  {mb:6.1f} / {total_mb:6.1f} MB  ({pct:5.1f}%)",
                            end="",
                            flush=True,
                        )
            print()  # newline after progress

    except URLError as e:
        print(f"\nERROR: Failed to download — {e}")
        print("Try manually downloading from the URL above and placing it at:")
        print(f"  {target}")
        sys.exit(1)


def file_hash(path: Path) -> str:
    """Compute SHA-256 for integrity logging."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists():
        size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
        if size_mb >= EXPECTED_SIZE_MB_MIN:
            print(f"Already exists: {TARGET_FILE}  ({size_mb:.1f} MB)")
            print(f"SHA-256: {file_hash(TARGET_FILE)}")
            print("Delete the file and re-run if you want to re-download.")
            return
        print(f"Existing file too small ({size_mb:.1f} MB), re-downloading...")

    download(DATA_URL, TARGET_FILE)

    size_mb = TARGET_FILE.stat().st_size / (1024 * 1024)
    print(f"\nDownloaded:  {TARGET_FILE}")
    print(f"Size:        {size_mb:.1f} MB")
    print(f"SHA-256:     {file_hash(TARGET_FILE)}")

    if size_mb < EXPECTED_SIZE_MB_MIN:
        print(
            f"\nWARNING: File smaller than expected ({EXPECTED_SIZE_MB_MIN} MB). "
            "Download may be incomplete."
        )
        sys.exit(1)

    print("\nDone. You can now load the data via src.data.loader.CylinderWakeDataset.")


if __name__ == "__main__":
    main()
