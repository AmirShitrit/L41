import os
import shutil
from pathlib import Path

from dotenv import load_dotenv

KAGGLE_DATASET = "maysee/mushrooms-classification-common-genuss-images"


def _load_kaggle_token():
    load_dotenv()
    token = os.getenv("KAGGLE_API_TOKEN")
    if token:
        os.environ["KAGGLE_TOKEN"] = token
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _find_class_root(base_dir):
    """Return the first directory whose immediate subdirs all contain image files."""
    base = Path(base_dir)
    candidates = [base] + [p for p in base.rglob("*") if p.is_dir()]
    for candidate in candidates:
        subdirs = [p for p in candidate.iterdir() if p.is_dir()]
        if subdirs and all(
            any(f.suffix.lower() in IMAGE_EXTENSIONS for f in sd.iterdir() if f.is_file())
            for sd in subdirs
        ):
            return candidate
    return None


def _kaggle_download(staging_dir):
    _load_kaggle_token()
    try:
        import kaggle
    except ImportError:
        raise SystemExit(
            "The 'kaggle' package is required. Install it with: uv pip install kaggle"
        )
    kaggle.api.authenticate()
    print(f"Downloading '{KAGGLE_DATASET}' from Kaggle...")
    kaggle.api.dataset_download_files(KAGGLE_DATASET, path=str(staging_dir), unzip=True)


def download_dataset_if_needed(data_dir):
    if Path(data_dir).is_dir():
        return

    staging_dir = Path("data/_staging")
    staging_dir.mkdir(parents=True, exist_ok=True)

    try:
        _kaggle_download(staging_dir)
        class_root = _find_class_root(staging_dir)
        if class_root is None:
            raise RuntimeError(f"Could not locate image class directories in {staging_dir}")
        Path(data_dir).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(class_root), data_dir)
    finally:
        shutil.rmtree(staging_dir, ignore_errors=True)

    print(f"Dataset ready at '{data_dir}'.")
