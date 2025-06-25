import kagglehub

import os
import shutil
from pathlib import Path
# Download latest version
def download_latest_version(dest: str = "./data/invasive-species") -> str:
    """
    Download the latest ‘invasive-species’ dataset via KaggleHub and copy it
    into *dest* (relative to the current directory).

    Parameters
    ----------
    dest : str, optional
        Where the dataset should live in your project.  Default is
        ``"./data/invasive-species"``.

    Returns
    -------
    str
        Absolute path to the folder containing the files.
    """
    # 1. Pull/update the dataset in KaggleHub’s cache  (~/.cache/kagglehub/…)
    cache_dir = Path(kagglehub.dataset_download("benmcewen1/invasive-species"))

    # 2. Resolve destination folder in this project and create it if needed
    dest_dir = Path(dest).resolve()
    dest_dir.mkdir(parents=True, exist_ok=True)

    # 3. Copy (or overwrite) everything from cache → dest
    #    `dirs_exist_ok=True` keeps already-existing sub-files.
    shutil.copytree(cache_dir, dest_dir, dirs_exist_ok=True)

    print("Dataset copied to:", dest_dir)
    return str(dest_dir)

def list_files(dir_path: str):
    """Recursively list *all* files under *dir_path*."""
    return [p.as_posix() for p in Path(dir_path).rglob("*") if p.is_file()]

