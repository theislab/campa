from typing import Union
from pathlib import Path
import os
import shutil
import tempfile

from tqdm import tqdm
import requests

from campa.constants import SCRIPTS_DIR

Path_t = Union[str, Path]


def load_example_data(data_dir: Path_t = None) -> Path_t:
    """
    Download example data to ``data_dir``.

    Parameters
    ----------
    data_dir
        Defaults to ``notebooks/example_data``.

    Returns
    -------
        Path to folder where dataset is stored.
    """
    from pathlib import Path

    fname = "example_data"
    if data_dir is None:
        data_dir = Path(__file__).parent.parent.parent / "notebooks"

    folder_dir = load_dataset(
        dataset_path=data_dir,
        fname=fname,
        backup_url="https://figshare.com/ndownloader/files/34987036",
    )

    return folder_dir


def load_example_experiment(experiment_dir: Path_t = None) -> Path_t:
    """
    Download example experiment to ``experiment_dir``.

    Parameters
    ----------
    experiment_dir
        Defaults to ``notebooks/example_experiments``.

    Returns
    -------
        Path to folder where experiment is stored
    """
    if experiment_dir is None:
        experiment_dir = Path(__file__).parent.parent.parent / "notebooks" / "example_experiments"
    url = "https://figshare.com/ndownloader/files/34987534"

    uncpacked_dir = Path(os.path.join(experiment_dir, "test_pre_trained"))
    archive_path = Path(os.path.join(experiment_dir, "test_pre_trained.zip"))
    os.makedirs(uncpacked_dir, exist_ok=True)
    foldercontent = os.listdir(str(uncpacked_dir))
    if "weights_epoch010.index" in foldercontent:
        return uncpacked_dir
    elif archive_path.exists():
        shutil.unpack_archive(archive_path, uncpacked_dir)
        return uncpacked_dir
    elif not archive_path.exists():
        print("Path or dataset does not yet exist. Attempting to download...")
        download(
            url,
            output_path=archive_path,
        )

        shutil.unpack_archive(archive_path, uncpacked_dir)
    return uncpacked_dir


def load_test_data():
    """
    Download test data to ``SCRIPTS_DIR/tests``.
    """
    url = "https://figshare.com/ndownloader/files/34988353"
    base_dir = os.path.join(SCRIPTS_DIR, "tests")
    archive_path = os.path.join(base_dir, "_test_data.zip")
    # check if is downloaded already
    if os.path.exists(os.path.join(base_dir, "_data")) and os.path.exists(os.path.join(base_dir, "_experiments")):
        datacontent = os.listdir(os.path.join(base_dir, "_data"))
        experimentcontent = os.listdir(os.path.join(base_dir, "_experiments"))
        if "channels_metadata.csv" in datacontent and "reference_experiment" in experimentcontent:
            return
    # have to unpack/redownload
    if os.path.exists(archive_path):
        shutil.unpack_archive(archive_path, base_dir)
    else:
        print("Path or dataset does not yet exist. Appemting to download...")
        download(url, output_path=archive_path)
        shutil.unpack_archive(archive_path, base_dir)
    return


def load_dataset(dataset_path: Path_t, fname: str, backup_url: str) -> Path_t:
    """
    Load dataset (from URL).

    In dataset_path, creates ierarhy of folders "raw", "archive".
    If unpacked files are already stored in "raw" doesn't do anything.
    Otherwise checks for archive file in "archive" folder and unpacks it into "raw" folder.
    If no files are present there, attempts to load the dataset from URL
    into "archive" folder and then unpacks it into "raw" folder.

    Parameters
    ----------
    dataset_path
        Path where folder for the dataset will be created.
    fname
        Desired name of the dataset
    backup_url
        Link from which dataset will be loaded

    Returns
    -------
    path to a folder where unpacked dataset is stored
    """
    uncpacked_dir = Path(os.path.join(dataset_path, fname, "raw"))
    archive_path = Path(os.path.join(dataset_path, fname, "archive", f"{fname}.zip"))

    os.makedirs(uncpacked_dir, exist_ok=True)
    foldercontent = os.listdir(str(uncpacked_dir))
    if "channels_metadata.csv" in foldercontent:
        return uncpacked_dir

    elif archive_path.exists():
        shutil.unpack_archive(archive_path, uncpacked_dir)
        return uncpacked_dir

    elif not archive_path.exists():
        if backup_url is None:
            raise Exception(
                f"File or directory {archive_path} does not exist and no backup_url was provided.\n"
                f"Please provide a backup_url or check whether path is spelled correctly."
            )

        print("Path or dataset does not yet exist. Attempting to download...")

        download(
            backup_url,
            output_path=archive_path,
        )

        shutil.unpack_archive(archive_path, uncpacked_dir)

    return uncpacked_dir


def getFilename_fromCd(cd):
    """
    Get filename from content-disposition or url request.
    """
    import re

    print(cd)
    if not cd:
        return None
    fname = re.findall("filename=(.+)", cd)
    if len(fname) == 0:
        return None
    fname = fname[0]
    if '"' in fname:
        fname = fname.replace('"', "")
    return fname


def download(
    url: str,
    output_path: Path_t = None,
    block_size: int = 1024,
    overwrite: bool = False,
) -> None:
    """
    Download a dataset irrespective of the format.

    Parameters
    ----------
    url
        URL to download
    output_path
        Path to download/extract the files to
    block_size
        Block size for downloads in bytes (default: 1024)
    overwrite
        Whether to overwrite existing files (default: False)
    """
    if output_path is None:
        output_path = tempfile.gettempdir()

    response = requests.get(url, stream=True)
    print(response.headers)
    filename = getFilename_fromCd(response.headers.get("content-disposition"))
    if filename is None:
        # use content-disposition is empty, try to guess filename
        filename = os.path.basename(output_path)
    print(f"Guessed filename: {filename}")

    # currently supports zip, tar, gztar, bztar, xztar
    download_to_folder = Path(output_path).parent
    os.makedirs(download_to_folder, exist_ok=True)

    archive_formats, _ = zip(*shutil.get_archive_formats())
    is_archived = str(Path(filename).suffix)[1:] in archive_formats
    assert is_archived

    download_to_path = os.path.join(download_to_folder, filename)

    if Path(download_to_path).exists():
        warning = f"File {download_to_path} already exists!"
        if not overwrite:
            print(warning)
            return
        else:
            print(f"{warning} Overwriting...")

    total = int(response.headers.get("content-length", 0))

    print(f"Downloading... {total}")
    with open(download_to_path, "wb") as file:
        for data in tqdm(response.iter_content(block_size)):
            file.write(data)

    os.replace(download_to_path, str(output_path))
