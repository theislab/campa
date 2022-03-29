from string import ascii_letters
from pathlib import Path
import os
import shutil
import tempfile

from tqdm import tqdm
import numpy as np
import pandas as pd
import requests

from campa.data import MPPData


def gen_vstr_recarray(m, n, dtype=None):
    size = m * n
    lengths = np.random.randint(3, 5, size)
    letters = np.array(list(ascii_letters))
    gen_word = lambda l: "".join(np.random.choice(letters, l))  # noqa: E731
    arr = np.array([gen_word(gen_len) for gen_len in lengths]).reshape(m, n)
    return pd.DataFrame(arr, columns=[gen_word(5) for i in range(n)]).to_records(index=False, column_dtypes=dtype)


def gen_metadata_df(n, obj_ids, possible_cell_cycles=None, ensure_None=True):
    # TODO: Think about allowing index to be passed for n

    lengths = np.random.randint(3, 5, 6)
    letters = np.array(list(ascii_letters))
    gen_word = lambda l: "".join(np.random.choice(letters, l))  # noqa: E731
    if possible_cell_cycles is None:
        possible_cell_cycles = [gen_word(gen_len) for gen_len in lengths]

    if ensure_None:
        cell_cycle = np.array([None for i in range(n)])
        non_None_size = int(0.85 * n)
        idx = np.random.choice(np.arange(0, n), non_None_size)
        cell_cycle[idx] = np.random.choice(possible_cell_cycles, non_None_size)
    else:
        cell_cycle = np.random.choice(possible_cell_cycles, n)

    letters = np.fromiter(iter(ascii_letters), "U1")
    if n > len(letters):
        letters = letters[: n // 2]  # Make sure categories are repeated

    metadata_dict = {
        "mapobject_id": obj_ids,
        "cell_cycle": cell_cycle,
        "cat": pd.Categorical(np.random.choice(letters, n)),
        "int64": np.random.randint(-50, 50, n),
        "float64": np.random.random(n),
        "uint16": np.random.randint(255, size=n, dtype="uint8"),
    }

    return pd.DataFrame(metadata_dict)


def gen_obj(shape, bounding_box, num_channels, mpp_dtype):
    mean = np.random.randint(0 + bounding_box, shape - bounding_box, 2)
    cov = [[int(bounding_box * 0.5), 0], [0, bounding_box]]
    num_values = np.random.randint(100, 200)
    x, y = np.random.multivariate_normal(mean, cov, num_values).T
    x, y = x.astype(np.uint8), y.astype(np.uint8)
    x = x[(x > 0) & (x < shape)]
    y = y[(y > 0) & (y < shape)]
    if len(x != len(y)):
        num_values = min(len(x), len(y))
        x = x[:num_values]
        y = y[:num_values]
    if np.issubdtype(mpp_dtype, np.floating):
        values = np.array([np.random.random(num_values) for ch in range(num_channels)]).T
    else:
        values = np.array([np.random.randint(100, 5000, num_values) for ch in range(num_channels)]).T
    return x, y, values


def gen_objs(shape, bounding_box, num_channels, obj_ids, mpp_dtype):
    x_all, y_all, values_all = (
        np.empty((0), dtype=np.uint8),
        np.empty((0), dtype=np.uint8),
        np.empty((0, num_channels), dtype=np.uint8),
    )
    obj_ids_all = np.empty((0), dtype=obj_ids.dtype)
    for obj_id in obj_ids:
        x, y, values = gen_obj(shape, bounding_box, num_channels, mpp_dtype)
        obj_ids_all = np.append(obj_ids_all, [obj_id] * len(x))
        x_all = np.append(x_all, x)
        y_all = np.append(y_all, y)
        values_all = np.append(values_all, values, axis=0)
    assert len(x_all) == len(y_all) == len(values_all) == len(obj_ids_all), print(
        len(x_all), len(y_all), len(values_all), len(obj_ids_all)
    )
    return x_all, y_all, values_all, obj_ids_all


def gen_mppdata(
    mpp_dtype=np.uint32,
    shape: int = 100,
    bounding_box: int = 10,
    num_channels: int = 5,
    num_obj_ids: int = 5,
    possible_cell_cycles: list = None,
    channels: list = None,
    data_config: str = "TestData",
    **kwargs,
) -> MPPData:
    """\
    generate several obj ids, for each - generate X, Y, MPP withing bounding box:
    1. generate obj_ids (number defined)
    2. for each obj_id: generate center of object (shape-bb), then generate X as rand (centerX-bb, centerX +bb)
        and Y jointly, and rand(0, bb*bb) values for each channel V
    3. generate channel names as var_names = pd.Index(f"gene{i}" for i in range(num_channels))
    4. generate metadata:
        - generate df with cell cycle:
             - generate 5 diff cell cycles, then assign random cols to that
             - generate  TR: float64, from 100 to 1000

    Params
    ------

    """

    obj_ids = np.array([np.uint32(i) for i in range(num_obj_ids)])

    # generate channel names
    lengths = np.random.randint(3, 5, num_channels)
    letters = np.array(list(ascii_letters))
    gen_word = lambda l: "".join(np.random.choice(letters, l))  # noqa: E731
    if channels is not None:
        num_channels = len(channels)
        channels_df = pd.DataFrame(np.array(channels), columns=["name"])
    else:
        channels_df = pd.DataFrame(np.array([gen_word(gen_len) for gen_len in lengths]), columns=["name"])

    channels_df.index.name = "channel_id"
    # .reset_index().set_index('name').loc[channels]['channel_id']

    metadata = gen_metadata_df(num_obj_ids, obj_ids, possible_cell_cycles, **kwargs)

    X, Y, mpp, obj_ids = gen_objs(shape, bounding_box, num_channels, obj_ids, mpp_dtype)

    data = {
        "x": X,
        "y": Y,
        "obj_ids": obj_ids,
        "mpp": mpp,
    }
    mppdata = MPPData(metadata, channels_df, data, data_config=data_config)
    return mppdata


class test_dataset:
    def __init__(self):
        return

    @staticmethod
    def load_test_dataset(datasetdir=None):
        """
        loads test dataset into a datasetdir
        Args:
            datasetdir (default tests/test_dataset)

        Returns:
            path to a folder where dataset is stored
        """
        from pathlib import Path
        import os

        fname = "test_dataset"
        if datasetdir is None:
            datasetdir = os.path.join(str(Path(__file__).parent))

        folder_dir = test_dataset.load_dataset(
            dataset_path=datasetdir,
            fname=fname,
            backup_url="https://figshare.com/ndownloader/files/34507349?private_link=f004270cd1eeeffdb340",
        )
        return folder_dir

    @staticmethod
    def load_dataset(dataset_path, fname, backup_url):
        """
        Generic function to load dataset
        In dataset_path, creates ierarhy of folders "raw", "archive".
        If unpacked files are already stored in "raw" doesn't do anything.
        Otherwise checks for archive file in "archive" folder and unpacks it into "raw" folder.
        If no files are present there, attempts to load the dataset from URL
         into "archive" folder and then unpacks it into "raw" folder.

        Args:
            dataset_path: path where folder for the dataset will be created.
            fname: desired name of the dataset
            backup_url: link from which dataset will be loaded

        Returns:
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

            test_dataset.download(
                backup_url,
                output_path=archive_path,
            )

            shutil.unpack_archive(archive_path, uncpacked_dir)

        return uncpacked_dir

    @staticmethod
    def getFilename_fromCd(cd):
        """
        Get filename from content-disposition or url request
        """

        import re

        if not cd:
            return None
        fname = re.findall("filename=(.+)", cd)
        if len(fname) == 0:
            return None
        fname = fname[0]
        if '"' in fname:
            fname = fname.replace('"', "")
        return fname

    @staticmethod
    def download(
        url: str,
        output_path=None,
        block_size: int = 1024,
        overwrite: bool = False,
    ) -> None:
        """Downloads a dataset irrespective of the format.

        Args:
            url: URL to download
            output_path: Path to download/extract the files to
            block_size: Block size for downloads in bytes (default: 1024)
            overwrite: Whether to overwrite existing files (default: False)
        """

        if output_path is None:
            output_path = tempfile.gettempdir()

        response = requests.get(url, stream=True)
        filename = test_dataset.getFilename_fromCd(response.headers.get("content-disposition"))

        # currently supports zip, tar, gztar, bztar, xztar
        download_to_folder = output_path.parent
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


if __name__ == "__main__":
    # tmp1=gen_mppdata()
    folder_dir = test_dataset.load_test_dataset()
    print(folder_dir)
