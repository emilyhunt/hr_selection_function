import requests
from hr_selection_function.config import _CONFIG
from tqdm import tqdm
from pathlib import Path
from functools import wraps
import shutil
import hashlib


def set_data_directory(directory: Path | str):
    """User-facing function for setting the data directory used by the package.

    Called at package initialization automatially, but can also be called by a user
    to change the directory used for data programmatically.

    Parameters
    ----------
    directory : Path | str
        Directory to set. Must be a pathlib.Path or a string that can be cast to a Path.
        The specified directory will be checked for validity
    """
    _CONFIG["data_dir"], _CONFIG["first_run"] = _check_data_directory(directory)
    _CONFIG["data_already_downloaded"] = _check_data_downloaded()


def download_data(redownload: bool = False):
    """Downloads data to be used by the package."""
    if _CONFIG["data_already_downloaded"] and redownload is False:
        return

    _download_file(
        _CONFIG["data_url"],
        unzip=True,
        write_location=_CONFIG["data_dir"],
        hash_to_check=_CONFIG["data_md5_hash"],
    )
    if not _check_data_downloaded():
        raise RuntimeError(
            "Data download failed. Data is not as expected, or where it should be."
        )
    _CONFIG["data_already_downloaded"] = True


def requires_data(func):
    """Wrapper for some function func that ensures that the data directory has been
    created before proceeding.
    """

    @wraps(func)
    def inner(*args, **kwargs):
        download_data(redownload=False)
        func(*args, **kwargs)

    return inner


def _download_file(
    data_link: str,
    unzip: bool = True,
    write_location: Path | None = None,
    hash_to_check: str | None = None,
):
    """Downloads a file at a given path. Can also unzip it."""
    if write_location is None:
        write_location = _CONFIG["data_dir"]
    write_location = Path(write_location)

    if unzip:
        write_location = write_location / "_temp.zip"

    print(f"Downloading data for hr_selection_function to {write_location}")

    # Fetch initial dataset
    response = requests.get(data_link, stream=True)
    with open(write_location, "wb") as handle:
        with tqdm.wrapattr(
            handle,
            "write",
            unit="GB",
            unit_scale=True,
            unit_divisor=1024**3,
            miniters=1,
            total=int(response.headers.get("content-length", 0)),
        ) as file:
            for chunk in response.iter_content(chunk_size=1024**2):  # Chunk 1 MB / time
                file.write(chunk)

    # Optionally check hash
    if hash_to_check is not None:
        print("Checking file md5 hash")
        _check_md5_hash(write_location, hash_to_check)
        print("File hash is good - download was successful")

    # Optionally unzip data
    if not unzip:
        return

    print(f"Unzipping data to {write_location.parent}")
    shutil.unpack_archive(write_location, write_location.parent)

    # Delete raw .zip file
    print(f"Removing .zip file at {write_location}")
    write_location.unlink()

    print("Data downloaded successfully.")


def _check_md5_hash(file: Path | str, expected_hash: str):
    with open(file, "rb") as f:
        file_hash = hashlib.md5()
        while chunk := f.read(1024**2):
            file_hash.update(chunk)

    actual_hash = file_hash.hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError(
            "md5 hash of downloaded file does not match expected file. Your download "
            "was likely corrupted, and will need to be repeated."
            f"Expected hash: {expected_hash}\nActual hash: {actual_hash}"
        )


def _check_data_directory(directory: Path | str) -> tuple[Path, bool]:
    """Performs setup of the data directory to be used in the package, checking user
    arguments & creating the folder."""
    # Initial checks
    if isinstance(directory, str):
        try:
            directory = Path(directory)
        except Exception:
            raise ValueError(
                f"Unable to cast user-specified directory '{directory}' into a path. "
                "Are you sure it is a valid path on your operating system?"
            )

    if not isinstance(directory, Path):
        raise ValueError("Data directory path must be a pathlib.Path or string.")

    # Make the directory
    first_time = False
    if not directory.exists():
        print(
            "This looks like your first time running the hr_selection_function package,"
            " or at least with this data directory.\nTrying to create data directory at"
            f" {directory}..."
        )
        directory.mkdir(parents=True)
        first_time = True

    return directory, first_time


def _check_data_downloaded(directory: Path | None = None) -> bool:
    # Todo this function should really check file size too

    if directory is None:
        directory = _CONFIG["data_dir"]
    directory = Path(directory)

    # Initial directory checks (it should always exist)
    if not directory.exists():
        raise ValueError(
            "Specified data path does not exist. This shouldn't be able to happen, as "
            "the directory should be created during package import."
        )
    if not directory.is_dir():
        raise ValueError("Specified path is not a directory.")

    # Check if all files there
    if not (directory / "density_hp7.parquet").exists():
        return False
    if not (directory / "mcmc_samples.parquet").exists():
        return False
    if not (directory / "subsample_cuts_hp7.parquet").exists():
        return False
    for i in range(250):
        if not (directory / f"nstars_models/{i}.ubj").exists():
            return False
    return True
