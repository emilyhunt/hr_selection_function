import requests
from hr_selection_function.config import _CONFIG
from tqdm import tqdm
from pathlib import Path
from functools import wraps


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

    _download_file(_CONFIG["data_url"], unzip=True, write_location=_CONFIG["data_dir"])
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
):
    """Downloads a file at a given path. Can also unzip it."""
    if write_location is None:
        write_location = _CONFIG["data_dir"]
    write_location = Path(write_location)

    print(f"Downloading data for hr_selection_function to {write_location}")

    if unzip:
        write_location = write_location / "_temp.zip"

    # Fetch initial dataset
    response = requests.get(data_link, stream=True)
    with tqdm.wrapattr(
        open(write_location, "wb"),
        "write",
        unit="GB",
        unit_scale=True,
        unit_divisor=1024**3,
        miniters=1,
        desc="Data",
        total=int(response.headers.get("content-length", 0)),
    ) as file:
        for chunk in response.iter_content(chunk_size=4096):
            file.write(chunk)

    if not unzip:
        return

    # Unzip data
    print("Unzipping data...")
    # Todo

    # Delete raw .zip file
    print("Removing .zip file...")


def _check_data_directory(directory: Path | str) -> tuple[Path, bool]:
    """Performs setup of the data directory to be used in the package, checking user
    arguments & creating the folder."""
    # Initial checks
    if isinstance(directory, str):
        try:
            directory = Path(directory)
        except Exception as e:
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
            "This looks like your first time running the hr_selection_function package "
            "/ with this data directory. Trying to create data directory at "
            f"{directory}..."
        )
        directory.mkdir(parents=True)
        first_time = False

    return directory, first_time


def _check_data_downloaded(directory: Path | None = None) -> bool:
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
