"""File specifying various configuration options for the package."""

import os
from pathlib import Path


# Internal config storage dict (user access not supported!)
_CONFIG = dict()


# Some helpers for config things
def set_data_directory(directory: Path | str):
    """User-facing function for the 

    Parameters
    ----------
    directory : Path | str
        Directory to set. Must be a pathlib.Path or a string that can be cast to a Path.
        The specified directory will be checked for validity
    """
    directory, first_time = _setup_data_directory(directory)
    _CONFIG['data_dir'], _CONFIG['first_run'] = directory, first_time



def _setup_data_directory(directory: Path | str) -> tuple[Path, bool]:
    """Performs setup of the data directory to be used in the package, checking user
    arguments & creating it."""
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


# Setup default directory
directory = os.getenv("HRSF_DATA", None)
if directory is None:
    directory = Path.home() / ".hr_selection_function"
directory, first_time = _setup_data_directory(directory)
