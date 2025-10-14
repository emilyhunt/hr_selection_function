"""File specifying various configuration options for the package."""

import os
from pathlib import Path


# Internal config storage dict (user access not supported!)
_CONFIG = dict()

# Setup default directory
_DEFAULT_DIRECTORY = os.getenv("HRSF_DATA", None)
if _DEFAULT_DIRECTORY is None:
    _DEFAULT_DIRECTORY = Path.home() / ".hr_selection_function"

# Some hard defaults
# See https://github.com/zenodo/zenodo/issues/1629 for how to get this link
_CONFIG["data_url"] = (
    "https://zenodo.org/api/records/17350533/files/hr_selection_function.zip/content"
)
_CONFIG["data_md5_hash"] = "62ca0bcd5d4a76acf6ea7c2308e18f3e"
