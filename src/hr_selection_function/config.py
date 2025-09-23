"""File specifying various configuration options for the package."""

import os
from pathlib import Path

from hr_selection_function.data import set_data_directory


# Internal config storage dict (user access not supported!)
_CONFIG = dict()

# Setup default directory
directory = os.getenv("HRSF_DATA", None)
if directory is None:
    directory = Path.home() / ".hr_selection_function"
set_data_directory(directory)

# Some hard defaults
_CONFIG["data_url"] = "https://example.com"  # Todo
