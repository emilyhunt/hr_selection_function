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
_CONFIG["data_url"] = "https://example.com"  # Todo
