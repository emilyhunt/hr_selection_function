"""Test data handling and download in the library."""

from hr_selection_function.data import (
    _download_file,
    _check_data_directory,
    _check_data_downloaded,
    set_data_directory,
)
from hr_selection_function.config import _CONFIG, _DEFAULT_DIRECTORY
from pathlib import Path
import shutil
import warnings
import pytest


location = Path.cwd() / "hr_selection_function_test_data_578322146558"


def test__download_file():
    file = "https://sample-files.com/downloads/compressed/zip/basic-text.zip"  # Todo find something better
    location.mkdir(exist_ok=True, parents=True)

    try:
        _download_file(file, unzip=True, write_location=location)

        # Check all is as expected
        assert location.exists()
        assert (location / "file1.txt").exists()
        assert (location / "file2.txt").exists()
        assert (location / "file3.txt").exists()
        assert (location / "file4.txt").exists()
        assert (location / "file5.txt").exists()
        assert not (location / "_temp.zip").exists()

    finally:
        shutil.rmtree(location)


def test_file_download_no_zip():
    file = "https://sample-files.com/downloads/compressed/zip/basic-text.zip"  # Todo find something better
    location.mkdir(exist_ok=True, parents=True)

    write = location / "a_zip_file.zip"

    try:
        _download_file(file, unzip=False, write_location=write)

        # Check all is as expected
        assert location.exists()
        assert write.exists()

    finally:
        shutil.rmtree(location)


def test__check_data_directory():
    # Assert arg type
    with pytest.raises(
        ValueError, match="Data directory path must be a pathlib.Path or string."
    ):
        _check_data_directory(42)

    # Check directory is made in usual circumstance
    try:
        location_made, first_time = _check_data_directory(location)
        assert first_time
        assert location.exists()
        assert location == location_made
    finally:
        location.rmdir()

    # Check again, if it's a string is made in usual circumstance
    try:
        location_made, first_time = _check_data_directory(str(location))
        assert first_time
        assert location.exists()
        assert location == location_made
    finally:
        location.rmdir()

    # Check first time arg
    # Check directory is made in usual circumstance
    location.mkdir(parents=True)
    try:
        location_made, first_time = _check_data_directory(location)
        assert first_time is False
        assert location.exists()
    finally:
        location.rmdir()


def test_download_data():
    warnings.warn("Todo once data is online somewhere. md5 hash is also not tested!")


def test__check_data_downloaded():
    # Initial setup
    (location / "nstars_models").mkdir(parents=True)

    files = [
        "density_hp7.parquet",
        "mcmc_samples.parquet",
        "subsample_cuts_hp7.parquet",
    ]
    files = files + [f"nstars_models/{i}.ubj" for i in range(250)]
    files = [location / file for file in files]
    for file in files:
        file.touch()

    try:
        assert _check_data_downloaded(location)

        files[0].unlink()
        assert _check_data_downloaded(location) is False

        files[0].touch()
        assert _check_data_downloaded(location)

        files[-100].unlink()  # Checks sensitivity to n_stars models
        assert _check_data_downloaded(location) is False

    finally:
        shutil.rmtree(location)


def test_set_data_directory():
    # Tests won't work if this isn't right
    assert _CONFIG["data_dir"] == _DEFAULT_DIRECTORY
    assert not location.exists()

    try:
        set_data_directory(location)
        assert location.exists()
        assert _CONFIG["data_dir"] == location
        assert _CONFIG["first_run"]
        assert _CONFIG["data_already_downloaded"] is False

        # Now reset it
        set_data_directory(_DEFAULT_DIRECTORY)
        assert _CONFIG["data_dir"] == _DEFAULT_DIRECTORY

    finally:
        shutil.rmtree(location)
