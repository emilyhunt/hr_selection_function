import requests
from hr_selection_function.config import _CONFIG
from tqdm import tqdm


def _check_data_exists():
    pass


def _download_data():
    data_link = "https://example.com"  # Todo
    write_location = _CONFIG["data_dir"] / "raw_data.zip"

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

    # Unzip data
    print("Unzipping data...")
    # Todo

    # Delete raw .zip file
    print("Removing .zip file...")
    # Todo
