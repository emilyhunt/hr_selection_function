# hr_selection_function

The following package contains a selection function for the [Hunt & Reffert 2024 open cluster catalogue](https://ui.adsabs.harvard.edu/abs/2024A%26A...686A..42H/abstract), as presented in Hunt et al. 2025/2026 (year & reference TBC.)


## Installation

The package can be installed via pip with

```bash
pip install hr_selection_function
```

Note that installation on Windows is probably not possible at this time, as [healpy](https://github.com/healpy/healpy) does not support Windows.

The package will handle fetching data [from Zenodo](https://zenodo.org/records/17350533) automatically. You can control which directory is used for storing data either with the `HRSF_DATA` environment variable, or by calling `set_data_directory` **every time** you use the package:

```python
from hr_selection_function import set_data_directory
set_data_directory("/path/to/desired/location")
```

Otherwise, data is stored by default at `$HOME/.hr_selection_function`.


## Basic use

This library contains two main models: 

- A predictor of detection probability as a function of number of stars & median parallax error of a cluster, `hr_selection_function.HR24SelectionFunction`.
- A 'cluster simulator' that can predict number of stars and median parallax error based on fundamental cluster parameters, `hr_selection_function.NStarsPredictor`.

The former is super easy to use. It takes three arguments: Gaia data density ($\rho_\mathrm{data}$ in the paper), number of stars, median parallax error, and your chosen significance threshold.

```python
import numpy as np
from hr_selection_function import HR24SelectionFunction

# Decide what parameters the clusters
gaia_density = np.full(50, 1000)
n_stars = np.geomspace(1, 200)
median_parallax_error = np.full(50, 0.1)
threshold = np.full(50, 3)

# Query the selection function
model = HR24SelectionFunction()
result = model(gaia_density, n_stars, median_parallax_error, threshold)
```

However, if you don't know the Gaia data density at your chosen position (who would???), then instead, you can pass 5D coordinates defining the location of your cluster in astrometric space, and the data density will be calculated behind the scenes for you.


```python
from astropy.coordinates import SkyCoord
from astropy import units as u

# Setup cluster location to query
# Ensure that it has full, 5D astrometric positions
coordinates = SkyCoord(
    l=np.full(50, 0)*u.deg,
    b=np.full(50, 0)*u.deg,
    pm_l_cosb=np.full(50, 0)*u.mas/u.yr,
    pm_b=np.full(50, 0)*u.mas/u.yr,
    distance=np.full(50, 2000)*u.pc,
    frame="galactic"
)

result = model(coordinates, n_stars, median_parallax_error)
```

And if you don't know n_stars or median_parallax_error for your cluster, then that's ok too! You don't need to simulate the entire cluster to find out - instead, you can predict it with

```python
from hr_selection_function import NStarsPredictor

# Initialize with lower models number to make it faster for demo
nstars_model = NStarsPredictor(models=10)

# Pick values to query
mass = np.linspace(50, 5000)
extinction = np.full(50, 1)
log_age = np.full(50, 7.5)
metallicity = np.full(50, 0.0)
differential_extinction = np.full(50, 0.2)

n_stars_predicted, median_parallax_error_predicted = nstars_model(
    coordinates, 
    mass=mass, 
    extinction=extinction, 
    log_age=log_age, 
    metallicity=metallicity, 
    differential_extinction=differential_extinction
)
```

You can find more examples in the 'examples' directory.


## Documentation

Although the package does not have a documentation page (it's a bit small for that!), all the user-facing functions (such as the `HR24SelectionFunction` class) have function documentation that you can get with `help()` in Python - e.g. like this:

```python
# To get the function docs of the sf itself:
help(HR24SelectionFunction)

# To get the function docs of the __call__() method of the sf:
sf = HR24SelectionFunction()
help(sf)
```


## Issues

If you encounter any problems with using the package, then please raise an issue on [GitHub](https://github.com/emilyhunt/hr_selection_function/issues).


## Developing

We recommend using [uv](https://docs.astral.sh/uv/) if you want to contribute to the package. Installing the same Python version and packages to a new virtual environment is then as easy as

```bash
uv sync --all-extras
```

with the `--all-extras` to ensure that development dependencies are included.

Tests for the library can then be ran with

```bash
uv run pytest
```