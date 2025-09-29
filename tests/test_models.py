import numpy as np
from hr_selection_function import HR24SelectionFunction, NStarsPredictor
from .test_gaia import _get_test_density_data
from astropy.coordinates import SkyCoord
from astropy import units as u
import pytest
import pandas as pd


def test_HR24SelectionFunction():
    ra, dec, pmra, pmdec, parallax, densities = _get_test_density_data(l_and_b=False)
    result_expected = _get_expected_detectability()
    result_expected_mean = _get_expected_detectability("mean")

    # Basic setup of params
    n_clusters = len(ra)
    coords = SkyCoord(
        ra=np.asarray(ra) * u.deg,
        dec=np.asarray(dec) * u.deg,
        pm_ra_cosdec=np.asarray(pmra) * u.mas / u.yr,
        pm_dec=np.asarray(pmdec) * u.mas / u.yr,
        distance=(1000 / np.asarray(parallax)) * u.pc,
    )
    n_stars = np.linspace(0, 100, num=n_clusters)
    errors = np.geomspace(1, 0.01, num=n_clusters)
    thresholds = np.linspace(1, 10, num=n_clusters)
    model = HR24SelectionFunction()

    # Check against previous package values when using density directly
    result = model(densities, n_stars, errors, thresholds)
    np.testing.assert_allclose(result, result_expected, rtol=1e-2, atol=1e-8)

    # Check against previous package values when using coordinates
    result = model(coords, n_stars, errors, thresholds)
    result_expected = _get_expected_detectability()
    np.testing.assert_allclose(result, result_expected, rtol=1e-2, atol=1e-8)

    # Check one-example performance
    result_zero = model(densities[3], n_stars[3], errors[3], thresholds[3])
    assert result[3] == result_zero

    # Check input error raising.
    bad_density = densities.copy()
    bad_density[3] = np.nan
    with pytest.raises(ValueError, match="All input values must be finite"):
        model(bad_density, n_stars, errors, thresholds)

    bad_density[3] = -1
    with pytest.raises(ValueError, match="All input values must be >= 0"):
        model(bad_density, n_stars, errors, thresholds)

    with pytest.raises(ValueError, match="Shape mismatch"):
        model(densities, n_stars, errors, 3.0)

    bad_threshold = thresholds.copy()
    bad_threshold[3] = 10.1

    with pytest.warns(UserWarning, match="Our model was only fit for CST threshold"):
        model(densities, n_stars, errors, bad_threshold)

    # Integration test of other modes
    result = model(densities, n_stars, errors, thresholds, mode="mean")
    np.testing.assert_allclose(result, result_expected_mean, rtol=3e-2, atol=1e-8)

    results = []
    for i in range(100):
        results.append(
            model(densities, n_stars, errors, thresholds, mode="random_sample")
        )

    result = np.mean(results, axis=0)
    np.testing.assert_allclose(
        result, result_expected, rtol=3e-1
    )  # Big error cos samples


def test_n_stars_predictor():
    # Setup test data
    coordinates, mass, extinction, log_age, metallicity, differential_extinction = (
        get_test_cluster()
    )
    test_df = pd.DataFrame.from_dict(
        dict(
            mass=mass,
            extinction=extinction,
            log_age=log_age,
            metallicity=metallicity,
            differential_extinction=differential_extinction,
        )
    )
    n_models = 5
    n_clusters = len(mass)

    # Check if it works at all tbh
    model = NStarsPredictor(models=n_models)
    result = model(
        coordinates,
        mass=mass,
        extinction=extinction,
        log_age=log_age,
        metallicity=metallicity,
        differential_extinction=differential_extinction,
    )
    assert result[0].shape == result[1].shape == (n_clusters,)
    assert np.all(np.isfinite(result[0]))
    assert np.all(np.isfinite(result[1]))
    assert np.all(result[0] >= 0)
    assert np.all(result[1] > 0)

    # Check the dataframe type works & gives near-identical answer
    result_df = model(coordinates, test_df)
    assert np.allclose(result[0], result_df[0])
    assert np.allclose(result[1], result_df[1])

    # Check some certain values
    assert np.allclose(result[0].mean(), 371.06874640000007)
    assert np.allclose(result[1].mean(), 0.05853619699999999)

    # Check we get correct type in (mode == "full")
    result = model(coordinates, test_df, mode="full")
    assert result[0].shape == result[1].shape == (n_clusters, n_models)


def get_test_cluster():
    coordinates = SkyCoord(
        l=np.linspace(0, 360, endpoint=False) * u.deg,
        b=np.linspace(-80, 80) * u.deg,
        distance=np.linspace(500, 5000) * u.pc,
        frame="galactic",
    )
    mass = np.linspace(50, 5000)
    extinction = np.linspace(0, 5)
    log_age = np.linspace(10, 6)
    metallicity = np.linspace(-0.5, 0.5)
    differential_extinction = np.linspace(0, 1)
    return coordinates, mass, extinction, log_age, metallicity, differential_extinction


def _get_expected_detectability(mode="median"):
    if mode == "median":
        return [
            0.0,
            0.007085723421269008,
            0.010443618654938702,
            0.042600789320291654,
            0.18525819893633644,
            0.9669179306367179,
            0.8782480444661485,
            0.965296187018378,
            0.9699061206414834,
            0.9728711248322095,
        ]
    if mode == "mean":
        return [
            0.0,
            0.005972830700591403,
            0.008844614639174742,
            0.036354620277421756,
            0.16404563065598354,
            0.9647549051366275,
            0.8688448242342212,
            0.9626120540069898,
            0.967396788562127,
            0.9702594734699409,
        ]
    raise ValueError("Specified mode not recognized.")
