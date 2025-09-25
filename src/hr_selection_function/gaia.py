from hr_selection_function.data import requires_data
from numpy.typing import ArrayLike
import pandas as pd
from hr_selection_function.config import _CONFIG
from hr_selection_function.math import (
    vectorized_1d_interpolation,
    vectorized_multivariate_normal,
)
import numpy as np
import healpy as hp


class GaiaDensityEstimator:
    _N_BINS = 21
    _HP7_PIXEL_AREA = hp.nside2pixarea(2**7, degrees=True)
    _HP5_FIELD_AREA = hp.nside2pixarea(2**5, degrees=True) * 9

    @requires_data
    def __init__(self):
        self._map = pd.read_parquet(
            _CONFIG["data_dir"] / "m10_hp7.parquet"
        )  # Todo make this

    def __call__(
        self,
        l: ArrayLike,
        b: ArrayLike,
        pmra: ArrayLike,
        pmdec: ArrayLike,
        parallax: ArrayLike,
    ) -> ArrayLike:
        """Estimates the density within a given region based on a pre-computed set of
        multivariate Gaussian fits in HEALPix level 7 bins.

        l, b, etc. can all be arrays, but should always have the same shape.

        Takes upto 1 second per 100,000 points.

        Parameters
        ----------
        l : ArrayLike
            Galactic longitudes of points to query [in degrees].
        b : ArrayLike
            Galactic latitudes of points to query [in degrees].
        pmra : ArrayLike
            Galactic latitudes of points to query [in mas/yr].
        pmdec : ArrayLike
            Galactic latitudes of points to query [in mas/yr].
        parallax : ArrayLike
            Galactic latitudes of points to query [in mas].

        Returns
        -------
        ArrayLike
            Array of stellar density estimates, measured per mas^-3 yr^2 per HR23
            clustering field, at the chosen l/b/pmra/pmdec/parallax.

        Raises
        ------
        ValueError
            When parallax values are outside of allowed range.
        """
        # Todo: check shapes too
        l, b, pmra, pmdec, parallax = (
            np.atleast_1d(l),
            np.atleast_1d(b),
            np.atleast_1d(pmra),
            np.atleast_1d(pmdec),
            np.atleast_1d(parallax),
        )
        if np.any(parallax > 2) or np.any(parallax < 0):
            raise ValueError(
                "Parallax must be positive and less than 2 mas (i.e. distance > 500 pc.)"
            )

        # Get indices of everything to fetch
        id_pix = hp.ang2pix(2**7, l, b, nest=True, lonlat=True)
        df_indices = self._generate_dataframe_indices(id_pix)

        # Perform interpolations
        parallax_map = self._map.loc[df_indices, "parallax"].to_numpy().reshape(-1, 21)
        values = dict()
        for col in (
            "mean_0",
            "mean_1",
            "cov_0",
            "cov_1",
            "cov_3",
            "n_stars_per_mas",
        ):
            values[col] = vectorized_1d_interpolation(
                parallax,
                parallax_map,
                self._map.loc[df_indices, col].to_numpy().reshape(-1, 21),
            )

        # Since the matrix is symmetric, we skip this one
        values["cov_2"] = values["cov_1"]

        # Query the multivar normal
        proper_motions = np.asarray([pmra, pmdec]).T
        means = np.asarray([values["mean_0"], values["mean_1"]]).T
        covariances = np.asarray([values[f"cov_{i}"] for i in range(4)]).T.reshape(
            -1, 2, 2
        )

        normal_values = vectorized_multivariate_normal(
            proper_motions, means, covariances
        )

        density_per_degree = (
            values["n_stars_per_mas"] * normal_values / self._HP7_PIXEL_AREA
        )

        return density_per_degree * self._HP5_FIELD_AREA

    def _generate_dataframe_indices(self, id_pix):
        return (
            self._N_BINS * id_pix.reshape(-1, 1) + np.arange(self._N_BINS)
        ).flatten()


class M10Estimator:
    @requires_data
    def __init__(self):
        self._map = pd.read_parquet(
            _CONFIG["data_dir"] / "m10_hp7.parquet"
        )  # Todo make this

    def __call__(self, ra: ArrayLike, dec: ArrayLike) -> ArrayLike:
        """Returns value of m10 from Cantat-Gaudin+23 at the given ra/dec specified.

        Parameters
        ----------
        ra : ArrayLike
            Right ascension of points to query [in degrees].
        dec : ArrayLike
            Declination of points to query [in degrees].

        Returns
        -------
        ArrayLike
            Values of m10 at the specified ra/dec.
        """
        healpix_level_7 = hp.ang2pix(2**7, ra, dec, nest=True, lonlat=True)

        # Todo will need to change when using .parquet map
        return self._map[healpix_level_7, 2].to_numpy()


class MSubsampleEstimator:
    @requires_data
    def __init__(self):
        self._map = pd.read_parquet(_CONFIG["data_dir"] / "subsample_cuts_hp7.parquet")

    def __call__(self, l: ArrayLike, b: ArrayLike) -> ArrayLike:
        """Estimates the median magnitude of stars removed from Gaia data by HR23's
        subsample cuts at the given l and b values.

        Parameters
        ----------
        l : ArrayLike
            Galactic longitudes of points to query [in degrees].
        b : ArrayLike
            Galactic latitudes of points to query [in degrees].

        Returns
        -------
        ArrayLike
            Values of m_subsample at the specified l/b.
        """
        healpix_level_7_galactic = hp.ang2pix(2**7, l, b, nest=True, lonlat=True)
        return self._map.loc[healpix_level_7_galactic, "median_mag"].to_numpy()
