from hr_selection_function.data import requires_data
from hr_selection_function.gaia import (
    M10Estimator,
    MSubsampleEstimator,
    GaiaDensityEstimator,
)
import numpy as np
from numpy.typing import ArrayLike
import pandas as pd
import warnings
from astropy.coordinates import SkyCoord
from astropy import units as u
from hr_selection_function.config import _CONFIG
from xgboost import XGBRegressor


class HR24SelectionFunction:
    _N_WALKERS = 64

    @requires_data
    def __init__(self, seed=None, burn_in: int = 1000):
        """Primary selection function from the open cluster selection function paper of
        Hunt et al. 2026.

        Parameters
        ----------
        seed : many (see numpy docs), optional
            Random seed to use when mode=='random_sample'. Default: None
        burn_in : int, optional
            Number of MCMC steps to discard as burn-in. Default: 1000
        """
        self._gaia_density_estimator = GaiaDensityEstimator()

        # Grab mean/median stats
        sample_stats = pd.read_parquet(
            _CONFIG["data_dir"] / "mcmc_sample_stats.parquet"
        )
        self.params_mean = sample_stats.loc[0].to_dict()
        self.params_median = sample_stats.loc[1].to_dict()
        self.parameters = [col for col in sample_stats.columns if col != "type"]

        # Other initialization
        self._burn_in = burn_in
        self._samples = None
        self._n_samples = None
        self._rng = np.random.default_rng(seed=seed)

    @property
    def samples(self):
        """Since the MCMC samples are quite large, this class property ensures they're
        only read in when requested.
        """
        if self._samples is None:
            self._samples = (
                pd.read_parquet(_CONFIG["data_dir"] / "mcmc_samples.parquet")
                .loc[self._burn_in * self._N_WALKERS :]
                .reset_index(drop=True)
            )
        return self._samples

    @property
    def n_samples(self):
        """Since the MCMC samples are quite large, this class property ensures their
        length is only calculated when requested.
        """
        if self._n_samples is None:
            self._n_samples = self.samples.shape[0]
        return self._n_samples

    def __call__(
        self,
        density_or_coordinates: SkyCoord | ArrayLike,
        n_stars: ArrayLike,
        median_parallax_error: ArrayLike,
        threshold: ArrayLike,
        mode: str = "median",
    ) -> np.ndarray:
        """Estimates the detection probability of a given open cluster based on its
        parameters.

        Parameters
        ----------
        density_or_coordinates : SkyCoord | ArrayLike | float | int
            Either a (multi-object) array of astropy SkyCoord objects that include
            a 3D position (can just be ra/dec/distance) and proper motions, *or* the
            values of Gaia data density at these positions of shape (n_clusters,). If a
            SkyCoord is requested, Gaia data density will be calculated automatically by
            this function. May also be a single value.
        n_stars : ArrayLike
            The number of stars in the cluster(s) to query of shape (n_clusters,).
        median_parallax_error : ArrayLike
            The median parallax error of the member stars of the cluster(s) to query of
            shape (n_clusters,).
        threshold : ArrayLike
            The chosen significance threshold for each of the cluster(s) to query of
            shape (n_clusters,).
        mode : str, optional
            Mode to use when selecting which estimate of the MCMC fitting parameters to
            use. Can be one of 'median', 'mean', or 'random_sample', the latter of
            which can be used to estimate epistemic (i.e. systematic/model)
            uncertainties on whether or not a cluster is detected. Default: "median"

        Returns
        -------
        np.ndarray
            A 1D array of detection probabilities for the clusters.
        """
        n_log, scaled_density_log, threshold = self._process_input(
            density_or_coordinates, n_stars, median_parallax_error, threshold
        )
        params = self._get_params(mode)

        good = n_log >= 0
        out = np.zeros_like(n_log)
        out[good] = self._logistic(
            n_log[good], scaled_density_log[good], threshold[good], params
        )

        return out

    def _process_input(
        self,
        data: SkyCoord | ArrayLike,
        n_stars: ArrayLike,
        median_parallax_error: ArrayLike,
        threshold: ArrayLike,
    ):
        """Checks user input and returns all required things to calculate selection."""
        if not isinstance(data, SkyCoord):
            data = np.atleast_1d(data).flatten()
        if not (isinstance(data, SkyCoord) or isinstance(data, np.ndarray)):
            raise ValueError(
                "data must be an astropy SkyCoord specifying the location of points to "
                "query in 5D astrometric space, or a numpy.ndarray containing Gaia data"
                " density for the given cluster at the given points. Instead, data has "
                f"type {type(data)}."
            )

        n_stars = np.atleast_1d(n_stars).flatten()
        median_parallax_error = np.atleast_1d(median_parallax_error).flatten()
        threshold = np.atleast_1d(threshold).flatten()

        # Calculate Gaia density if requested
        if isinstance(data, SkyCoord):
            gaia_density = self._coords_to_gaia_data_density(data)
        else:
            gaia_density = data
        gaia_density = np.atleast_1d(gaia_density).flatten()

        # Perform various other checks
        self._check_input_finite(
            n_stars, median_parallax_error, threshold, gaia_density
        )

        # Convert things into log scale etc
        # We ignore warnings here as np.nan values are dealt with appropriately
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            n_log = np.log10(n_stars)
            scaled_density_log = np.log10(
                n_stars / median_parallax_error**3 / gaia_density
            )
        return n_log, scaled_density_log, threshold

    def _check_input_finite(
        self, n_stars, median_parallax_error, threshold, gaia_density
    ):
        for value, name in zip(
            (n_stars, median_parallax_error, threshold, gaia_density),
            ("n_stars", "median_parallax_error", "threshold", "gaia_density"),
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"All input values must be finite, but not all of {name} is!"
                )
            if not np.all(value >= 0):
                raise ValueError(
                    f"All input values must be >= 0, but not all of {name} is!"
                )

        if not (
            n_stars.shape
            == median_parallax_error.shape
            == threshold.shape
            == gaia_density.shape
        ):
            raise ValueError(
                f"Shape mismatch! All arguments must have the same shape, but instead "
                f"have shapes {gaia_density.shape}, {n_stars.shape}, "
                f"{median_parallax_error.shape}, and {threshold.shape}."
            )

        # Final warnings
        if np.any(threshold < 1) or np.any(threshold > 10):
            warnings.warn(
                "Our model was only fit for CST threshold values between 1 and 10. "
                "Values of the threshold outside of this range are not guaranteed to "
                "be accurate."
            )

    def _coords_to_gaia_data_density(self, coords: SkyCoord):
        """Converts astropy SkyCoord input into Gaia data density at said location."""
        coords_icrs = coords.transform_to("icrs")
        coords_galactic = coords.transform_to("galactic")
        l, b = coords_galactic.l.to(u.deg).value, coords_galactic.b.to(u.deg).value  # noqa: E741
        pmra, pmdec, distance = (
            coords_icrs.pm_ra_cosdec.to(u.mas / u.yr).value,
            coords_icrs.pm_dec.to(u.mas / u.yr).value,
            coords_icrs.distance.to(u.pc).value,
        )
        parallax = 1000 / distance
        return self._gaia_density_estimator(l, b, pmra, pmdec, parallax)

    def _get_params(self, mode):
        """Fetches the correct set of parameters to use for the model, depending on user
        preference. Mode can be median, mean, or random_sample.
        """
        match mode:
            case "median":
                params = self.params_median
            case "mean":
                params = self.params_mean
            case "random_sample":
                i = self._rng.integers(0, self.n_samples)
                params = self.samples.loc[i].to_dict()
            case _:
                raise ValueError(
                    f"selected mode {mode} not supported. Available options: 'median', "
                    "'mean', or 'random_sample'."
                )
        return params

    def _calculate_exponential_terms(self, params, scaled_density, threshold):
        """Calculates the exponential terms that go inside the logistic function."""
        n_0 = self._multi_smoothing_function(
            scaled_density, params["m_n"], params["b_n"], params["c_n"], params["d_n"]
        ) + self._threshold_power_law(threshold, params["u_n"], params["v_n"])
        sigma = self._multi_smoothing_function(
            scaled_density, params["m_s"], params["b_s"], params["c_s"], params["d_s"]
        )
        k = self._multi_smoothing_function(
            scaled_density, params["m_k"], params["b_k"], params["c_k"], params["d_k"]
        )
        return n_0, sigma, k

    def _logistic(self, n, scaled_density, threshold, params):
        """Skew logistic function defined in terms of 50% of stars point n_0 as in
        paper.
        """
        # Offset by pixel area, since this wasn't originally done when fitting the model
        scaled_density += np.log10(self._gaia_density_estimator._HP5_FIELD_AREA)

        n_0, sigma, k = self._calculate_exponential_terms(
            params, scaled_density, threshold
        )
        a = self._calculate_a(n_0, sigma, k)
        return 1 - (1 + np.exp((n - a) / sigma)) ** (-k)

    def _calculate_a(self, n_0, sigma, k):
        """Calculates supporting constant for model."""
        return n_0 - sigma * np.log(0.5 ** (-1 / k) - 1)

    def _threshold_power_law(self, threshold, scale, power):
        """Simple power law on threshold value."""
        return scale * threshold**power

    def _multi_smoothing_function(self, x, gradient, shift, constant, knee):
        """Smoothing function as defined in the paper."""
        return (
            gradient / np.abs(knee) * np.log(1 + np.exp(knee * (x - shift))) + constant
        )


class NStarsPredictor:
    # Todo
    @requires_data
    def __init__(self, n_models: int = 250):
        if n_models > 250 or n_models < 1:
            raise ValueError("n_models must be between 1 and 250 (inclusive).")

        self._m10_estimator = M10Estimator()
        self._m_subsample_estimator = MSubsampleEstimator()

        self.n_models = n_models
        self.models = []
        for i in range(self.n_models):
            model = XGBRegressor()
            file = _CONFIG["data_dir"] / f"nstars_models/{i}.ubj"
            model.load_model(file)
            self.models.append(model)

        self.columns = [
            "ra",
            "dec",
            "mass",
            "extinction",
            "distance",
            "log_age",
            "metallicity",
            "differential_extinction",
        ]
        self.model_columns = [
            "mass",
            "extinction",
            "distance",
            "log_age",
            "metallicity",
            "differential_extinction",
            "m10",
            "m_rybizki",
        ]

    def __call__(self, data: pd.DataFrame, mode: str = "full"):
        predictions = np.asarray(
            [model.predict(data[self.model_columns]) for model in self.models]
        )

        predictions = np.transpose(predictions, (1, 0, 2))
        predictions[:, :, 0] = np.clip(10 ** predictions[:, :, 0] - 1, 0, np.inf)
        predictions[:, :, 1] = np.clip(10 ** predictions[:, :, 1], 0, np.inf)
