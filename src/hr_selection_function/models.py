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
        l, b = (  # noqa: E741
            coords_galactic.l.to(u.deg).value,
            coords_galactic.b.to(u.deg).value,
        )
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
    @requires_data
    def __init__(self, models: int = 250):
        """Trained XGBoost model that can predict the number of stars in a cluster and
        the median parallax error of its member stars, as an alternative to simulating
        an entire cluster with ocelot.

        Try setting n_models to a lower value if this model is too slow to run (and if
        accuracy isn't too important). Values of 10-50 can still get good predictions.

        Parameters
        ----------
        models : int, optional
            Number of ensemble models to use. Higher values linearly increase runtime
            but increase accuracy and the number of samples available for uncertainty
            prediction. Default: 250

        Raises
        ------
        ValueError
            On issue with input parameters.
        """
        if models > 250 or models < 1:
            raise ValueError("n_models must be between 1 and 250 (inclusive).")

        self._m10_estimator = M10Estimator()
        self._m_subsample_estimator = MSubsampleEstimator()

        self.n_models = models
        self.models = []
        for i in range(self.n_models):
            model = XGBRegressor()
            file = _CONFIG["data_dir"] / f"nstars_models/{i}.ubj"
            model.load_model(file)
            self.models.append(model)

        self.input_columns = [
            "mass",
            "extinction",
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

    def __call__(
        self,
        coordinates: SkyCoord,
        data: pd.DataFrame | None = None,
        mass: ArrayLike | None = None,
        extinction: ArrayLike | None = None,
        log_age: ArrayLike | None = None,
        metallicity: ArrayLike | None = None,
        differential_extinction: ArrayLike | None = None,
        mode: str = "median",
        verbose: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """_summary_

        Parameters
        ----------
        coordinates : SkyCoord
            A (multi-object) array of astropy SkyCoord objects that include
            a 3D position (can just be ra/dec/distance). Must be convertible into 3D
            ICRS and Galactic coordinate systems.
        data : pd.DataFrame | None, optional
            pandas DataFrame of shape (n_clusters, n_features) containing the columns
            mass [MSun], extinction [mag], log_age [log_10(yrs)], metallicity
            [dex / M/H as in PARSEC models] and differential_extinction [mag], with
            units in the square brackets. Default: None, meaning these columns need to
            be specified with the arguments below.
        mass : ArrayLike | None, optional
            Mass of clusters of shape (n_clusters,) in units of solar masses.
            Default: None
        extinction : ArrayLike | None, optional
            Extinction of clusters of shape (n_clusters,) in units of extinction in the
            V-band. Default: None
        log_age : ArrayLike | None, optional
            Logarithmic age of clusters of shape (n_clusters,) in units of
            log_10(age / yrs). Default: None.
        metallicity : ArrayLike | None, optional
            Metallicity of clusters of shape (n_clusters), specified in dex and in
            metals/hydrogen (M/H) as in the PARSEC isochrones used to train
            this model. Default: None
        differential_extinction : ArrayLike | None, optional
            Differential extinction of clusters of shape (n_clusters,) in units of
            extinction in the V-band. Default: None
        mode : str, optional
            Mode to return predictions in. If "mean" or "median", returns two arrays
            of shape (n_clusters,) which contain the number of stars & median parallax
            error for each cluster, averaged over all models. If "full", returns
            a full array for both number of stars and median parallax error of shape
            (n_clusters, n_models), which can be used to extract uncertainty
            information from the model ensemble. Default: "median"
        verbose : bool, optional
            Whether or not to print when running each model. Default: True

        Returns
        -------
        np.ndarray
            Predicted number of stars for each cluster, of shape (n_clusters,), or shape
            (n_clusters, n_models) when mode=='full'.
        np.ndarray
            Predicted median parallax error of the member stars of each cluster, of
            shape (n_clusters,), or shape (n_clusters, n_models) when mode=='full'.

        Raises
        ------
        ValueError
            On issue with input parameters.
        """
        data = self._process_input(
            coordinates,
            data,
            mass,
            extinction,
            log_age,
            metallicity,
            differential_extinction,
        )
        full_predictions = self._query_models(data, verbose=verbose)
        desired_predictions = self._convert_predictions(full_predictions, mode)
        return desired_predictions

    def _process_input(
        self,
        coordinates: SkyCoord,
        data: pd.DataFrame | None = None,
        mass: ArrayLike | None = None,
        extinction: ArrayLike | None = None,
        log_age: ArrayLike | None = None,
        metallicity: ArrayLike | None = None,
        differential_extinction: ArrayLike | None = None,
    ) -> pd.DataFrame:
        """Checks user input and returns all required things to calculate n_stars and
        median_parallax_error.
        """
        # Todo need to check that arguments are in correct range

        if data is not None:
            # Check we have enough columns
            columns_in = [col in data.columns for col in self.input_columns]
            if not all(columns_in):
                raise ValueError(
                    "Input data is missing columns. It must contain "
                    f"{self.input_columns}"
                )

            # wow i wrote this code when tired didn't i
            mass = data["mass"].to_numpy()
            extinction = data["extinction"].to_numpy()
            log_age = data["log_age"].to_numpy()
            metallicity = data["metallicity"].to_numpy()
            differential_extinction = data["differential_extinction"].to_numpy()

        else:
            args = (mass, extinction, log_age, metallicity, differential_extinction)
            if any([x is None for x in args]):
                raise ValueError(
                    "When not specifying a database of arguments, ALL of mass, "
                    "extinction, log_age, metallicity, and differential_extinction "
                    "must instead be specified."
                )

        # Calculate other stuff
        coordinates_icrs = coordinates.transform_to("icrs")
        coordinates_galactic = coordinates.transform_to("galactic")
        m10 = self._m10_estimator(
            coordinates_icrs.ra.to(u.deg).value, coordinates_icrs.dec.to(u.deg).value
        )
        m_subsample = self._m_subsample_estimator(
            coordinates_galactic.l.to(u.deg).value,
            coordinates_galactic.b.to(u.deg).value,
        )
        distance = coordinates_galactic.distance.to(u.pc).value

        # Create new DataFrame so we don't modify the user's, if they specified it
        # I am also god-awfully afraid of getting a column out of order and fucking it
        # all up so hence why we use a DataFrame internally, it's just easier IMO
        data_for_xgboost = pd.DataFrame.from_dict(
            dict(
                mass=mass,
                extinction=extinction,
                distance=distance,
                log_age=log_age,
                metallicity=metallicity,
                differential_extinction=differential_extinction,
                m10=m10,
                m_rybizki=m_subsample,
            )
        )
        return data_for_xgboost

    def _query_models(self, data: pd.DataFrame, verbose: bool = True) -> np.ndarray:
        """Queries each model & converts predictions back into desired units."""
        predictions = []
        for i, model in enumerate(self.models, start=1):
            if verbose:
                print(f"\rQuerying model {i} of {len(self.models)}", end="")
            predictions.append(model.predict(data[self.model_columns]))
        predictions = np.asarray(predictions)

        if verbose:
            print("")

        # Convert predictions back into n_stars & median_parallax_error
        predictions = np.transpose(predictions, (1, 0, 2))
        predictions[:, :, 0] = np.clip(10 ** predictions[:, :, 0] - 1, 0, np.inf)
        predictions[:, :, 1] = np.clip(10 ** predictions[:, :, 1], 0, np.inf)
        return predictions

    def _convert_predictions(
        self, predictions: np.ndarray, mode: str
    ) -> tuple[np.ndarray, np.ndarray]:
        """Matches user's requested "mode" and converts into two arrays."""
        match mode:
            case "full":
                return predictions[:, :, 0], predictions[:, :, 1]
            case "median":
                averages = np.median(predictions, axis=1)
                return (
                    averages[:, 0],
                    averages[:, 1],
                )
            case "mean":
                averages = np.mean(predictions, axis=1)
                return (
                    averages[:, 0],
                    averages[:, 1],
                )
            case _:
                raise ValueError(f"Specified mode '{mode}' not recognized.")
