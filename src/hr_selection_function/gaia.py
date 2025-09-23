from hr_selection_function.data import requires_data
from numpy.typing import ArrayLike
import pandas as pd
from hr_selection_function.config import _CONFIG


class GaiaDensityEstimator():
    @requires_data
    def __init__(self):
        self._map = pd.read_parquet(_CONFIG['data_dir'] / "density_hp7.parquet")

    def __call__(self, l: ArrayLike, b: ArrayLike):
        # Todo
        pass


class M10Estimator():
    @requires_data
    def __init__(self):
        self._map = pd.read_parquet(_CONFIG['data_dir'] / "m10_hp7.parquet")  # Todo make this

    def __call__(self, l: ArrayLike, b: ArrayLike):
        # Todo
        pass


class MSubsampleEstimator():
    @requires_data
    def __init__(self):
        self._map = pd.read_parquet(_CONFIG['data_dir'] / "subsample_cuts_hp7.parquet")

    def __call__(self, l: ArrayLike, b: ArrayLike):
        # Todo
        pass
