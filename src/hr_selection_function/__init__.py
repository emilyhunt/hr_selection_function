from .config import _DEFAULT_DIRECTORY
from .data import set_data_directory
from .models import HR24SelectionFunction, NStarsPredictor  # noqa: F401
from .gaia import M10Estimator, MSubsampleEstimator, GaiaDensityEstimator  # noqa: F401


set_data_directory(_DEFAULT_DIRECTORY)
