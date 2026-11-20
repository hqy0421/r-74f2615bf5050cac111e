"""
The :mod:`tsfresh.feature_extraction` module contains methods to extract the features from the time series
"""

from tsfresh.feature_extraction.alignment import (
    DualSourceResult,
    extract_features_dual_source,
    match_ordered_observations,
)
from tsfresh.feature_extraction.extraction import extract_features
from tsfresh.feature_extraction.settings import (
    ComprehensiveFCParameters,
    EfficientFCParameters,
    MinimalFCParameters,
)
