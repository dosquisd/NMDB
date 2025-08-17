"""Constants and configuration for neutron monitor data analysis.

This module defines constants, metrics, and type definitions used throughout
the NMDB (Neutron Monitor Database) analysis pipeline.
"""

import nolds
from scipy.stats import entropy
from antropy import higuchi_fd, perm_entropy

import numpy as np
from typing import Dict, TypedDict, Callable, Any, Literal


class MetricDetails(TypedDict):
    """Type definition for metric configuration.
    
    Attributes:
        func: The function to calculate the metric.
        kwargs: Keyword arguments to pass to the function.
    """
    func: Callable[..., np.float64 | np.ndarray]
    kwargs: Dict[str, Any]


WINDOW_SIZE = 130  # 1 unit === 1 minute each window

# Dictionary containing all available metrics for analysis
METRICS: Dict[str, MetricDetails] = {
    # Scipy metrics
    "entropy": {
        "func": lambda x, kwargs: entropy(x, **kwargs),
        "kwargs": {},
    },

    # AntroPy metrics
    "permutation_entropy": {
        "func": lambda x, kwargs: perm_entropy(x, **kwargs),
        "kwargs": {"normalize": False, "order": 3, "delay": 1},  # Default parameters
    },
    "higuchi": {
        "func": lambda x, kwargs: higuchi_fd(x, **kwargs),
        "kwargs": {"kmax": 10},
    },
    # Search for more metrics

    # Nolds metrics. Maybe, will be necessary to change each metric kwargs
    # e.g. order for dfa, emb_dim using elbow method, etc.
    "sampen": {
        "func": lambda x, kwargs: nolds.sampen(x, **kwargs),
        "kwargs": {},
    },
    "hurst": {
        "func": lambda x, kwargs: nolds.hurst_rs(x, **kwargs),
        "kwargs": {},
    },
    "dfa": {
        "func": lambda x, kwargs: nolds.dfa(x, **kwargs),
        # By my tests, order > 2 make outliers on edges
        # and orverlap=False add a lot of noise
        "kwargs": {"order": 2, "overlap": True},
    },
    "corr_dim": {
        "func": lambda x, kwargs: nolds.corr_dim(x, **kwargs),
        "kwargs": {"emb_dim": 2},
    },
    "lyap_r": {
        "func": lambda x, kwargs: nolds.lyap_r(x, **kwargs),
        "data": [],
        "kwargs": {},
    },
    # This returns a vector of `matrix_dim`. I'll skip it meanwhile
    # "lyap_e": {
    #     "func": lambda x, kwargs: nolds.lyap_e(x, **kwargs),
    #     # `emb_dim` - 1 must be divisible by `matrix_dim` - 1
    #     "kwargs": {"emb_dim": 10, "matrix_dim": 4},  # Default parameters
    # },
    "mfhurst_b": {
        "func": lambda x, kwargs: nolds.mfhurst_b(x, **kwargs),
        "kwargs": {},
    },
    # This returns a vector too. I'll skip it meanwhile
    # "mfhurst_dm": {
    #     "func": lambda x, kwargs: nolds.mfhurst_dm(x, **kwargs),
    #     "kwargs": {},
    # },
}

# Type alias for supported event types
Events = Literal["Forbush Decrease", "Ground Level Enhancement"]
