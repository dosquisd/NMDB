"""Constants and configuration for neutron monitor data analysis.

This module defines constants, metrics, and type definitions used throughout
the NMDB (Neutron Monitor Database) analysis pipeline.
"""

import nolds
import antropy as ant
from scipy.stats import entropy

import numpy as np
from typing import Dict, TypedDict, Callable, Any, Literal, Optional


DatetimeBounds = list[str, str]

# Type alias for supported event types
Events = Literal["Forbush Decrease", "Ground Level Enhancement"]


class MetricDetails(TypedDict):
    """Type definition for metric configuration.

    Attributes:
        func: The function to calculate the metric.
        kwargs: Keyword arguments to pass to the function.
    """

    func: Callable[..., np.float64 | np.ndarray]
    kwargs: Dict[str, Any]


class DateEventsInfo(TypedDict):
    """
    Information about the event date.

    Attributes:
        bounds: list[DatetimeBounds]. The start and end bounds of the event date. Generally,
                the bounds are the same for all stations.
        freq: str. The frequency of the data for the event date.
        stations: dict[str, Optional[DatetimeBounds]]. The list of relevant stations
                  for the event date.
    """

    bounds: list[DatetimeBounds]
    freq: str
    stations: dict[str, Optional[DatetimeBounds]]


NAN_THRESHOLD: float = 0.5

WINDOW_SIZE = 130  # 1 unit === 1 minute each window

EWM_ALPHA: Optional[float] = 0.15  # Smoothing factor for Exponential Weighted Mean

OFFSET: int = 10  # Offset for plotting to avoid edge effects

# Dictionary containing all available metrics for analysis
METRICS: Dict[str, MetricDetails] = {
    # Entropy
    "entropy": {
        "func": lambda x, kwargs: entropy(x, **kwargs),
        "kwargs": {},
    },
    "sampen": {
        "func": lambda x, kwargs: nolds.sampen(x, **kwargs),
        "kwargs": {},
    },
    "permutation_entropy": {
        "func": lambda x, kwargs: ant.perm_entropy(x, **kwargs),
        "kwargs": {"normalize": False, "order": 3, "delay": 1},  # Default parameters
    },
    "shannon_entropy": {
        "func": lambda x, kwargs: entropy(
            np.histogram(x, bins="auto", density=True)[0], **kwargs
        ),
        "kwargs": {},
    },
    "spectral_entropy": {
        "func": lambda x, kwargs: ant.spectral_entropy(x, **kwargs),
        "kwargs": {"method": "welch", "sf": 0.1, "normalize": True},
    },
    "app_entropy": {
        "func": lambda x, kwargs: ant.app_entropy(x, **kwargs),
        "kwargs": {},
    },
    # Hurst exponent
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
    "mfhurst_b": {
        "func": lambda x, kwargs: nolds.mfhurst_b(x, **kwargs),
        "kwargs": {},
    },
    # Fractal dimension
    "higuchi_fd": {
        "func": lambda x, kwargs: ant.higuchi_fd(x, **kwargs),
        "kwargs": {"kmax": 10},
    },
    "katz_fd": {
        "func": lambda x, kwargs: ant.katz_fd(x, **kwargs),
        "kwargs": {},
    },
    "petrosian_fd": {
        "func": lambda x, kwargs: ant.petrosian_fd(x, **kwargs),
        "kwargs": {},
    },
    "lepel_ziv": {
        "func": lambda x, kwargs: ant.lziv_complexity(np.array(x), **kwargs),
        "kwargs": {},
    },
    # Chaos indicators
    # "lyap_r": {
    #     "func": lambda x, kwargs: nolds.lyap_r(x, **kwargs),
    #     "data": [],
    #     "kwargs": {},
    # },
    "corr_dim": {
        "func": lambda x, kwargs: nolds.corr_dim(x, **kwargs),
        "kwargs": {"emb_dim": 2},
    },
}

# Relevant dates for the event
# Stations lists are only examples where the event was clear;
# can be modify them as needed
# TODO: Fix datetime event for each station
datetimes: dict[str, Optional[DateEventsInfo]] = {
    "2023-04-23": {
        "bounds": ["2023-04-23 23:00:00", "2023-04-24 06:00:00"],
        "freq": "1h",
        "stations": {
            "AATB": None,
            "APTY": None,
            "IRK2": None,
            "LMKS": None,
            "NEWK": None,
            "NAIN": None,
            "SOPO": None,
        },
    },
    "2024-03-24": {
        "bounds": ["2024-03-24 14:00:00", "2024-03-25 04:30:00"],
        "freq": "90min",
        "stations": {
            "APTY": None,
            "DOMC": None,
            "INVK": None,
            "JUNG1": None,
            "KIEL2": None,
            "LMKS": None,
            "MWSN": None,
            "NEWK": None,
            "MXCO": None,
            "OULU": None,
            # TXBY, YKTK
        },
    },
    "2024-05-10": {
        "bounds": ["2024-05-10 18:00:00", "2024-05-11 01:00:00"],
        "freq": "1h",
        "stations": {
            "APTY": None,
            "DOMB": None,
            "DOMC": None,
            "INVK": None,
            "IRK3": None,
            "JBGO": None,
            "KERG": None,
            "KIEL2": None,
            "LMKS": None,
            "MWSN": None,
            # SOPB, PWNK, SOPO, TERA, THUL, TXBY, YKTK
        },
    },
}
