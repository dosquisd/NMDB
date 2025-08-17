"""Utilities package for neutron monitor data analysis.

This package provides modules for loading, processing, calculating metrics,
and visualizing neutron monitor data from cosmic ray events.

Modules:
    load: Functions for loading data from text files into pandas DataFrames.
    calcs: Functions for calculating statistical and complexity metrics.
    plots: Functions for creating visualizations of the calculated metrics.
    constants: Configuration constants and metric definitions.
"""

from utils import load, plots, calcs, constants

from utils.load import load_data
from utils.calcs import calc_metrics
from utils.constants import METRICS, WINDOW_SIZE
from utils.plots import read_metrics_file, plot_metrics

__all__ = [
    "load",
    "plots",
    "calcs",
    "constants",
    "load_data",
    "calc_metrics",
    "read_metrics_file",
    "plot_metrics",
    "METRICS",
    "WINDOW_SIZE",
]
