"""Calculation utilities for neutron monitor data analysis.

This module provides functions to calculate various metrics on neutron monitor
data using rolling windows and statistical measures.
"""

import warnings
import pandas as pd
from utils.constants import WINDOW_SIZE, METRICS


def calc_metrics(df: pd.DataFrame, station: str, date: str) -> pd.DataFrame:
    """Calculate statistical metrics for a given station and date.

    Computes various complexity and statistical metrics on neutron monitor data
    using a rolling window approach. The metrics are calculated for each time
    point using a centered window of fixed size.

    Args:
        df: DataFrame containing the neutron monitor data with station columns.
        station: The station identifier (will be converted to uppercase).
        date: The date for which to calculate metrics in 'YYYY-MM-DD' format.

    Returns:
        A DataFrame with calculated metrics indexed by datetime. Contains
        columns for each metric defined in METRICS constant, plus 'value_i'
        and 'window_shape'.

    Raises:
        KeyError: If the specified station is not found in the DataFrame.
        ValueError: If the date format is invalid.

    Side Effects:
        Saves the results to a CSV file in the format:
        './data/ForbushDecrease/{date}/{station}_metrics-windowsize_{WINDOW_SIZE}.csv'

    Note:
        Uses a rolling window of size WINDOW_SIZE (defined in constants).
        Warnings are caught and suppressed during metric calculations.
    """
    station = station.upper()

    df_station = df[[station]]  # I do it this way to avoid pd.Series
    station_rolling = df_station.rolling(WINDOW_SIZE, center=True)

    metric_data = {
        **{"datetime": [], "value": [], "window_shape": []},
        **{metric: [] for metric in METRICS},
    }

    with warnings.catch_warnings(record=True) as _:
        warnings.simplefilter("always")

        for i, window in enumerate(station_rolling):
            if window.empty:
                continue

            data = window[station].dropna().to_numpy()
            metric_data["datetime"].append(df_station.index[i])
            metric_data["value"].append(df_station.iloc[i, 0])
            metric_data["window_shape"].append(data.shape[0])

            for metric, details in METRICS.items():
                try:
                    result = details["func"](data, kwargs=details["kwargs"])
                except Exception as e:
                    print(
                        f"Error: {repr(e)} -- Index: {i} -- Size: {data.shape[0]} -- Metric: {metric}"
                    )
                    continue

                if metric == "mfhurst_b":
                    result = result[0]
                metric_data[metric].append(result)

    df_result = pd.DataFrame(metric_data)
    df_result.index = pd.to_datetime(df_result["datetime"])
    df_result.drop(columns=["datetime"], inplace=True)

    df_result.to_csv(
        f"./data/ForbushDecrease/{date}/{station.lower()}_"
        + f"metrics-windowsize_{WINDOW_SIZE}.csv"
    )

    return df_result
