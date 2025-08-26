"""Data loading utilities for neutron monitor data.

This module provides functions to load and process neutron monitor data
from text files into pandas DataFrames.
"""

import pandas as pd
from functools import lru_cache
from utils.constants import Events


@lru_cache(maxsize=None)
def load_data(file_path: str) -> pd.DataFrame:
    """Load and read neutron monitor data from a specified file path.

    Reads data from a text file containing neutron monitor measurements
    and converts it into a pandas DataFrame. The file should be in the
    format used by ./data/ForbushDecrease or similar directories.

    Args:
        file_path (str): Path to the text file containing the data.

    Returns:
        A pandas DataFrame with datetime index and station data columns.

    Raises:
        FileNotFoundError: If the specified file path does not exist.
        ValueError: If the file format is invalid or cannot be parsed.
    """

    def clean_row(row: str) -> list[pd.Timestamp | float | None]:
        """Process a single row of data, converting values appropriately.

        Converts row values to appropriate types: datetime for the first column,
        float for numeric values, and None for 'null' entries.

        Args:
            row: A single row of data as a string.

        Returns:
            A list of processed values with appropriate types.

        Note:
            All rows must have the same format: first column is datetime,
            subsequent columns are float values or 'null'.
        """
        values = row.strip().split(";")
        cleaned_values = []
        for value in values:
            value = value.strip()

            # Append None values
            if value.lower() == "null":
                cleaned_values.append(None)
                continue

            # Parse float values
            try:
                cleaned_values.append(float(value))
                continue
            except ValueError:
                cleaned_values.append(value)

            # Parse datetime values
            try:
                cleaned_values.append(pd.to_datetime(value, format="%Y-%m-%d %H:%M:%S"))
                continue
            except Exception:  # ValueError, DateParseError
                pass

        return cleaned_values

    with open(file_path, "r") as file:
        lines = file.readlines()
        header = lines[0].strip().split("   ")
        columns = ["datetime"] + list(map(lambda x: x.strip(), header))
        rows = list(map(clean_row, lines[1:]))

        if len(rows[0] != len(columns)):
            rows = list(map(lambda x: x[1:], rows))  # Remove first column (duplicate datetime)

    df = pd.DataFrame(rows, columns=columns)
    return df


def read_metrics_file(
    event: Events,
    date: str,
    station: str,
    window_size: int,
    datetime_cols: dict[str, str] = None,
    suffix: str = "",
) -> pd.DataFrame:
    """Read a pre-calculated metrics file for visualization.

    Loads the CSV file containing calculated metrics for a specific event,
    date, and station combination.

    Args:
        event (Events): The type of event (e.g., 'ForbushDecrease', 'GroundLevelEnhancement').
        date (str): The date in 'YYYY-MM-DD' format.
        station (str): The station identifier.
        window_size (int): The window size used for metrics calculation.
        datetime_cols (dict[str, str]): Optional dictionary mapping column names to datetime formats.
        suffix (str): Optional suffix for the metrics file name.

    Returns:
        A DataFrame containing the metrics data with datetime index.

    Raises:
        FileNotFoundError: If the metrics file does not exist.
        pandas.errors.EmptyDataError: If the file is empty or corrupted.
    """
    file_path = f"./data/{event.replace(' ', '')}/{date}/{station.lower()}_metrics-windowsize_{window_size}{suffix}.csv"

    df = pd.read_csv(file_path)
    if datetime_cols is None:
        return df

    for col, fmt in datetime_cols.items():
        if col in df.columns:
            fmt = fmt or "%Y-%m-%d %H:%M:%S"  # Default format if not provided
            try:
                df[col] = pd.to_datetime(df[col], format=fmt, errors="coerce")
            except ValueError:
                raise ValueError(f"Column '{col}' could not be parsed as datetime.")

    return df