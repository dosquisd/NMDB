"""Data loading utilities for neutron monitor data.

This module provides functions to load and process neutron monitor data
from text files into pandas DataFrames.
"""

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """Load and read neutron monitor data from a specified file path.
    
    Reads data from a text file containing neutron monitor measurements
    and converts it into a pandas DataFrame. The file should be in the
    format used by ./data/ForbushDecrease or similar directories.
    
    Args:
        file_path: Path to the text file containing the data.
        
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
            if value.lower() == "null":
                cleaned_values.append(None)
            else:
                try:
                    cleaned_values.append(float(value))
                except ValueError:
                    cleaned_values.append(pd.to_datetime(value))
        return cleaned_values

    with open(file_path, "r") as file:
        lines = file.readlines()
        header = lines[0].strip().split("   ")
        columns = ["datetime"] + list(map(lambda x: x.strip(), header))
        rows = list(map(clean_row, lines[1:]))

    df = pd.DataFrame(rows, columns=columns)
    return df
