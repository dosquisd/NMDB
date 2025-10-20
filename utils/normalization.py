from typing import Optional, Union

import pandas as pd


def z_score(
    orig_data: Union[pd.DataFrame, pd.Series],
    columns: Optional[list[str]] = None,
) -> Union[pd.DataFrame, pd.Series]:
    """Apply z-score normalization to a DataFrame or Series.

    Args:
        data (Union[pd.DataFrame, pd.Series]): Input data to normalize.
        columns (Optional[list[str]]): Column name to normalize if input is a DataFrame.
            Defaults to [""].

    Returns:
        Union[pd.DataFrame, pd.Series]: Normalized data.
    """
    data = orig_data.copy()

    if isinstance(data, pd.DataFrame) and columns:
        mean = data[columns].mean()
        std = data[columns].std()
        data[columns] = (data[columns] - mean) / std
        return data[columns]

    mean = data.mean()
    std = data.std()
    return (data - mean) / std
