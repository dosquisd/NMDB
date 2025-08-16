import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Loads and read the data from a specified file path into a pandas DataFrame.
    The filepath should point to a text file in ./data/ForbushDecrease or similar format.
    """

    def clean_row(row: str) -> list[pd.Timestamp | float | None]:
        """
        Process a single row of data, converting values to float or None for 'null' entries.

        All rows must have the same format, first column is datetime, the others are float or 'null'.
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
