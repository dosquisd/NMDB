"""Plotting utilities for neutron monitor data visualization.

This module provides functions to create visualizations of calculated metrics
from neutron monitor data analysis, including time series plots and comparative
analysis plots.
"""

import pandas as pd

import scienceplots  # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt

from utils.constants import METRICS, Events


# LaTeX must be installed previously for this to work
plt.style.use(["science", "nature"])
plt.rcParams.update(
    {
        "font.size": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "axes.labelsize": 12,
        "legend.fontsize": 12,
    }
)


def read_metrics_file(
    event: str,
    date: str,
    station: str,
    window_size: int,
    datetime_cols: dict[str, str] = None,
) -> pd.DataFrame:
    """Read a pre-calculated metrics file for visualization.

    Loads the CSV file containing calculated metrics for a specific event,
    date, and station combination.

    Args:
        event: The type of event (e.g., 'ForbushDecrease', 'GroundLevelEnhancement').
        date: The date in 'YYYY-MM-DD' format.
        station: The station identifier.
        window_size: The window size used for metrics calculation.
        datetime_cols: Optional dictionary mapping column names to datetime formats.

    Returns:
        A DataFrame containing the metrics data with datetime index.

    Raises:
        FileNotFoundError: If the metrics file does not exist.
        pandas.errors.EmptyDataError: If the file is empty or corrupted.
    """
    file_path = (
        f"./data/{event}/{date}/{station.lower()}_metrics-windowsize_{window_size}.csv"
    )

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


def plot(
    df: pd.DataFrame,
    ax: plt.Axes,
    *,
    metrics: list[str] = ["*"],
    freq_date_range: str = "",
    are_metrics: bool = True,
) -> None:
    """Create a line plot of metrics over time.

    Plots selected metrics as time series using seaborn lineplot with
    customizable styling and date formatting.

    Args:
        df: DataFrame containing metrics data with 'datetime', 'metric', and 'value' columns.
        ax: Matplotlib axes object to plot on.
        metrics: List of metric names to plot. Use ["*"] to plot all metrics.
        freq_date_range: Frequency string for x-axis tick spacing (e.g., '2h', '30min').

    Side Effects:
        Modifies the provided axes object by adding the plot, grid, legend, and formatting.

    Note:
        The function expects the DataFrame to have 'datetime', 'metric', and 'value' columns
        in long format for seaborn compatibility.
    """
    if are_metrics and "*" not in metrics:
        try:
            plot_df = df[df["metric"].isin(metrics)]
        except KeyError:
            plot_df = df.copy()
    else:  # If all metrics are requested or it's an empty list
        plot_df = df.copy()

    if are_metrics:
        sns.lineplot(
            data=plot_df,
            x="datetime",
            y="value",
            hue="metric",
            ax=ax,
        )
    else:
        sns.lineplot(
            data=plot_df,
            x="datetime",
            y="value",
            ax=ax,
        )

    if freq_date_range:
        date_range = pd.date_range(
            start=plot_df["datetime"].min(),
            end=plot_df["datetime"].max(),
            freq=freq_date_range,
        ).to_list()

        ax.set_xticks(
            ticks=date_range,
            labels=list(map(lambda x: x.strftime("%Y-%m-%d %H:%M:%S"), date_range)),
            rotation=60,
            ha="right",
        )

    ax.grid()
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", borderaxespad=0.0)


def plot_metrics(
    *,
    window_size: int,
    relevant_metrics: list[str],
    df: pd.DataFrame = None,
    event: Events = "Forbush Decrease",
    date: str = "",
    station: str = "",
    min_datetime: str = "",
    max_datetime: str = "",
    freq_date_range_1: str = "2h",
    freq_date_range_2: str = "30min",
    save_format: str = "pdf",
    suffix: str = "",
    show: bool = True,
) -> None:
    """Create comprehensive metric plots for neutron monitor data analysis.

    Generates a two-panel plot showing: (1) the raw data over the entire time series,
    and (2) selected relevant metrics over the time series with vertical markers
    indicating the specified time window boundaries.

    Args:
        window_size: Window size used for metric calculations.
        relevant_metrics: List of metric names to highlight in the detailed view.
        df: Optional DataFrame containing pre-calculated metrics. If None, reads from file.
        event: Type of cosmic ray event being analyzed.
        date: Date of the event in 'YYYY-MM-DD' format.
        station: Station identifier for the neutron monitor.
        min_datetime: Start datetime for the time window markers in ISO format.
        max_datetime: End datetime for the time window markers in ISO format.
        freq_date_range_1: Frequency for x-axis ticks in the overview plot.
        freq_date_range_2: Frequency for x-axis ticks in the metrics plot.
        save_format: File format for saving the plot ('pdf', 'png', etc.).
        suffix: Optional suffix for the saved plot filename.
        show: Whether to display the plot interactively.

    Side Effects:
        - Displays the plot using plt.show()
        - Saves the plot to ./figures/{event}/{date}/{station}_metrics-windowsize_{window_size}.{format}

    Raises:
        FileNotFoundError: If the metrics file cannot be found.
        ValueError: If datetime strings cannot be parsed.

    Note:
        The function expects metrics files to be in the standard format created
        by the calc_metrics function.
    """
    if df is None:
        df = read_metrics_file(event.replace(" ", ""), date, station, window_size)

    # Just in case the datetime is not parsed correctly
    df["datetime"] = pd.to_datetime(df["datetime"])

    if not min_datetime:
        min_datetime = df["datetime"].min()

    if not max_datetime:
        max_datetime = df["datetime"].max()

    metrics_columns = list(filter(lambda x: x in METRICS.keys(), df.columns))
    data = {"datetime": [], "metric": [], "value": [], "window_shape": []}
    for _, row in df.iterrows():
        for metric in metrics_columns:
            data["datetime"].append(row["datetime"])
            data["metric"].append(metric)
            data["value"].append(row[metric])
            data["window_shape"].append(row["window_shape"])

    df_plot = pd.DataFrame(data)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plotting all metrics for the PWNK station of all time serie
    plot(df, axes[0], freq_date_range=freq_date_range_1, are_metrics=False)

    # Plot metrics
    plot(
        df_plot,
        axes[1],
        freq_date_range=freq_date_range_2,
        metrics=relevant_metrics,  # Metrics important for me
        are_metrics=True,
    )

    min_date = pd.to_datetime(min_datetime)
    max_date = pd.to_datetime(max_datetime)

    # Plot vertical lines for min and max dates
    for ax in axes.flatten():
        ax.axvline(
            x=min_date,
            color="red",
            linestyle="--",
        )
        ax.axvline(
            x=max_date,
            color="red",
            linestyle="--",
        )

    fig.suptitle(
        f"Metrics for {station.upper()} Station - {event}"
        + f"- Window Size {window_size} units (minutes)",
        fontsize=16,
    )

    fig.tight_layout()
    plt.savefig(
        f"./figures/{event.replace(' ', '')}/{date}/{station.lower()}"
        + f"_metrics-windowsize_{window_size}{(f'-{suffix}') if suffix else ''}.{save_format}"
    )

    if show:
        plt.show()
