"""Plotting utilities for neutron monitor data visualization.

This module provides functions to create visualizations of calculated metrics
from neutron monitor data analysis, including time series plots and comparative
analysis plots.
"""

import pandas as pd

import scienceplots  # noqa: F401
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from utils.normalization import z_score
from utils.load import read_metrics_file
from utils.constants import METRICS, Events, OFFSET


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


def setup_datetime_axis(ax, hours=2):
    major_locator = mdates.DayLocator()
    minor_locator = mdates.HourLocator(interval=hours)
    major_formatter = mdates.DateFormatter("%m-%d")
    minor_formatter = mdates.DateFormatter("%H")

    ax.xaxis.set_major_locator(major_locator)
    ax.xaxis.set_major_formatter(major_formatter)
    ax.xaxis.set_minor_locator(minor_locator)
    ax.xaxis.set_minor_formatter(minor_formatter)

    ax.tick_params(which="major", labelsize=10, rotation=0, pad=10)
    ax.tick_params(which="minor", labelsize=8, rotation=0, pad=2)


def plot(
    df: pd.DataFrame,
    ax: plt.Axes,
    *,
    metrics: list[str] = ["*"],
    freq_hours: int = 0,
    are_metrics: bool = True,
) -> None:
    """Create a line plot of metrics over time.

    Plots selected metrics as time series using seaborn lineplot with
    customizable styling and date formatting.

    Args:
        df: DataFrame containing metrics data with 'datetime', 'metric', and 'value' columns.
        ax: Matplotlib axes object to plot on.
        metrics: List of metric names to plot. Use ["*"] to plot all metrics.
        freq_hours: Frequency hours string for x-axis tick spacing.
        rotation_xticks: Rotation angle for x-axis tick labels.
        are_metrics: Boolean indicating if the DataFrame contains multiple metrics.

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

    if freq_hours > 0:
        setup_datetime_axis(ax, freq_hours)

    ax.grid(True, which="major", alpha=0.8, linestyle="--")
    ax.legend(bbox_to_anchor=(1, 1), loc="upper left", borderaxespad=0.0)


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
    freq_hours: int = 2,
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
        freq_hours: Frequency hours for x-axis ticks in the overview plot.
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
        df = read_metrics_file(event, date, station, window_size)

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

    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plotting the time series data
    plot(df, axes[0], are_metrics=False)

    # Plot metrics
    plot(
        df_plot,
        axes[1],
        freq_hours=freq_hours,
        metrics=relevant_metrics,  # Metrics important for me
        are_metrics=True,
    )

    min_date = pd.to_datetime(min_datetime)
    max_date = pd.to_datetime(max_datetime)

    # Plot vertical lines for min and max dates
    for ax in axes.flatten():
        ax.axvline(
            x=min_date,
            color="black",
            linestyle="--",
        )
        ax.axvline(
            x=max_date,
            color="black",
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


def plot_metrics_one(
    *,
    window_size: int,
    relevant_metrics: list[str] = None,
    df: pd.DataFrame = None,
    event: Events = "Forbush Decrease",
    date: str = "",
    station: str = "",
    suffix: str = "",
    min_datetime: str = "",
    max_datetime: str = "",
    freq_hours: int = 2,
    save_format: str = "pdf",
    figsize: tuple[int, int] = None,
    show: bool = True,
) -> None:
    """
    Create a single-panel plot of all metrics for neutron monitor data analysis.

    Args:
        window_size (int): Window size used for metric calculations.
        relevant_metrics (list[str]): List of metric names to highlight in the detailed view.
        df (pandas.DataFrame): Optional DataFrame containing pre-calculated metrics. If None, reads from file.
        event (str): Type of cosmic ray event being analyzed.
        date (str): Date of the event in 'YYYY-MM-DD' format.
        station (str): Station identifier for the neutron monitor.
        suffix (str): Optional suffix for the metrics file name.
        min_datetime (str): Start datetime for the time window markers in ISO format.
        max_datetime (str): End datetime for the time window markers in ISO format.
        freq_hours (int): Frequency hours for x-axis ticks in the overview plot.
        save_format (str): File format for saving the plot ('pdf', 'png', etc.).
        show (bool): Whether to display the plot interactively.

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
    delta = 3

    if df is None:
        df = read_metrics_file(event, date, station, window_size, suffix=suffix)

    # Just in case the datetime is not parsed correctly
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"])

    # Set default relevant metrics if none provided
    if relevant_metrics is None:
        relevant_metrics = ["*"]

    if not min_datetime:
        min_datetime = df["datetime"].min()

    if not max_datetime:
        max_datetime = df["datetime"].max()

    if "*" in relevant_metrics:
        metrics_columns = list(METRICS.keys()) + ["value"]
    else:
        metrics_columns = list(filter(lambda x: x in relevant_metrics, df.columns)) + [
            "value"
        ]

    if figsize is None:
        figsize = (16, 9)

    fig, ax = plt.subplots(figsize=figsize)

    # Normalize and offset metrics for better visualization
    plot_df = df.copy().set_index("datetime")
    plot_df[metrics_columns] = z_score(plot_df, metrics_columns)

    for i, col in enumerate(metrics_columns, start=0):
        plot_df[col] = plot_df[col] + OFFSET * i
        ax.text(
            x=plot_df.index[15],
            y=OFFSET * i + delta,
            s=col if col != "value" else station.upper(),
            fontsize=12,
            va="center",
            ha="left",
            fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.2),
        )

    # Plot all metrics with offsets
    data = {"datetime": [], "metric": [], "value": [], "window_shape": []}
    for _, row in plot_df.reset_index().iterrows():
        for metric in metrics_columns:
            data["datetime"].append(row["datetime"])
            data["metric"].append(metric)
            data["value"].append(row[metric])
            data["window_shape"].append(row["window_shape"])

    plot_df = pd.DataFrame(data)

    sns.lineplot(
        data=plot_df,
        x="datetime",
        y="value",
        hue="metric",
        ax=ax,
    )

    # Set title and labels
    ax.set_xlabel("Date", fontsize=14)
    ax.set_ylabel("")

    ax.set_title(
        f"{station.upper()} Station - {event} - Window Size {window_size} units",
        fontsize=16,
    )

    ax.set_yticklabels([])

    # Configure x-axis with date formatting
    if freq_hours > 0:
        setup_datetime_axis(ax, freq_hours)

    # Plot vertical lines for min and max dates
    min_date = pd.to_datetime(min_datetime)
    max_date = pd.to_datetime(max_datetime)

    ax.axvline(
        x=min_date,
        color="black",
        linestyle="--",
    )
    ax.axvline(
        x=max_date,
        color="black",
        linestyle="--",
    )

    ax.grid(True, which="major", alpha=0.8, linestyle="--")
    ax.legend().remove()

    # Save the figure and show it (if specified)
    plt.savefig(
        f"./figures/{event.replace(' ', '')}/{date}/{station.lower()}"
        + f"_metrics-windowsize_{window_size}.{save_format}"
    )

    if show:
        plt.show()
