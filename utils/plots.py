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
    event: str, date: str, station: str, window_size: int
) -> pd.DataFrame:
    """Read a pre-calculated metrics file for visualization.

    Loads the CSV file containing calculated metrics for a specific event,
    date, and station combination.

    Args:
        event: The type of event (e.g., 'ForbushDecrease', 'GroundLevelEnhancement').
        date: The date in 'YYYY-MM-DD' format.
        station: The station identifier.
        window_size: The window size used for metrics calculation.

    Returns:
        A DataFrame containing the metrics data with datetime index.

    Raises:
        FileNotFoundError: If the metrics file does not exist.
        pandas.errors.EmptyDataError: If the file is empty or corrupted.
    """
    file_path = (
        f"./data/{event}/{date}/{station.lower()}_metrics-windowsize_{window_size}.csv"
    )
    return pd.read_csv(file_path)


def plot(
    df: pd.DataFrame,
    ax: plt.Axes,
    metrics: list[str] = ["*"],
    freq_date_range: str = "",
    colors: dict[str, str] = None,
) -> None:
    """Create a line plot of metrics over time.

    Plots selected metrics as time series using seaborn lineplot with
    customizable styling and date formatting.

    Args:
        df: DataFrame containing metrics data with 'datetime', 'metric', and 'value' columns.
        ax: Matplotlib axes object to plot on.
        metrics: List of metric names to plot. Use ["*"] to plot all metrics.
        freq_date_range: Frequency string for x-axis tick spacing (e.g., '2h', '30min').
        colors: Dictionary mapping metric names to colors. If None, uses default seaborn palette.

    Side Effects:
        Modifies the provided axes object by adding the plot, grid, legend, and formatting.

    Note:
        The function expects the DataFrame to have 'datetime', 'metric', and 'value' columns
        in long format for seaborn compatibility.
    """
    if "*" not in metrics:
        try:
            plot_df = df[df["metric"].isin(metrics)]
        except KeyError:
            plot_df = df.copy()
    else:  # If all metrics are requested or it's an empty list
        plot_df = df.copy()

    sns.lineplot(
        data=plot_df,
        x="datetime",
        y="value",
        hue="metric",
        ax=ax,
        palette=colors,
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
    delta: float = 0.2,
    freq_date_range_1: str = "2h",
    freq_date_range_2: str = "30min",
    save_format: str = "pdf",
    suffix: str = "",
) -> None:
    """Create comprehensive metric plots for neutron monitor data analysis.

    Generates a two-panel plot showing: (1) all metrics over the entire time series,
    and (2) selected relevant metrics over a specified time window with enhanced detail.

    Args:
        window_size: Window size used for metric calculations.
        relevant_metrics: List of metric names to highlight in the detailed view.
        df: Optional DataFrame containing pre-calculated metrics. If None, reads from file.
        event: Type of cosmic ray event being analyzed.
        date: Date of the event in 'YYYY-MM-DD' format.
        station: Station identifier for the neutron monitor.
        min_datetime: Start datetime for the detailed view in ISO format.
        max_datetime: End datetime for the detailed view in ISO format.
        delta: Padding for y-axis limits in the detailed view.
        freq_date_range_1: Frequency for x-axis ticks in the overview plot.
        freq_date_range_2: Frequency for x-axis ticks in the detailed plot.
        save_format: File format for saving the plot ('pdf', 'png', etc.).
        suffix: Optional suffix for the saved plot filename.

    Side Effects:
        - Displays the plot using plt.show()
        - Saves the plot to ./data/{event}/{date}/{station}_metrics-windowsize_{window_size}.{format}

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
    df["datetime_i"] = pd.to_datetime(df["datetime_i"])

    if not min_datetime:
        min_datetime = df["datetime_i"].min()

    if not max_datetime:
        max_datetime = df["datetime_i"].max()

    if "*" not in relevant_metrics:
        metrics_columns = list(filter(lambda x: x in METRICS.keys(), df.columns))
    else:
        metrics_columns = list(df.columns)

    data = {"datetime": [], "metric": [], "value": [], "window_shape": []}
    for _, row in df.iterrows():
        for metric in metrics_columns:
            data["datetime"].append(row["datetime_i"])
            data["metric"].append(metric)
            data["value"].append(row[metric])
            data["window_shape"].append(row["window_shape"])

    df_plot = pd.DataFrame(data)

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plotting all metrics for the PWNK station of all time serie
    plot(df_plot, axes[0], freq_date_range=freq_date_range_1)

    colors = {}
    for line in axes[0].get_lines():
        label = line.get_label()
        color = line.get_color()
        if label and not label.startswith("_"):  # Skip unnamed lines
            colors[label] = color

    plot(
        df_plot,
        axes[1],
        freq_date_range=freq_date_range_2,
        colors=colors,
        metrics=relevant_metrics,  # Metrics important for me
    )

    min_date = pd.to_datetime(min_datetime)
    max_date = pd.to_datetime(max_datetime)

    axes[1].set_xlim(
        min_date,
        max_date,
    )

    ylim_condition = df_plot[
        (df_plot["metric"].isin(relevant_metrics))
        & (df_plot["datetime"] <= max_date)
        & (df_plot["datetime"] >= min_date)
    ]
    delta = 0.2

    axes[1].set_ylim(
        ylim_condition["value"].min() - delta,
        ylim_condition["value"].max() + delta,
    )

    fig.suptitle(
        f"Metrics for {station.upper()} Station - {event}"
        + f"- Window Size {window_size} units (minutes)",
        fontsize=16,
    )

    fig.tight_layout()
    plt.savefig(
        f"./data/{event.replace(' ', '')}/{date}/{station.lower()}"
        + f"_metrics-windowsize_{window_size}{(f'-{suffix}') if suffix else ''}.{save_format}"
    )
    plt.show()
