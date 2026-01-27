# NMDB — Analysis of NMDB neutron monitor events

> [!NOTE]
> There is another approach using graphs in this repo: [dosquisd/NMDB-FD-PredictorWithGraphs](https://github.com/dosquisd/NMDB-FD-PredictorWithGraphs)

This repository contains code and notebooks used to analyse neutron monitor data from the NMDB (Neutron Monitor Database). The code in `utils/` implements data loading, metric calculations and plotting helpers. The notebooks run the analysis pipeline (data loading -> sliding-window metrics -> visualization) for a few cosmic-ray events of interest.

Most examples and the provided data focus on Forbush Decrease events (see `./data/ForbushDecrease`).

Important: the repository contains code used as a record of what was done while working with NMDB data. It is not packaged or documented as a public library. Treat the modules as analysis scripts rather than a reusable API.

## Data source

All raw neutron monitor data used here was obtained from: [https://www.nmdb.eu/nest/index.php](https://www.nmdb.eu/nest/index.php)

The project expects the data arranged under `./data/` (example: `./data/ForbushDecrease/<YYYY-MM-DD>/all.txt`) and writes metric CSVs and figures back into `./data/` and `./figures/` respectively.

## Repository layout (relevant files)

- `001_process_data.ipynb` — initial data exploration and simple plots.
- `002_calc_some_metrics.ipynb` — example for calculating metrics for a single station and plotting them.
- `003_final_results.ipynb` — batch calculation of metrics across many stations/dates and plotting to `./figures/`.
- `utils/` — analysis utilities used by the notebooks:
  - `__init__.py` — convenience exports for the package.
  - `load.py` — `load_data(file_path)` reads the `all.txt` style files into a pandas DataFrame (with caching via `lru_cache`).
  - `constants.py` — configuration: `WINDOW_SIZE`, metric definitions `METRICS`, `NAN_THRESHOLD`, and helper types. This file maps metric names to the functions used (antropy, nolds, scipy, etc.).
  - `calcs.py` — `calc_metrics(df, station, date)` computes rolling-window metrics (using `WINDOW_SIZE`) and writes CSVs to `./data/ForbushDecrease/<date>/<station>_metrics-windowsize_<WINDOW_SIZE>.csv`.
  - `plots.py` — helpers to read metric CSVs and produce publication-ready plots (saves into `./figures/`).

## Dependencies

The project `pyproject.toml` lists the required Python packages. Key dependencies used by the code are (as recorded in `pyproject.toml`):

- Python >= 3.12
- pandas
- numpy (pin: <= 2.2 in the project)
- polars (optional in the repo)
- matplotlib, seaborn, scienceplots
- antropy, nolds, scipy
- scikit-learn

Note: this repository includes an `uv.lock` lockfile (used by the [uv](https://docs.astral.sh/uv/) package manager). If you prefer `uv` to manage the environment, that lock file can be used as a starting point to reproduce the environment.

## Example environment setup

Create a virtual environment and install the runtime dependencies (one possible approach):

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install \
    antropy ipykernel matplotlib nolds numpy<=2.2 pandas polars scienceplots \
    scikit-learn scipy seaborn
```

## Running the analysis

The notebooks drive the analysis. A typical workflow:

1. Open the workspace in VS Code or JupyterLab.
2. Ensure the virtual environment is active and the dependencies installed.
3. Run `001_process_data.ipynb` to inspect the raw `all.txt` files and produce simple example plots.
4. Run `002_calc_some_metrics.ipynb` to calculate metrics for a single station and see how `plot_metrics` is used.
5. Run `003_final_results.ipynb` to perform batch metric calculations and generate figures for many stations/dates. This notebook uses multiprocessing to parallelize calculation and plotting.

## Files written by the code

- Calculated metrics: `./data/ForbushDecrease/<date>/<station>_metrics-windowsize_<WINDOW_SIZE>.csv`
- Figures: `./figures/<Event>/<date>/<station>_metrics-windowsize_<WINDOW_SIZE>[-<suffix>].pdf`

## Notes, assumptions and caveats

- The `load_data` function expects the `all.txt` files to follow the specific semi-colon separated format used by the author; mismatches may raise parsing errors.
- Metric implementations rely on third-party functions (antropy, nolds, scipy) and some metrics may return arrays (see `mfhurst_b` handling in `calcs.py`). Edge cases (short windows, many NaNs) are handled by skipping or catching exceptions, but results should be inspected.
- The code saves outputs to `./data/` and `./figures/` — ensure those directories are writable.
