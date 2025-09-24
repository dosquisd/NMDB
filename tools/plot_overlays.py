#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Overlay plots for FD events with a fixed station across all events.

Inputs:
- data/ForbushDecrease/<date>/<station>_metrics-windowsize_130-ewm_alpha_0.15.csv
  (o ..._windowsize_130.csv)
- outputs/station_results_<date>.csv  (solo si usas la selección automática)

Outputs:
- figures/overlays/overlay_<date>_<station>.png
- figures/overlays/grid_overlays_<date>_<station>.png
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import os
import scienceplots  # noqa: F401
import seaborn as sns
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

# ========= CONFIG =========
BASE = Path(".")
DATA_DIR = BASE / "data" / "ForbushDecrease"
OUT_DIR  = BASE / "figures" / "overlays"; OUT_DIR.mkdir(parents=True, exist_ok=True)

EVENTS = ["2023-04-23", "2024-03-24", "2024-05-10"]

# —— estación fija para todos los eventos ——
FIXED_STATION = "CALG"      # <-- cámbiala por "JBGO", "DOMC", "NAIN", "NEWK", "APTY", "OULU"
                            # "SOPO", etc. 

# parámetros coherentes con generate_results.py
ALPHA = 0.00001#0.15
WINSOR_P = 0.001
BOOT_B = 500
BOOT_BLOCK = 30

# marcadores del panel (ajusta nombres si las columnas son otras)
PANEL_MARKERS = ["katz_fd", "hurst_dfa", "shannon_entropy", "sample_entropy", "lzc"]
ALIASES = {
    "hurst": "hurst_dfa",
    "dfa_hurst": "hurst_dfa",
    "lempel_ziv": "lzc",
    "lempel_ziv_complexity": "lzc",
    "sampen": "sample_entropy",
    "approximate_entropy": "sample_entropy",
    "ap_entropy": "sample_entropy",
}

TIME_CAND  = ["time","timestamp","datetime","date","index"]
COUNT_CAND = ["count","counts","x","signal","raw","nm_count","value"]

# ========= helpers =========
def _find_col(df, cands):
    for c in cands:
        if c in df.columns: return c
    low = {c.lower(): c for c in df.columns}
    for c in cands:
        if c.lower() in low: return low[c.lower()]
    return None

def ewm1d(x, alpha):
    y = np.empty_like(x, dtype=float); y[0] = x[0]; a=float(alpha)
    for i in range(1, len(x)): y[i] = a*x[i] + (1-a)*y[i-1]
    return y

def zscore(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(~np.isfinite(x)): return np.zeros_like(x)
    m = np.nanmean(x); s = np.nanstd(x, ddof=1)
    if not np.isfinite(s) or s == 0: return np.zeros_like(x)
    return (x - m) / s

def central_diff(x):
    x = np.asarray(x, dtype=float); n = x.size
    if n < 3 or np.all(~np.isfinite(x)): return np.zeros_like(x, float)
    d = np.zeros_like(x, float)
    d[1:-1] = (x[2:] - x[:-2]) / 2.0; d[0] = x[1]-x[0]; d[-1] = x[-1]-x[-2]
    return d

def winsorize(x, p=0.01):
    if p <= 0: return x
    lo, hi = np.quantile(x, [p, 1-p]); return np.clip(x, lo, hi)

def moving_block_bootstrap(x, block, B):
    n = len(x)
    if n == 0: return np.empty((B,0))
    idx = np.arange(n); out = np.empty((B, n), float)
    for b in range(B):
        cur = []
        while len(cur) < n:
            start = np.random.randint(0, n)
            seg = idx[start:start+block]
            if len(seg) < block: seg = np.concatenate([seg, idx[:block-len(seg)]])
            cur.extend(seg.tolist())
        out[b] = x[np.array(cur[:n])]
    return out

def load_station_df(date, station):
    patt1 = DATA_DIR / date / f"{station}_metrics-windowsize_130-ewm_alpha_0.15.csv"
    patt2 = DATA_DIR / date / f"{station}_metrics-windowsize_130.csv"
    f = patt1 if patt1.exists() else patt2
    if not f.exists():
        return None
    df = pd.read_csv(f)
    tcol = _find_col(df, TIME_CAND); ccol = _find_col(df, COUNT_CAND)
    if tcol is None or ccol is None: return None
    df = df.sort_values(tcol).reset_index(drop=True)
    df["__t__"] = pd.to_datetime(df[tcol], errors="coerce")
    df["__x__"] = pd.to_numeric(df[ccol], errors="coerce")
    return df

def marker_columns(df):
    cols = []
    for m in PANEL_MARKERS:
        if m in df.columns: cols.append(m); continue
        alts = [k for k,v in ALIASES.items() if v == m and k in df.columns]
        if alts: cols.append(alts[0])
    # filtra NaN-only
    cols = [c for c in cols if pd.to_numeric(df[c], errors="coerce").notna().any()]
    return cols

def prepare_derivatives(df, col):
    x = pd.to_numeric(df[col], errors="coerce").to_numpy()
    xs = ewm1d(x, ALPHA)
    xd = central_diff(zscore(xs))
    if WINSOR_P > 0: xd = winsorize(xd, WINSOR_P)
    return xd

# ========= plotting =========
def plot_overlay(date, station, markers=PANEL_MARKERS, save_panel=True):
    df = load_station_df(date, station)
    if df is None:
        print(f"[WARN] Missing data for {station} on {date}. Skipping.")
        return
    dx = prepare_derivatives(df, "__x__")
    t  = df["__t__"].to_numpy()

    mcols = marker_columns(df)
    use = []
    for m in markers:
        if m in mcols: use.append(m)
        else:
            alts = [k for k,v in ALIASES.items() if v == m and k in mcols]
            if alts: use.append(alts[0])

    if not use:
        print(f"[WARN] No panel markers found for {station} on {date}.")
        return

    # figura con todos los marcadores
    fig, ax = plt.subplots(figsize=(15, 3.5))
    ax.plot(t, dx, lw=1.2, label="count deriv")
    for m in use:
        dm = prepare_derivatives(df, m)
        bs = moving_block_bootstrap(dm, BOOT_BLOCK, BOOT_B)
        lo = np.nanpercentile(bs, 2.5, axis=0)
        hi = np.nanpercentile(bs, 97.5, axis=0)
        ax.plot(t, dm, lw=0.9, label=m.replace("_"," "))
        ax.fill_between(t, lo, hi, alpha=0.12)
    ax.set_title(f"{date} — {station}: derivative overlays")
    ax.set_xlabel("time"); ax.set_ylabel("standardized derivative")
    ax.legend(ncol=3, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"overlay_{date}_{station}.png", dpi=180)
    plt.close(fig)

    # panel 2×2 con 4 primeros
    if save_panel:
        m4 = use[:4]
        r = int(np.ceil(len(m4)/2))
        fig, axes = plt.subplots(r, 2, figsize=(15, 3.5*r), squeeze=False)
        axes = axes.ravel()
        for i, m in enumerate(m4):
            dm = prepare_derivatives(df, m)
            bs = moving_block_bootstrap(dm, BOOT_BLOCK, BOOT_B)
            lo = np.nanpercentile(bs, 2.5, axis=0)
            hi = np.nanpercentile(bs, 97.5, axis=0)
            ax = axes[i]
            ax.plot(t, dx, lw=1.2, label="count deriv")
            ax.plot(t, dm, lw=0.9, label=m.replace("_"," "))
            ax.fill_between(t, lo, hi, alpha=0.12)
            ax.set_title(m.replace("_"," "))
            ax.set_xlabel("time"); ax.set_ylabel("derivative")
        for j in range(i+1, len(axes)):
            axes[j].axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3)
        fig.tight_layout(rect=[0,0,1,0.95])
        fig.savefig(OUT_DIR / f"grid_overlays_{date}_{station}.png", dpi=180)
        plt.close(fig)

def make_overlays_fixed_station(station=None, events=None):
    if station is None:
        station = FIXED_STATION
    if events is None:
        events = EVENTS
    for date in events:
        print(f"[overlay] {date} — {station}")
        plot_overlay(date, station)

if __name__ == "__main__":
    # genera overlays para la estación fija en los 3 eventos
    make_overlays_fixed_station()
