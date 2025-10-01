#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate lead/lag results for Forbush Decrease events from per-station metric CSVs.

Directory layout (as per your repo):
  data/ForbushDecrease/{date}/
    <station>_metrics-windowsize_130-ewm_alpha_0.15.csv
    <station>_metrics-windowsize_130.csv
  data/ForbushDecrease/summary_derivatives-ewm_alpha_0.15.csv  (opcional, para cobertura)

Outputs:
  outputs/global_rank.csv
  outputs/event_summary.csv
  tables/rank_global.tex
  tables/event_summary.tex
  figures/heatmaps/median_lead_heatmap.png
  figures/violins/lag_violin_<date>.png

Requirements: pandas, numpy, scipy, statsmodels, matplotlib
"""

import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import correlate
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401

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


# ---------------------- CONFIG ----------------------
CFG = {
    # Eventos presentes en la estructura
    "events": ["2023-04-23", "2024-03-24", "2024-05-10"],

    # Carpeta base (ajustada al repo)
    "base_dir": "data/ForbushDecrease",

    # Patrones de archivo (primero intenta ewm, si no, sin ewm)
    "pattern_ewm": "{base}/{date}/{station}_metrics-windowsize_130-ewm_alpha_0.15.csv",
    "pattern_raw": "{base}/{date}/{station}_metrics-windowsize_130.csv",

    # Detección de columnas (candidatas por tipo)
    "time_cols": ["time", "timestamp", "datetime", "date", "index"],
    "count_cols": ["count", "counts", "x", "signal", "raw", "nm_count", "value"],

    # Nombres de invariantes tal como suelen salir en notebooks
    # (ajustar aquí si en los CSVs se usan otras etiquetas)
    "invariants": [
        # Entropy related
        "permutation_entropy",
        "sample_entropy",
        "sampen",
        "approximate_entropy",
        "app_entropy",
        "spectral_entropy",
        "shannon_entropy",

        # Complexity related
        "lzc",
        "lepel_ziv",
        "lempel_ziv",
        "lempel_ziv_complexity",

        # Hurst related
        "hurst_dfa",
        "dfa",
        "dfa_hurst",
        "hurst",
        "mfhurst_b",

        # Fractal dimension related
        "higuchi_fd",
        "katz_fd",
        "petrosian_fd",
        "corr_dim",
        "corr_dimension",
        "correlation_dimension",
    ],
    "unwanted": [
        "window_shape"
    ],

    # Parámetros de derivada y bootstrap
    "alpha": 0.15,  # EWM smoothing
    "W": 130,  # window length (min) para búsqueda local de picos
    "lag_window": 360,  # buscar lags en [-360, 360]
    "winsor_p": 0.01,  # winsor 1-99% para derivadas (0 = off)
    "B": 1000,  # bootstrap replicates
    "block": 30,  # bootstrap block size (min)
    "fdr_alpha": 0.05,

    # Salidas
    "out_dir": "outputs",
    "tables_dir": "tables",
    "fig_dir": "figures",
}

# ------------------ UTILIDADES ----------------------


def _find_first_column(df, candidates):
    # exact match
    for c in candidates:
        if c in df.columns:
            return c

    # case-insensitive
    low = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in low:
            return low[c.lower()]
    return None


def ewm1d(x, alpha):
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    a = float(alpha)
    for i in range(1, len(x)):
        y[i] = a * x[i] + (1 - a) * y[i - 1]
    return y


def central_diff(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 3 or np.all(~np.isfinite(x)):
        return np.zeros_like(x, dtype=float)

    d = np.zeros_like(x, dtype=float)
    d[1:-1] = (x[2:] - x[:-2]) / 2.0
    d[0] = x[1] - x[0]
    d[-1] = x[-1] - x[-2]
    return d


def winsorize(x, p=0.01):
    if p <= 0:
        return x

    lo, hi = np.quantile(x, [p, 1 - p])
    return np.clip(x, lo, hi)


def zscore(x):
    x = np.asarray(x, dtype=float)
    if x.size == 0 or np.all(~np.isfinite(x)):
        return np.zeros_like(x)

    m = np.nanmean(x)
    s = np.nanstd(x, ddof=1)
    if not np.isfinite(s) or s == 0:
        return np.zeros_like(x)

    return (x - m) / s


def xcorr_lag(x, y, max_lag):
    """Lag de máxima correlación (escala relativa); retorna (lag, rho_at_peak)."""
    x = zscore(np.asarray(x, dtype=float))
    y = zscore(np.asarray(y, dtype=float))
    n = len(x)
    if len(y) != n or n < 5:
        return np.nan, np.nan

    full = correlate(y, x, mode="full")
    lags = np.arange(-n + 1, n)
    sel = (lags >= -max_lag) & (lags <= max_lag)
    full = full[sel]
    lags = lags[sel]
    norm = (n - np.abs(lags)).astype(float)
    corr = full / np.nanmax(norm)  # escala aproximada
    i = np.nanargmax(np.abs(corr))
    return int(lags[i]), float(corr[i])


def moving_block_bootstrap(x, block, B):
    n = len(x)
    if n == 0:
        return np.empty((B, 0))

    idx = np.arange(n)
    out = np.empty((B, n), dtype=float)
    for b in range(B):
        cur = []
        while len(cur) < n:
            start = np.random.randint(0, n)
            seg = idx[start : start + block]
            if len(seg) < block:
                seg = np.concatenate([seg, idx[: block - len(seg)]])
            cur.extend(seg.tolist())
        cur = np.array(cur[:n])
        out[b] = x[cur]
    return out


def safe_corr_at_lag(a, b, lag):
    a = zscore(np.asarray(a, dtype=float))
    b = zscore(np.asarray(b, dtype=float))
    if np.isnan(lag):
        return np.nan
    lag = int(lag)

    if lag > 0:
        a2 = a[:-lag]
        b2 = b[lag:]
    elif lag < 0:
        a2 = a[-lag:]
        b2 = b[:lag]
    else:
        a2 = a
        b2 = b

    if len(a2) < 5:
        return np.nan

    C = np.corrcoef(a2, b2)
    return float(C[0, 1])


def onset_from_derivative(dx):
    """Onset robusto con mediana/MAD y persistencia L=5; fallback al mínimo."""
    med = np.median(dx)
    mad = np.median(np.abs(dx - med))
    thr = med - 2 * mad
    below = dx < thr
    L = 5
    if not below.any():
        return int(np.argmin(dx))

    run, t0 = 0, int(np.argmin(dx))
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= L:
            t0 = i - L + 1
            break
    return t0


# ------------------ CARGA Y PARSING ----------------------


def list_station_files(date):
    base = CFG["base_dir"]

    # buscar ambos patrones y unir
    files = glob.glob(f"{base}/{date}/*_metrics-windowsize_130.csv")

    # dict por estación, prefiriendo ewm si hay duplicados
    by_station = {}
    for f in sorted(files):
        st = Path(f).name.split("_metrics", 1)[0]
        if st not in by_station:
            by_station[st] = f
    return by_station  # {station: filepath}


def read_station_df(filepath):
    df = pd.read_csv(filepath)

    # localizar tiempo
    tcol = _find_first_column(df, CFG["time_cols"])
    if tcol is None:
        raise ValueError(
            f"No time column found in {filepath}. Available: {df.columns.tolist()}"
        )

    df = df.sort_values(tcol).reset_index(drop=True)
    df["__time__"] = pd.to_datetime(df[tcol], errors="coerce")

    # localizar conteo original (o serie base)
    ccol = _find_first_column(df, CFG["count_cols"])
    if ccol is None:
        # no hay conteo: no podemos calcular lag respecto a derivada del conteo
        # devolvemos df igualmente; el caller decidirá saltar
        df["__count__"] = np.nan
    else:
        df["__count__"] = pd.to_numeric(df[ccol], errors="coerce")

    return df


def infer_marker_columns(df):
    cols = []
    lower_map = {c.lower(): c for c in df.columns}
    for name in CFG["invariants"]:
        key = name.lower()
        if key in lower_map:
            cols.append(lower_map[key])

    # además, incluye columnas que claramente son métricas (heurística):
    extra = [
        c
        for c in df.columns
        if c not in ["__time__", "__count__"]
        and c not in CFG["time_cols"]
        and c not in CFG["count_cols"]
        and df[c].dtype != "O"
        and c not in CFG["unwanted"]
    ]  # no strings

    # pero evita duplicados
    for c in extra:
        if c not in cols:
            cols.append(c)
    return cols


# ------------------ PIPELINE POR EVENTO ----------------------


def process_event(date):
    files = list_station_files(date)
    if not files:
        print(f"[WARN] No station files for {date}")
        return pd.DataFrame()

    rows = []
    for st, fpath in files.items():
        df = read_station_df(fpath)

        # si no hay conteo usable, saltar estación
        valid_count = df["__count__"].notna().sum()
        if valid_count < 10:
            print(
                f"[WARN] Too few valid count points in {Path(fpath).name} — skipping {st} {date}"
            )
            continue

        # si no hay conteo usable, saltar estación
        if df["__count__"].isna().all():
            print(
                f"[WARN] Missing count column in {Path(fpath).name} — skipping station {st} for {date}"
            )
            continue

        # Forzar que el windows shape sea igual siempre
        if "window_shape" in df.columns:
            df = df[df["window_shape"] == CFG["W"]].reset_index(drop=True)

        print(f"[DEBUG] DF shape for {st} {date} - After: {df.shape}")

        x = df["__count__"].astype(float).to_numpy()

        # suavizado + zscore + derivada para el conteo
        xs = ewm1d(x, CFG["alpha"])
        xz = zscore(xs)
        dx = central_diff(xz)
        if CFG["winsor_p"] > 0:
            dx = winsorize(dx, CFG["winsor_p"])

        t0_idx = onset_from_derivative(dx)
        n = len(dx)
        mask_pre = np.zeros(n, dtype=bool)
        mask_pre[: max(t0_idx, 0)] = True

        # detectar columnas de marcadores (ya calculadas en tus CSVs)
        mcols = infer_marker_columns(df)
        if not mcols:
            print(f"[WARN] No marker columns found in {Path(fpath).name}")
            continue

        for col in mcols:
            # convierte a numérico y a ndarray
            m = pd.to_numeric(df[col], errors="coerce").to_numpy()

            # descarta marcadores sin datos o con muy pocos puntos válidos
            if m.size < 10 or np.count_nonzero(np.isfinite(m)) < 10:
                continue

            # EWM + zscore + derivada, con winsor opcional
            ms = ewm1d(m, CFG["alpha"])
            md = central_diff(zscore(ms))
            if CFG["winsor_p"] > 0:
                md = winsorize(md, CFG["winsor_p"])

            # lag de máxima correlación
            lag, rho = xcorr_lag(md, dx, CFG["lag_window"])

            # bootstrap p-valor unidireccional para la correlación al lag observado
            B = CFG["B"]
            block = CFG["block"]
            bs = moving_block_bootstrap(md, block, B)
            corr_obs = safe_corr_at_lag(md, dx, lag)
            if np.isnan(corr_obs):
                pval = 1.0
            elif corr_obs >= 0:
                pval = float(
                    (
                        np.apply_along_axis(
                            lambda a: safe_corr_at_lag(a, dx, lag), 1, bs
                        )
                        >= corr_obs
                    ).mean()
                )
            else:
                pval = float(
                    (
                        np.apply_along_axis(
                            lambda a: safe_corr_at_lag(a, dx, lag), 1, bs
                        )
                        <= corr_obs
                    ).mean()
                )

            # desfase de picos en ventana ±W
            w = min(CFG["W"], n // 2)
            i_cnt_min = np.argmin(dx[max(0, t0_idx - w) : min(n, t0_idx + w)]) + max(
                0, t0_idx - w
            )
            i_mrk_min = np.argmin(md[max(0, t0_idx - w) : min(n, t0_idx + w)]) + max(
                0, t0_idx - w
            )
            peak_align = i_mrk_min - i_cnt_min

            rows.append(
                {
                    "date": date,
                    "station": st.upper(),
                    "invariant": col,
                    "lag_star": lag,
                    "rho_at_peak": rho,
                    "corr_at_lag": corr_obs,
                    "pval": pval,
                    "peak_align": peak_align,
                }
            )

    res = pd.DataFrame(rows)
    if res.empty:
        return res

    # FDR por invariante (global en el evento)
    res["pval_adj"] = np.nan
    res["significant"] = False
    for inv in res["invariant"].unique():
        mask = res["invariant"] == inv
        p = res.loc[mask, "pval"].values
        rej, p_adj, _, _ = multipletests(p, alpha=CFG["fdr_alpha"], method="fdr_bh")
        res.loc[mask, "pval_adj"] = p_adj
        res.loc[mask, "significant"] = rej.astype(bool)
    return res


# ------------------ AGREGACIONES Y TEX ----------------------


def summarize_global(df):
    # agrega por invariante a través de eventos/estaciones
    g = (
        df.groupby("invariant")
        .agg(
            median_lag=("lag_star", "median"),
            iqr_lag=("lag_star", lambda x: np.subtract(*np.nanpercentile(x, [75, 25]))),
            sig_pct=("significant", lambda x: 100.0 * np.mean(x)),
            n_stations=("station", "nunique"),
        )
        .reset_index()
    )
    # score Rk
    Lref = 120.0
    w1, w2, w3 = 0.4, 0.3, 0.3
    Nstn = df["station"].nunique()
    g["Rk"] = (
        w1 * (np.abs(g["median_lag"]) / Lref).clip(0, 1)
        + w2 * (1 - (g["iqr_lag"] / Lref).clip(0, 1))
        + w3 * (g["n_stations"] / max(Nstn, 1))
    )
    g = g.sort_values("Rk", ascending=False).reset_index(drop=True)
    return g


def summarize_by_event(df):
    e = (
        df.groupby(["date", "invariant"])
        .agg(
            median_lag=("lag_star", "median"),
            sig_pct=("significant", lambda x: 100.0 * np.mean(x)),
        )
        .reset_index()
    )
    return e


def tex_rank_table(g):
    rows = []
    for _, r in g.iterrows():
        rows.append(
            f"{r['invariant'].replace('_', '\\_')} & {r['median_lag']:.1f} & {r['iqr_lag']:.1f} & {r['sig_pct']:.0f} & {r['Rk']:.2f} \\\\"
        )
    body = "\n".join(rows)
    tex = (
        r"""
\begin{table}[t]
\centering
\small
\caption{Global ranking of invariants by robustness score $R_k$ and median lead $\widetilde{\ell}_k$ (min; negative = precedes).}
\label{tab:rank_global}
\begin{tabular}{@{}l r r r r@{}}
\toprule
\textbf{Invariant} & $\widetilde{\ell}_k$ & IQR & Sig.\ stations [\%%] & $R_k$ \\
\midrule
"""
        + body
        + r"""
\bottomrule
\end{tabular}
\end{table}
"""
    )
    return tex


def tex_event_table(e, events):
    # pivot inv × evento con (lag, sig)
    frames = []
    for ev in events:
        sub = e[e["date"] == ev][["invariant", "median_lag", "sig_pct"]].copy()
        sub.columns = ["invariant", f"lag_{ev}", f"sig_{ev}"]
        frames.append(sub)
    if not frames:
        return r"\begin{table}[t]\centering\small\caption{No data}\label{tab:event_summary}\begin{tabular}{@{}l@{}}\toprule No data\\ \bottomrule\end{tabular}\end{table}"

    M = frames[0]
    for sub in frames[1:]:
        M = M.merge(sub, on="invariant", how="outer")
    M = M.fillna(np.nan)

    # filas
    rows_list = []
    for _, r in M.iterrows():
        parts = [r["invariant"].replace("_", r"\_")]
        for ev in events:
            lag = r.get(f"lag_{ev}", np.nan)
            sig = r.get(f"sig_{ev}", np.nan)
            parts.append(f"{lag:.1f} & {sig:.0f}")
        rows_list.append(" & ".join(parts) + r" \\")
    rows = "\n".join(rows_list)

    # encabezados y especificación de columnas
    ev_heads = " & ".join([r"\multicolumn{2}{c}{\textbf{%s}}" % ev for ev in events])
    ev_sub = " & ".join([r"$\widetilde{\ell}_k$ & Sig.\ [\%%]"] * len(events))
    colspec = "@{}l " + " ".join(["r r"] * len(events)) + "@{}"
    endcol = 2 * len(events) + 1  # para \cmidrule(lr){2-endcol}

    # plantilla con % (no choca con llaves de LaTeX)
    tex = r"""\begin{table}[t]
\centering
\small
\caption{Per–event summary: median lead $\widetilde{\ell}_k$ (min; negative = precedes) and percent of stations with significant pre–onset change.}
\label{tab:event_summary}
\begin{tabular}{%s}
\toprule
 & %s \\
\cmidrule(lr){2-%d}
\textbf{Invariant} & %s \\
\midrule
%s
\bottomrule
\end{tabular}
\end{table}
""" % (colspec, ev_heads, endcol, ev_sub, rows)
    return tex


# ------------------ FIGURAS AUXILIARES ----------------------


def heatmap_median_lead(e, events, outpath):
    piv = e.pivot(index="invariant", columns="date", values="median_lag").reindex(
        columns=events
    )
    if piv.empty:
        return
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(piv))))
    im = ax.imshow(piv.values, aspect="auto")
    ax.set_yticks(range(len(piv)))
    ax.set_yticklabels([s.replace("_", " ") for s in piv.index])
    ax.set_xticks(range(len(events)))
    ax.set_xticklabels(events)
    ax.set_title("Median lead by invariant and event (min; negative = precedes)")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if pd.notna(v):
                ax.text(j, i, f"{v:.0f}", ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


def violin_lags(df, date, outpath):
    sub = df[df["date"] == date].copy()
    if sub.empty:
        return
    invs = list(sub["invariant"].unique())
    data = [sub.loc[sub["invariant"] == k, "lag_star"].dropna().values for k in invs]
    labels = [k.replace("_", " ") for k in invs]
    fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(labels))))
    _ = ax.violinplot(data, showmedians=True, vert=False)
    ax.set_yticks(np.arange(1, len(labels) + 1))
    ax.set_yticklabels(labels)
    ax.set_xlabel("lag* (min; negative = precedes)")
    ax.set_title(f"Station-wise lag distributions by invariant — {date}")
    fig.tight_layout()
    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outpath, dpi=180)
    plt.close(fig)


# ------------------ MAIN ----------------------


def main():
    out_dir = Path(CFG["out_dir"])
    out_dir.mkdir(exist_ok=True)
    tab_dir = Path(CFG["tables_dir"])
    tab_dir.mkdir(exist_ok=True)
    fig_dir = Path(CFG["fig_dir"])
    fig_dir.mkdir(exist_ok=True)

    all_res = []
    for date in CFG["events"]:
        print(f"[INFO] Processing event {date}")
        res = process_event(date)
        if res.empty:
            print(f"[WARN] No results for {date}")
            continue
        res.to_csv(out_dir / f"station_results_{date}.csv", index=False)
        all_res.append(res)
        # figuras por evento (violines)
        violin_lags(res, date, fig_dir / "violins" / f"lag_violin_{date}.png")
        print()

    if not all_res:
        print("[ERROR] No events processed. Check column names in your CSVs.")
        return 1

    df = pd.concat(all_res, ignore_index=True)
    g = summarize_global(df)
    e = summarize_by_event(df)
    g.to_csv(out_dir / "global_rank.csv", index=False)
    e.to_csv(out_dir / "event_summary.csv", index=False)

    # tablas LaTeX
    (tab_dir / "rank_global.tex").write_text(tex_rank_table(g))
    (tab_dir / "event_summary.tex").write_text(tex_event_table(e, CFG["events"]))

    # heatmap global
    heatmap_median_lead(
        e, CFG["events"], fig_dir / "heatmaps" / "median_lead_heatmap.png"
    )

    print("[OK] Results written to outputs/, tables/, figures/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
