#!/usr/bin/env python3
"""step6_makere_n_statusre.py

Recursively traverse the script directory and all subdirectories to find
all files ending with re5.csv. Keep only four columns: name, smiles,
relative_value, relative_log. Plot the distributions of relative_value
and relative_log using seaborn.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── Configuration (ACS JCIM style) ────────────────────────────────────────
plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "axes.linewidth": 1.5,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "xtick.major.width": 1.2,
    "ytick.major.width": 1.2,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "axes.grid": False,
    "axes.unicode_minus": False,
})

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "status_figures"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── 1. Recursively find all re5.csv files ─────────────────────────────────
re5_files = sorted(SCRIPT_DIR.rglob("re5.csv"))
print(f"Found {len(re5_files)} re5.csv file(s):")
for f in re5_files:
    print(f"  {f.relative_to(SCRIPT_DIR)}")

if not re5_files:
    print("No re5.csv files found; exiting.")
    exit(0)

# ── 2. Read, clean, and save ──────────────────────────────────────────────
all_data = []  # For aggregated plotting

for fpath in re5_files:
    try:
        df = pd.read_csv(fpath, sep="\t")
    except Exception as e:
        print(f"[Skipped] Read failed: {fpath.relative_to(SCRIPT_DIR)} — {e}")
        continue

    # Check required columns
    required_cols = ["name", "smiles", "relative_value", "relative_log"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[Skipped] {fpath.relative_to(SCRIPT_DIR)} missing columns: {missing}")
        continue

    # Keep only four columns
    df = df[required_cols].copy()

    # Write back to original file (tab-separated)
    df.to_csv(fpath, sep="	", index=False)
    print(f"[Done] {fpath.relative_to(SCRIPT_DIR)} — {len(df)} rows")

    # Append to aggregated data, tag source directory
    df["source"] = fpath.parent.name
    all_data.append(df)

# ── 3. Merge all data ─────────────────────────────────────────────────────
if not all_data:
    print("No valid data for plotting; exiting.")
    exit(0)
combined = pd.concat(all_data, ignore_index=True)
print(f"\nAggregated data: {len(combined)} rows from {combined['source'].nunique()} source(s)")

# ── 4. Seaborn visualization (ACS JCIM style) ─────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left panel: relative_value distribution
ax1 = axes[0]
sns.histplot(
    combined["relative_value"],
    bins=80,
    kde=True,
    stat="density",
    color="#4472C4",
    edgecolor="#333333",
    linewidth=0.5,
    ax=ax1,
)
ax1.set_title("Distribution of relative_value", fontweight="bold")
ax1.set_xlabel("relative_value")
ax1.set_ylabel("Density")

# Right panel: relative_log distribution
ax2 = axes[1]
sns.histplot(
    combined["relative_log"],
    bins=80,
    kde=True,
    stat="density",
    color="#ED7D31",
    edgecolor="#333333",
    linewidth=0.5,
    ax=ax2,
)
ax2.set_title("Distribution of relative_log", fontweight="bold")
ax2.set_xlabel("relative_log")
ax2.set_ylabel("Density")

fig.suptitle("re5.csv — relative_value & relative_log Distribution", fontsize=20, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUTPUT_DIR / "relative_value_log_distribution.png")
plt.close()
print(f"\nDistribution plot saved to: {OUTPUT_DIR / 'relative_value_log_distribution.png'}")

# ── 5. Faceted plots by source directory (manual grid: HDAC1→11, SIRT1→7, SIRT starts a new row) ──
def _plot_by_source_grid(data, x_col, *, bins=40, kde=True, stat="count",
                         color="#4472C4", xlabel=None, ylabel="Count",
                         title="", outpath=None, xlim=None):
    """Faceted plotting on a manual N_cols × N_rows grid, ordered HDAC1→11
then SIRT1→7. SIRT1 always starts a new row."""
    # Sort order: HDAC1→11, SIRT1→7
    hdac_order = [f"HDAC{i}" for i in range(1, 12)]
    sirt_order = [f"SIRT{i}" for i in range(1, 8)]
    hdac_sources = [s for s in hdac_order if s in data["source"].unique()]
    sirt_sources = [s for s in sirt_order if s in data["source"].unique()]

    NCOLS = 3
    hdac_rows = int(np.ceil(len(hdac_sources) / NCOLS))
    sirt_rows = int(np.ceil(len(sirt_sources) / NCOLS))
    nrows = hdac_rows + sirt_rows  # HDAC and SIRT each occupy their own rows

    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 3.5 * 1.3, nrows * 3.5))
    axes = np.atleast_2d(axes)

    # HDAC: occupy the first hdac_rows rows
    for idx, src in enumerate(hdac_sources):
        r, c = divmod(idx, NCOLS)
        ax = axes[r, c]
        subset = data[data["source"] == src]
        if len(subset) > 0:
            sns.histplot(
                subset[x_col], bins=bins, kde=kde, stat=stat,
                color=color, edgecolor="#333333", linewidth=0.5, ax=ax,
            )
            if xlim is not None:
                ax.set_xlim(xlim)
        ax.set_title(src, fontweight="bold", fontsize=14)

    # SIRT: start from row hdac_rows
    for idx, src in enumerate(sirt_sources):
        r = hdac_rows + idx // NCOLS
        c = idx % NCOLS
        ax = axes[r, c]
        subset = data[data["source"] == src]
        if len(subset) > 0:
            sns.histplot(
                subset[x_col], bins=bins, kde=kde, stat=stat,
                color=color, edgecolor="#333333", linewidth=0.5, ax=ax,
            )
            if xlim is not None:
                ax.set_xlim(xlim)
        ax.set_title(src, fontweight="bold", fontsize=14)

    # Set xlabel for all non-empty subplots; ylabel for first column only
    for r in range(nrows):
        for c in range(NCOLS):
            ax = axes[r, c]
            if ax is not None and ax.has_data():
                ax.set_xlabel(xlabel if xlabel else x_col)
    for r in range(nrows):
        ax = axes[r, 0]
        if ax is not None and ax.has_data():
            ax.set_ylabel(ylabel, fontweight="bold")

    # Hide empty subplots (trailing slots in HDAC rows + SIRT rows)
    for r in range(nrows):
        for c in range(NCOLS):
            ax = axes[r, c]
            if ax is not None and not ax.has_data():
                ax.set_visible(False)

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if outpath:
        fig.savefig(outpath)
        print(f"Faceted plot saved to: {outpath}")
    plt.close(fig)

if combined["source"].nunique() > 1:
    # relative_log faceted by source
    _plot_by_source_grid(
        combined, "relative_log",
        bins=40, kde=True, stat="count", color="#ED7D31",
        xlabel="$pIC_{50}$", ylabel="Count",
        title="relative_log Distribution by Source Directory",
        outpath=OUTPUT_DIR / "relative_log_by_source.png",
    )

    # relative_value faceted by source
    _plot_by_source_grid(
        combined, "relative_value",
        bins=40, kde=True, stat="density", color="#4472C4",
        xlabel="Relative Value", ylabel="Density",
        title="relative_value Distribution by Source Directory",
        outpath=OUTPUT_DIR / "relative_value_by_source.png",
    )

# ── 6. Faceted plots for reg_origin.csv by source (manual grid, same ordering) ─
reg_files = sorted(SCRIPT_DIR.rglob("reg_origin.csv"))
print(f"\nFound {len(reg_files)} reg_origin.csv file(s)")

if reg_files:
    reg_data = []
    for fpath in reg_files:
        try:
            df = pd.read_csv(fpath, sep="\t")
        except Exception as e:
            print(f"[Skipped] Read failed: {fpath.relative_to(SCRIPT_DIR)} — {e}")
            continue

        if "log_value" not in df.columns:
            print(f"[Skipped] {fpath.relative_to(SCRIPT_DIR)} missing column: log_value")
            continue

        df = df[["name", "smiles", "log_value"]].copy()
        df["source"] = fpath.parent.name
        reg_data.append(df)
        print(f"[Read] {fpath.relative_to(SCRIPT_DIR)} — {len(df)} rows")

    if reg_data:
        reg_combined = pd.concat(reg_data, ignore_index=True)
        print(f"reg_origin aggregated data: {len(reg_combined)} rows from {reg_combined['source'].nunique()} source(s)")

        _plot_by_source_grid(
            reg_combined, "log_value",
            bins=40, kde=True, stat="count", color="#4472C4",
            xlabel="$pIC_{50}$", ylabel="Count",
            title="log_value Distribution by Source Directory (reg_origin.csv)",
            outpath=OUTPUT_DIR / "log_value_by_source_reg_origin.png",
        )
    else:
        print("reg_origin.csv has no valid data; skipping plot.")
else:
    print("No reg_origin.csv files found; skipping.")

print("\n=== All processing completed. ===")
