#!/usr/bin/env python3
"""step9_makere_n_statusre.py

递归遍历脚本所在目录及其所有子目录，查找所有以 re5.csv 结尾的文件。
只保留 name, smiles, relative_value, relative_log 四列。
使用 seaborn 对 relative_value 和 relative_log 分别绘图，查看值的分布。
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ── 配置 (ACS JCIM 风格) ──────────────────────────────────────────────────
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

# ── 1. 递归查找所有 re5.csv ────────────────────────────────────────────────
re5_files = sorted(SCRIPT_DIR.rglob("re5.csv"))
print(f"共找到 {len(re5_files)} 个 re5.csv 文件：")
for f in re5_files:
    print(f"  {f.relative_to(SCRIPT_DIR)}")

if not re5_files:
    print("未找到任何 re5.csv 文件，退出。")
    exit(0)

# ── 2. 读取、清洗、保存 ────────────────────────────────────────────────────
all_data = []  # 用于汇总绘图

for fpath in re5_files:
    try:
        df = pd.read_csv(fpath, sep="\t")
    except Exception as e:
        print(f"[跳过] 读取失败: {fpath.relative_to(SCRIPT_DIR)} — {e}")
        continue

    # 检查必需列
    required_cols = ["name", "smiles", "relative_value", "relative_log"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[跳过] {fpath.relative_to(SCRIPT_DIR)} 缺少列: {missing}")
        continue

    # 只保留四列
    df = df[required_cols].copy()

    # 保存回原文件（保持 tab 分隔）
    df.to_csv(fpath, sep="	", index=False)
    print(f"[完成] {fpath.relative_to(SCRIPT_DIR)}  — {len(df)} 行")

    # 加入汇总数据，标记来源目录
    df["source"] = fpath.parent.name
    all_data.append(df)

# ── 3. 合并所有数据 ────────────────────────────────────────────────────────
if not all_data:
    print("没有有效数据可供绘图，退出。")
    exit(0)
combined = pd.concat(all_data, ignore_index=True)
print(f"\n汇总数据共 {len(combined)} 行，来源: {combined['source'].nunique()} 个目录")

# ── 4. Seaborn 可视化 (ACS JCIM 风格) ────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# 左图：relative_value 分布
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

# 右图：relative_log 分布
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
print(f"\n分布图已保存至: {OUTPUT_DIR / 'relative_value_log_distribution.png'}")

# ── 5. 按来源目录的分面图 (手动网格：HDAC1→11, SIRT1→7, SIRT1 另起一行) ────
def _plot_by_source_grid(data, x_col, *, bins=40, kde=True, stat="count",
                         color="#4472C4", xlabel=None, ylabel="Count",
                         title="", outpath=None, xlim=None):
    """在手动的 6 列 × N 行网格中，按 HDAC1→11 然后 SIRT1→7 的顺序分面绘图。
    SIRT1 强制另起一行。"""
    # 排序：HDAC1→11, SIRT1→7
    hdac_order = [f"HDAC{i}" for i in range(1, 12)]
    sirt_order = [f"SIRT{i}" for i in range(1, 8)]
    hdac_sources = [s for s in hdac_order if s in data["source"].unique()]
    sirt_sources = [s for s in sirt_order if s in data["source"].unique()]

    NCOLS = 3
    hdac_rows = int(np.ceil(len(hdac_sources) / NCOLS))
    sirt_rows = int(np.ceil(len(sirt_sources) / NCOLS))
    nrows = hdac_rows + sirt_rows  # HDAC 和 SIRT 各自独立分行

    fig, axes = plt.subplots(nrows, NCOLS, figsize=(NCOLS * 3.5 * 1.3, nrows * 3.5))
    axes = np.atleast_2d(axes)

    # HDAC：占前 hdac_rows 行
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

    # SIRT：从第 hdac_rows 行开始
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

    # 设置所有有数据的子图 xlabel，第一列 ylabel
    for r in range(nrows):
        for c in range(NCOLS):
            ax = axes[r, c]
            if ax is not None and ax.has_data():
                ax.set_xlabel(xlabel if xlabel else x_col)
    for r in range(nrows):
        ax = axes[r, 0]
        if ax is not None and ax.has_data():
            ax.set_ylabel(ylabel, fontweight="bold")

    # 隐藏无数据的子图（HDAC 行末尾空位 + SIRT 行末尾空位）
    for r in range(nrows):
        for c in range(NCOLS):
            ax = axes[r, c]
            if ax is not None and not ax.has_data():
                ax.set_visible(False)

    fig.suptitle(title, fontsize=20, fontweight="bold", y=1.02)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    if outpath:
        fig.savefig(outpath)
        print(f"分面图已保存至: {outpath}")
    plt.close(fig)

if combined["source"].nunique() > 1:
    # relative_log 按来源分面
    _plot_by_source_grid(
        combined, "relative_log",
        bins=40, kde=True, stat="count", color="#ED7D31",
        xlabel="$pIC_{50}$", ylabel="Count",
        title="relative_log Distribution by Source Directory",
        outpath=OUTPUT_DIR / "relative_log_by_source.png",
    )

    # relative_value 按来源分面
    _plot_by_source_grid(
        combined, "relative_value",
        bins=40, kde=True, stat="density", color="#4472C4",
        xlabel="Relative Value", ylabel="Density",
        title="relative_value Distribution by Source Directory",
        outpath=OUTPUT_DIR / "relative_value_by_source.png",
    )

# ── 6. reg_origin.csv 按来源分面图 (手动网格，同排序) ────────────────────────
reg_files = sorted(SCRIPT_DIR.rglob("reg_origin.csv"))
print(f"\n共找到 {len(reg_files)} 个 reg_origin.csv 文件")

if reg_files:
    reg_data = []
    for fpath in reg_files:
        try:
            df = pd.read_csv(fpath, sep="\t")
        except Exception as e:
            print(f"[跳过] 读取失败: {fpath.relative_to(SCRIPT_DIR)} — {e}")
            continue

        if "log_value" not in df.columns:
            print(f"[跳过] {fpath.relative_to(SCRIPT_DIR)} 缺少列: log_value")
            continue

        df = df[["name", "smiles", "log_value"]].copy()
        df["source"] = fpath.parent.name
        reg_data.append(df)
        print(f"[读取] {fpath.relative_to(SCRIPT_DIR)} — {len(df)} 行")

    if reg_data:
        reg_combined = pd.concat(reg_data, ignore_index=True)
        print(f"reg_origin 汇总数据共 {len(reg_combined)} 行，来源: {reg_combined['source'].nunique()} 个目录")

        _plot_by_source_grid(
            reg_combined, "log_value",
            bins=40, kde=True, stat="count", color="#4472C4",
            xlabel="$pIC_{50}$", ylabel="Count",
            title="log_value Distribution by Source Directory (reg_origin.csv)",
            outpath=OUTPUT_DIR / "log_value_by_source_reg_origin.png",
        )
    else:
        print("reg_origin.csv 无有效数据，跳过绘图。")
else:
    print("未找到任何 reg_origin.csv 文件，跳过。")

print("\n=== 全部处理完成 ===")
