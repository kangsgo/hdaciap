import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

"""
使用 python 和 pandas 写一个 step8_statusre.py 的脚本，遍历所有目录，
如果包含 re3.csv 结尾的文件则依次执行：

1. 仅保留 CID,name,smiles,activity_value 列的数据，并新增一列 log_value，
   数值为 activity_value 取 log 后保留 2 位小数。保存为 reg.csv，
   分隔符为 \t，输出到对应目录下。
2. 使用 seaborn 依据 reg.csv 的 log_value 列进行作图，图片保存在对应目录下。
"""

CHUNK_SIZE = 50000
KEEP_COLUMNS = ["CID", "name", "smiles", "activity_value"]


def _safe_log_series(value_series: pd.Series) -> pd.Series:
    """
    对 activity_value 计算自然对数：
    - 非数值、空值、<=0 的值统一转为 NaN
    - 保留 2 位小数
    """
    numeric_values = pd.to_numeric(value_series, errors="coerce")
    log_values = pd.Series(np.nan, index=value_series.index, dtype="float64")

    valid_mask = numeric_values > 0
    log_values.loc[valid_mask] = np.log(numeric_values.loc[valid_mask])

    return log_values.round(2)


def _plot_log_values(log_values: List[float], output_image: str) -> None:
    """
    使用 seaborn 对 log_value 作图并保存。
    采用 histplot + kde 展示单列数值分布。
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 6))

    valid_values = [v for v in log_values if pd.notna(v)]

    if valid_values:
        sns.histplot(valid_values, bins=30, kde=True, color="#2c7fb8")
        plt.title("Distribution of log_value")
        plt.xlabel("log_value")
        plt.ylabel("Count")
    else:
        plt.text(0.5, 0.5, "No valid log_value data", ha="center", va="center", fontsize=14)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(output_image, dpi=300, bbox_inches="tight")
    plt.close()


def process_re3_to_reg(input_file: str) -> None:
    """
    处理单个 re3.csv：
    1) 仅保留 CID,name,smiles,activity_value
    2) 新增 log_value = log(activity_value)，保留 2 位小数
    3) 保存为同目录下 reg.csv（\\t 分隔）
    4) 基于 log_value 作图并保存为 png
    """
    output_dir = os.path.dirname(input_file)
    output_file = os.path.join(output_dir, "reg.csv")
    output_image = os.path.join(output_dir, "log_value.png")

    header_written = False
    collected_log_values: List[float] = []

    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
        missing_cols = [col for col in KEEP_COLUMNS if col not in chunk.columns]
        if missing_cols:
            continue

        work = chunk.loc[:, KEEP_COLUMNS].copy()
        work["log_value"] = _safe_log_series(work["activity_value"])

        collected_log_values.extend(work["log_value"].dropna().tolist())

        mode = "w" if not header_written else "a"
        work.to_csv(output_file, sep="\t", index=False, mode=mode, header=not header_written)
        header_written = True

    if not header_written:
        pd.DataFrame(columns=KEEP_COLUMNS + ["log_value"]).to_csv(output_file, sep="\t", index=False)

    _plot_log_values(collected_log_values, output_image)

    print(f"[完成] {input_file} -> {output_file}")
    print(f"  - 输出图像: {output_image}")
    print(f"  - 有效 log_value 数量: {len(collected_log_values)}")


def main() -> None:
    root_dir = os.getcwd()

    for current_dir, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith("re3.csv"):
                input_path = os.path.join(current_dir, file_name)
                process_re3_to_reg(input_path)


if __name__ == "__main__":
    main()
