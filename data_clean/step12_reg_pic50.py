"""
遍历所有包含 reg_origin.csv 的目录：
将 log_value 替换为 pIC50 = -log10(activity_value * 1e-6)，保留2位小数
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

for entry in sorted(os.listdir(ROOT)):
    dirpath = os.path.join(ROOT, entry)
    if not os.path.isdir(dirpath):
        continue

    reg_path = os.path.join(dirpath, "reg_origin.csv")
    if not os.path.isfile(reg_path):
        continue

    try:
        df = pd.read_csv(reg_path, sep="\t")
    except Exception as e:
        print(f"[{entry}] 读取失败: {e}")
        continue

    if "activity_value" not in df.columns:
        print(f"[{entry}] 缺少 activity_value 列，跳过")
        continue

    df["activity_value"] = pd.to_numeric(df["activity_value"], errors="coerce")
    df["log_value"] = (-np.log10(df["activity_value"] * 1e-6)).round(2)
    df.to_csv(reg_path, index=False, sep="\t")
    print(f"[{entry}] 已转换 {len(df)} 行")
