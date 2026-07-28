"""
遍历所有同时包含 re3.csv 和 re5.csv 的目录：
1. 读取 re3.csv 第一行的 CID，计算该 CID 所有行 activity_value 的中位数 → first_mean
2. 读取 re5.csv，新增 relative_value = relative_fold * first_mean
3. 替换 relative_log = -log10(relative_value * 1e-6)  即 pIC50
4. 结果写回 re5.csv
"""

import os
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))

for entry in sorted(os.listdir(ROOT)):
    dirpath = os.path.join(ROOT, entry)
    if not os.path.isdir(dirpath):
        continue

    re3_path = os.path.join(dirpath, "re3.csv")
    re5_path = os.path.join(dirpath, "re5.csv")

    if not (os.path.isfile(re3_path) and os.path.isfile(re5_path)):
        continue

    # --- re3: 第一行 CID → 该 CID 所有 activity_value 的中位数 ---
    try:
        re3 = pd.read_csv(re3_path, sep="\t", on_bad_lines="skip")
    except Exception as e:
        print(f"[{entry}] 读取 re3.csv 失败: {e}")
        continue

    if "CID" not in re3.columns or "activity_value" not in re3.columns:
        print(f"[{entry}] re3.csv 缺少 CID 或 activity_value 列，跳过")
        continue

    first_cid = re3["CID"].iloc[0]
    cid_values = re3.loc[re3["CID"] == first_cid, "activity_value"].dropna()
    first_mean = cid_values.median()

    # --- re5: 计算 relative_value 和 pIC50 ---
    data = pd.read_csv(re5_path, sep="\t")
    data["relative_value"] = data["relative_fold"] * first_mean
    data["relative_log"] = (-np.log10(data["relative_value"] * 1e-6)).round(2)

    data.to_csv(re5_path, index=False, sep="\t")
    print(f"[{entry}] CID={first_cid}, first_mean(median)={first_mean:.4f}, rows={len(data)}")
