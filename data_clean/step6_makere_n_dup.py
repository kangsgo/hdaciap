import os

import numpy as np
import pandas as pd


"""
**角色设定**：
你是一位精通 Python 和 Pandas 的数据处理专家。请编写一个名为 `step6_makere_n_dup.py` 的 Python 脚本。

**核心任务**：
递归遍历当前脚本所在目录及其所有子目录，查找所有以 `re2.csv` 结尾的文件，并按照指定的业务逻辑进行数据清洗与特征计算，最终输出处理后的结果。

**详细处理逻辑**：
对找到的每一个 `re2.csv` 文件，执行以下数据处理流程：

1. **数据预处理**：
   - 读取 CSV 文件。
   - 新增一列 `log_value`，其值为 `activity_value` 列取自然对数（`np.log`）后保留 2 位小数。
2. **确定基准值（按 `AID` 分组）**：
   - 按 `AID` 对数据进行分组。
   - 在每一组内，找到 `index` 列值最小的那一行作为基准行。
   - 提取该基准行的 `CID` 作为该组的 `relative_cid` 基准值。
   - 提取该基准行的 `activity_value` 作为该组的 `relative_fold` 基准值（即分母）。
3. **计算与新增列**：
   - 新增列 `relative_cid`：将该组基准行的 `CID` 值广播填充到该组的所有行。
   - 新增列 `relative_fold`：使用当前行的 `activity_value` 除以该组基准行的 `activity_value`，并对结果保留 2 位小数（使用 `.round(2)`）。
4. **结果保存**：
   - 将处理后的 DataFrame 保存为与原文件同目录下的 `re3.csv`。
   - 保存时不保留 Pandas 索引（`index=False`）。
"""

def process_re2(input_file: str) -> None:
    """处理单个 re2.csv 文件，按 AID 分组计算 relative_cid 和 relative_fold。"""
    output_file = os.path.join(os.path.dirname(input_file), "re3.csv")

    df = pd.read_csv(input_file, sep="\t", low_memory=False)

    if df.empty:
        print(f"  [SKIP] {input_file}: empty file")
        return

    required_cols = ["activity_value", "AID", "CID"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"  [SKIP] {input_file}: missing columns: {missing}")
        return

    # Step 1: log_value = np.log(activity_value)，保留 2 位小数
    df["activity_value"] = pd.to_numeric(df["activity_value"], errors="coerce")
    df = df[df["activity_value"].notna() & (df["activity_value"] > 0)].copy()
    df["log_value"] = np.log(df["activity_value"]).round(2)

    # 重置索引使 index 成为列，用于确定每组中"最小 index"的基准行
    df = df.reset_index(drop=True)

    # Step 2: 按 AID 分组，找到每个组内 index 最小的行作为基准行
    baseline = df.loc[df.groupby("AID").apply(lambda g: g.index.min(), include_groups=False)]

    baseline_map_cid = dict(zip(baseline["AID"], baseline["CID"]))
    baseline_map_value = dict(zip(baseline["AID"], baseline["activity_value"]))

    # Step 3-4: 广播 relative_cid 和计算 relative_fold
    df["relative_cid"] = df["AID"].map(baseline_map_cid)
    df["relative_fold"] = df.apply(
        lambda row: round(row["activity_value"] / baseline_map_value[row["AID"]], 2),
        axis=1,
    )

    df.to_csv(output_file, sep="\t", index=False)
    print(f"  Saved {len(df)} rows to {output_file}")


def main() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))

    for current_dir, _, files in os.walk(root_dir):
        if "re2.csv" in files:
            input_path = os.path.join(current_dir, "re2.csv")
            output_path = os.path.join(current_dir, "re3.csv")
            print(f"Processing: {input_path}")
            process_re2(input_path)


if __name__ == "__main__":
    main()
