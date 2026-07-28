#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遍历所有目录，查找文件名以 molecules.csv 结尾的文件并清洗数据：
1. 读取 activity 列：
   - 若 activity 是 "Active"/"Inactive"，直接保留；
   - 若 activity 不是上述值，则仅当 activity_name 在 {"IC50", "EC50", "Ki"} 时保留，否则删除。
2. 仅对“activity 不是 Active/Inactive 且被保留”的数据处理 activity_value：
   - activity_value 为空或非数值 -> 删除该行
   - activity_value >= 10 -> activity 设为 "Inactive"
   - activity_value < 10  -> activity 设为 "Active"
3. 在对应目录下保存为 classic1.csv

用法：
    python step2_clean.py
"""

import os
import pandas as pd


ALLOWED_ACTIVITY = {"Active", "Inactive"}
ALLOWED_ACTIVITY_NAME = {"IC50", "EC50", "Ki"}


def read_table_auto(input_path: str) -> pd.DataFrame:
    """
    优先按制表符读取；若失败则回退到逗号分隔。
    """
    try:
        return pd.read_csv(input_path, sep="\t")
    except Exception:
        return pd.read_csv(input_path)


def process_file(input_path: str, output_path: str) -> None:
    """
    清洗单个 molecules.csv 文件并输出 classic1.csv。
    """
    try:
        df = read_table_auto(input_path)
    except Exception as e:
        print(f"  [错误] 读取失败，跳过: {input_path} | {e}")
        return

    required_cols = {"activity", "activity_name", "activity_value"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"  [错误] 缺少必要列 {missing}，跳过: {input_path}")
        return

    total_rows = len(df)

    # 统一转为字符串后去首尾空白，避免空值/类型导致判断异常
    activity = df["activity"].astype(str).str.strip()
    activity_name = df["activity_name"].astype(str).str.strip()

    # 规则1：activity 为 Active/Inactive 的行直接保留
    mask_valid_activity = activity.isin(ALLOWED_ACTIVITY)
    df_standard = df.loc[mask_valid_activity].copy()

    # activity 非 Active/Inactive 时，仅保留 activity_name 在允许集合中的行
    mask_non_standard = ~mask_valid_activity
    mask_valid_name = activity_name.isin(ALLOWED_ACTIVITY_NAME)
    df_non_standard = df.loc[mask_non_standard & mask_valid_name].copy()

    # 仅对非标准 activity 的保留行进行 activity_value 数值化与过滤
    df_non_standard["activity_value"] = pd.to_numeric(
        df_non_standard["activity_value"], errors="coerce"
    )
    df_non_standard = df_non_standard.dropna(subset=["activity_value"])

    # 按 activity_value 给非标准 activity 行重标注 activity
    df_non_standard["activity"] = "Active"
    df_non_standard.loc[df_non_standard["activity_value"] >= 10, "activity"] = "Inactive"

    # 合并两部分结果：标准行 + 处理后的非标准行
    df_kept = pd.concat([df_standard, df_non_standard], ignore_index=True)

    # 删除 activity_value 为空且 activity 为 "Active" 的行
    mask_empty_value_active = df_kept["activity"].eq("Active") & df_kept["activity_value"].isna()
    df_kept = df_kept.loc[~mask_empty_value_active]

    kept_rows = len(df_kept)
    deleted_rows = total_rows - kept_rows

    # 输出为 classic1.csv（制表符分隔，保持与原始数据风格一致）
    df_kept.to_csv(output_path, index=False, sep="\t", encoding="utf-8")
    print(f"    保留: {kept_rows} 行, 删除: {deleted_rows} 行 -> {output_path}")


def find_target_files(base_dir: str):
    """
    递归遍历目录，返回所有文件名以 molecules.csv 结尾的文件路径。
    例如：molecules.csv、HDAC1_molecules.csv、SIRT2_molecules.csv
    """
    targets = []
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith("molecules.csv"):
                targets.append(os.path.join(root, filename))
    return sorted(targets)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    all_files = find_target_files(base_dir)

    if not all_files:
        print("[信息] 未找到任何以 molecules.csv 结尾的文件。")
        return

    print(f"[信息] 共找到 {len(all_files)} 个文件待处理。\n")
    for input_path in all_files:
        output_path = os.path.join(os.path.dirname(input_path), "classic1.csv")
        print(f"处理: {input_path}")
        process_file(input_path, output_path)

    print("\n[完成] 所有文件处理完毕。")


if __name__ == "__main__":
    main()
