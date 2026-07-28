#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
遍历当前目录及所有子目录中的 classic1.csv 文件，按 CID 去重并生成 classic2.csv。

处理规则：
1. 仅处理文件名为 classic1.csv 的 TSV 文件（制表符分隔）。
2. 统计每个 CID 的出现次数，以及 activity 列中 Active / Inactive 的数量。
3. 若某个 CID 出现次数 >= 2：
   - 如果 Active 数量多于 Inactive，则最终保留的那条记录 activity 改为 Active
   - 否则改为 Inactive
4. 对于重复 CID，只保留一条记录（保留第一次出现的那条记录），其余重复行删除。
5. 结果保存为同目录下的 classic2.csv。

说明：
- 使用 pandas 一次性读取整个文件，不再分块读取。
"""

import os

import pandas as pd


def process_file(input_path, output_path):
    """
    处理单个 classic1.csv 文件，生成 classic2.csv。
    """
    try:
        df = pd.read_csv(input_path, sep="\t", dtype=str, encoding="utf-8")
    except Exception as e:
        print(f"  [错误] 读取失败: {input_path} -> {e}")
        return

    required_cols = ["CID", "activity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  [错误] 缺少必要列 {missing_cols}，跳过: {input_path}")
        return

    columns = list(df.columns)

    # 标准化字段
    df["CID"] = df["CID"].fillna("").astype(str).str.strip()
    df["activity"] = df["activity"].fillna("").astype(str).str.strip()

    # 仅处理有效 CID
    valid_df = df[df["CID"] != ""].copy()
    if valid_df.empty:
        print(f"  [信息] 未找到有效 CID 数据，跳过: {input_path}")
        return

    # 统计每个 CID 出现次数
    cid_total_count = valid_df.groupby("CID").size()

    # 统计 Active / Inactive 数量
    activity_stat = (
        valid_df[valid_df["activity"].isin(["Active", "Inactive"])]
        .groupby(["CID", "activity"])
        .size()
        .unstack(fill_value=0)
    )

    # 每个重复 CID 的最终 activity
    cid_final_activity = {}
    for cid, total in cid_total_count.items():
        if total >= 2:
            active_cnt = int(activity_stat.loc[cid, "Active"]) if "Active" in activity_stat.columns and cid in activity_stat.index else 0
            inactive_cnt = int(activity_stat.loc[cid, "Inactive"]) if "Inactive" in activity_stat.columns and cid in activity_stat.index else 0
            cid_final_activity[cid] = "Active" if active_cnt > inactive_cnt else "Inactive"

    # 保留每个 CID 的第一条记录
    dedup_df = valid_df.drop_duplicates(subset=["CID"], keep="first").copy()

    # 更新重复 CID 的 activity
    duplicated_mask = dedup_df["CID"].isin(cid_final_activity)
    dedup_df.loc[duplicated_mask, "activity"] = dedup_df.loc[duplicated_mask, "CID"].map(cid_final_activity)

    # 按原列顺序输出
    out_df = dedup_df[columns]
    try:
        out_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    except Exception as e:
        print(f"  [错误] 写出失败: {input_path} -> {e}")
        return

    rows_written = len(out_df)
    duplicate_cids = len(cid_final_activity)

    print(
        f"  [完成] {os.path.basename(input_path)} -> {os.path.basename(output_path)} "
        f"(唯一 CID: {rows_written}, 重复 CID: {duplicate_cids})"
    )


def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"[信息] 开始遍历目录: {base_dir}")
    found_files = []

    for root, _, files in os.walk(base_dir):
        if "classic1.csv" in files:
            found_files.append(os.path.join(root, "classic1.csv"))

    if not found_files:
        print("[信息] 未找到任何 classic1.csv 文件。")
        return

    print(f"[信息] 共找到 {len(found_files)} 个 classic1.csv 文件。\n")

    for input_path in sorted(found_files):
        output_path = os.path.join(os.path.dirname(input_path), "classic2.csv")
        print(f"处理: {input_path}")
        process_file(input_path, output_path)

    print("\n[完成] 所有文件处理完毕。")


if __name__ == "__main__":
    main()
