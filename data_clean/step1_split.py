#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
按照 target_a 列拆分 molecules.csv，将每组数据保存到当前目录下对应的子目录中。

用法：
    conda activate prolif
    python step1_split.py

要求：
    molecules.csv 与当前脚本在同一目录下，以制表符分隔，包含 target_a 列。
"""

import csv
import os
from collections import defaultdict

def main():
    input_file = "molecules.csv"
    output_base = "."  # 当前目录
    target_col = "target_a"

    # 第一阶段：按 target_a 收集所有行
    groups = defaultdict(list)
    header = None

    print(f"[信息] 正在读取: {input_file}")
    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if idx == 0:
                # 保存表头，并找到 target_a 所在的列索引
                header = row
                if target_col not in header:
                    raise SystemExit(
                        f"未在 CSV 表头中找到 '{target_col}' 列，"
                        f"当前列: {header}"
                    )
                target_idx = header.index(target_col)
                continue

            # 如果行数据列数不足（尤其是最后一列为空），跳过或归为 "unknown"
            if len(row) <= target_idx:
                group_key = "unknown"
            else:
                group_key = row[target_idx].strip()
                if group_key == "":
                    group_key = "unknown"

            groups[group_key].append(row)

    total_groups = len(groups)
    total_rows_written = 0
    print(f"[信息] 共发现 {total_groups} 个不同的 target_a 值。")

    # 第二阶段：将各组写入对应子目录的 CSV 文件
    for group_name, rows in groups.items():
        # 安全目录名：替换常见非法字符
        safe_name = group_name.replace("/", "_").replace("\\", "_").strip()
        if not safe_name:
            safe_name = "unknown"

        out_dir = os.path.join(output_base, safe_name)
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, f"{safe_name}_molecules.csv")
        with open(out_file, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)

        n_rows = len(rows)
        total_rows_written += n_rows
        print(f"  → {safe_name}: {n_rows} 行 -> {out_file}")

    print(
        f"[完成] 共写入 {total_rows_written} 行，"
        f"分布在 {total_groups} 个目录中。"
    )

if __name__ == "__main__":
    main()
