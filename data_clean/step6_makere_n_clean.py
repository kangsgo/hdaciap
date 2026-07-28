import os
from collections import Counter

import pandas as pd

"""
使用python和pandas写一个step6_makere_n_clean.py的脚本，遍历所有目录，如果包含re1.csv
（此文件很大，若需要读取请只读取前4行）结尾的文件则依次进行如下内容的执行：

1.仅保留activity_value有值且activity_qualifier为=号的数据

2.获取CID列重复数，按CID分组，按数目排序

保存为re2.csv,分隔为\t在对应目录下
"""

CHUNK_SIZE = 50000


def process_re1(input_file: str, output_file: str) -> None:
    """处理单个 re1.csv 文件，生成 re2.csv"""

    # Step 1: 分块读取，仅保留 activity_value 有值且 activity_qualifier 为 '=' 的数据
    filtered_chunks = []
    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
        if "activity_value" not in chunk.columns or "activity_qualifier" not in chunk.columns:
            continue
        mask = chunk["activity_value"].notna() & (chunk["activity_qualifier"] == "=")
        filtered = chunk[mask].copy()
        if not filtered.empty:
            filtered_chunks.append(filtered)

    if not filtered_chunks:
        print(f"  [SKIP] {input_file}: no rows match filter condition")
        return

    data = pd.concat(filtered_chunks, ignore_index=True)
    print(f"  Filtered {len(data)} rows with activity_value and activity_qualifier '='")

    # Step 2: 获取 CID 列重复数，按 CID 分组，按数目排序
    if "CID" not in data.columns:
        print(f"  [SKIP] {input_file}: no CID column found")
        return

    cid_counts = Counter(data["CID"].dropna().astype(str).str.strip())

    if not cid_counts:
        print(f"  [SKIP] {input_file}: no CID data")
        return

    # 按 CID 分组，按每组数目降序排序
    cid_order = sorted(cid_counts.keys(), key=lambda c: cid_counts[c], reverse=True)
    data["CID_sort_key"] = pd.Categorical(data["CID"].astype(str).str.strip(), categories=cid_order, ordered=True)
    result = data.sort_values("CID_sort_key").drop(columns=["CID_sort_key"])

    result.to_csv(output_file, sep="\t", index=False)
    print(f"  Saved {len(result)} rows to {output_file} (unique CIDs: {len(cid_counts)})")


def main():
    print('打印数据')
    root_dir = os.path.dirname(os.path.abspath(__file__))
    print('打印数据')
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "re1.csv" in filenames:
            input_path = os.path.join(dirpath, "re1.csv")
            output_path = os.path.join(dirpath, "re2.csv")
            print(f"Processing: {input_path}")
            process_re1(input_path, output_path)


if __name__ == "__main__":
    main()
