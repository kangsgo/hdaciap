import os
from collections import defaultdict
from typing import Dict, Set, Tuple

import pandas as pd

"""
使用 python 和 pandas 写一个 step7_makere_dup.py 的脚本，遍历所有目录，
如果包含 re2.csv 结尾的文件则依次执行：

1. 对 CID 重复的数据，activity_qualifier 设为 '='，activity_value 取重复 CID 的平均值。
2. 设置 activity_value >= 10 的数据 activity 为 Inactive，否则为 Active。
3. 重复 CID 仅保留 1 个，拷贝为 data2。
4. data2 保存为对应目录下 re3.csv，分隔符为 \\t。
"""

CHUNK_SIZE = 50000


def _collect_duplicate_cid_means(input_file: str) -> Tuple[Set[str], Dict[str, float]]:
    """
    第一遍分块扫描：
    - 统计每个 CID 出现次数
    - 统计每个 CID 的 activity_value 数值和/计数（用于均值）
    返回：
    - duplicated_cids: 出现次数 > 1 的 CID 集合
    - cid_mean_map: 重复 CID 的 activity_value 平均值映射
    """
    cid_count = defaultdict(int)
    cid_value_sum = defaultdict(float)
    cid_value_num_count = defaultdict(int)

    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
        if "CID" not in chunk.columns:
            continue

        cid_series = chunk["CID"].astype(str).str.strip()
        valid_cid_mask = cid_series.notna() & (cid_series != "")
        work = chunk.loc[valid_cid_mask].copy()
        work["CID"] = cid_series[valid_cid_mask]

        if work.empty:
            continue

        for cid, cnt in work["CID"].value_counts().items():
            cid_count[cid] += int(cnt)

        if "activity_value" in work.columns:
            work["activity_value_num"] = pd.to_numeric(work["activity_value"], errors="coerce")
            value_work = work.dropna(subset=["activity_value_num"])
            if not value_work.empty:
                grouped = value_work.groupby("CID")["activity_value_num"].agg(["sum", "count"])
                for cid, row in grouped.iterrows():
                    cid_value_sum[cid] += float(row["sum"])
                    cid_value_num_count[cid] += int(row["count"])

    duplicated_cids = {cid for cid, cnt in cid_count.items() if cnt > 1}
    cid_mean_map: Dict[str, float] = {}

    for cid in duplicated_cids:
        cnt = cid_value_num_count.get(cid, 0)
        if cnt > 0:
            cid_mean_map[cid] = cid_value_sum[cid] / cnt

    return duplicated_cids, cid_mean_map


def process_re2_to_re3(input_file: str) -> None:
    """
    处理单个 re2.csv：
    1) 重复 CID 行：activity_qualifier='='，activity_value=该 CID 平均值
    2) activity_value>=10 -> Inactive，否则 Active
    3) CID 去重仅保留首条
    4) 输出 re3.csv（\\t 分隔）
    """
    output_file = os.path.join(os.path.dirname(input_file), "re3.csv")

    duplicated_cids, cid_mean_map = _collect_duplicate_cid_means(input_file)

    header_written = False
    source_columns = None
    seen_cids: Set[str] = set()

    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
        if source_columns is None:
            source_columns = chunk.columns.tolist()

        if "CID" not in chunk.columns:
            continue

        work = chunk.copy()
        work["CID"] = work["CID"].astype(str).str.strip()
        work = work[work["CID"].notna() & (work["CID"] != "")].copy()

        if work.empty:
            continue

        dup_mask = work["CID"].isin(duplicated_cids)

        if "activity_qualifier" in work.columns:
            work.loc[dup_mask, "activity_qualifier"] = "="

        if "activity_value" in work.columns and cid_mean_map:
            mapped_mean = work["CID"].map(cid_mean_map)
            mean_mask = dup_mask & mapped_mean.notna()
            work.loc[mean_mask, "activity_value"] = mapped_mean[mean_mask]

        if "activity_value" in work.columns:
            activity_num = pd.to_numeric(work["activity_value"], errors="coerce")
            work["activity"] = "Active"
            work.loc[activity_num >= 10, "activity"] = "Inactive"
            work.loc[activity_num.isna(), "activity"] = "Active"
        else:
            work["activity"] = "Active"

        not_seen_before_mask = ~work["CID"].isin(seen_cids)
        first_in_chunk_mask = ~work["CID"].duplicated(keep="first")
        keep_mask = not_seen_before_mask & first_in_chunk_mask
        data2_chunk = work.loc[keep_mask].copy()

        seen_cids.update(data2_chunk["CID"].tolist())

        if not data2_chunk.empty:
            mode = "w" if not header_written else "a"
            data2_chunk.to_csv(output_file, sep="\t", index=False, mode=mode, header=not header_written)
            header_written = True

    if not header_written:
        empty_cols = source_columns if source_columns is not None else []
        if "activity" not in empty_cols:
            empty_cols = empty_cols + ["activity"]
        pd.DataFrame(columns=empty_cols).to_csv(output_file, sep="\t", index=False)

    print(f"[完成] {input_file} -> {output_file}")
    print(f"  - 重复 CID 数量: {len(duplicated_cids)}")
    print(f"  - 输出唯一 CID 数量: {len(seen_cids)}")


def main() -> None:
    root_dir = os.getcwd()

    for current_dir, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith("re2.csv"):
                input_path = os.path.join(current_dir, file_name)
                process_re2_to_re3(input_path)


if __name__ == "__main__":
    main()
