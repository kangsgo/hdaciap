import os
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import pandas as pd

'''
使用python和pandas写一个step6_makere_clean.py的脚本，遍历所有目录，如果包含re1.csv（此文件很大，若需要读取请只读取前4行）结尾的文件则依次进行如下内容的执行：

1.仅保留activity_name为IC50的数据，拷贝为data1

2.获取CID列重复数>10的列，按照CID分组，计算每组activity_value列的异常值，所有的异常组索引值均进行记录，记为error_index。

3.记录所有error_index数据的pubmed_ID的值并且在data1中删除，拷贝为data2

data2保存为re2.csv，分隔为\t在对应目录下
'''

CHUNK_SIZE = 50000


def _iter_ic50_chunks(input_file: str):
    """
    分块读取 re1.csv，并仅返回 activity_name == 'IC50' 的数据块（data1 的分块形式）。
    """
    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
        if "activity_name" not in chunk.columns:
            continue
        filtered = chunk[chunk["activity_name"] == "IC50"].copy()
        if not filtered.empty:
            yield filtered


def _get_valid_cids(input_file: str) -> Set[str]:
    """
    统计 data1 中各 CID 的出现次数，返回出现次数 > 10 的 CID 集合。
    """
    cid_counter = Counter()

    for ic50_chunk in _iter_ic50_chunks(input_file):
        if "CID" not in ic50_chunk.columns:
            continue
        cid_series = ic50_chunk["CID"].dropna().astype(str).str.strip()
        cid_series = cid_series[cid_series != ""]
        cid_counter.update(cid_series.tolist())

    return {cid for cid, cnt in cid_counter.items() if cnt > 10}


def _calc_iqr_bounds_for_valid_cids(input_file: str, valid_cids: Set[str]) -> Dict[str, Tuple[float, float]]:
    """
    对 valid_cids 中每个 CID，基于 activity_value 计算 IQR 异常值上下界。
    """
    cid_values: Dict[str, List[float]] = defaultdict(list)

    for ic50_chunk in _iter_ic50_chunks(input_file):
        if "CID" not in ic50_chunk.columns or "activity_value" not in ic50_chunk.columns:
            continue

        work = ic50_chunk[ic50_chunk["CID"].astype(str).isin(valid_cids)].copy()
        if work.empty:
            continue

        work["CID"] = work["CID"].astype(str)
        work["activity_value_num"] = pd.to_numeric(work["activity_value"], errors="coerce")
        work = work.dropna(subset=["activity_value_num"])

        for cid, group in work.groupby("CID"):
            cid_values[cid].extend(group["activity_value_num"].tolist())

    bounds: Dict[str, Tuple[float, float]] = {}
    for cid, values in cid_values.items():
        s = pd.Series(values, dtype="float64")
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        bounds[cid] = (lower, upper)

    return bounds


def clean_re1_to_re2(input_file: str) -> None:
    """
    按要求处理单个 re1.csv：
    1) 保留 activity_name == IC50 -> data1（分块处理，不落盘）
    2) CID 出现次数 > 10 的组，基于 activity_value 按 IQR 计算异常值，记录 error_index
    3) 记录 error_index 对应 pubmed_ID，并在 data1 中删除后得到 data2
    4) data2 保存为同目录 re2.csv（\\t 分隔）
    """
    output_file = os.path.join(os.path.dirname(input_file), "re2.csv")

    # 第 1+2 步：找出 CID>10
    valid_cids = _get_valid_cids(input_file)

    # 若没有可用 CID，仍需按规则输出 data2（即 data1 本身）
    bounds = _calc_iqr_bounds_for_valid_cids(input_file, valid_cids) if valid_cids else {}

    # 第 3 步：按 data1 的全局索引记录异常行并删除，输出 data2
    error_index: List[int] = []
    removed_pubmed_ids: List[str] = []

    data1_global_index = 0
    header_written = False
    source_columns = None

    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
        if source_columns is None:
            source_columns = chunk.columns.tolist()

        if "activity_name" not in chunk.columns:
            continue

        data1_chunk = chunk[chunk["activity_name"] == "IC50"].copy().reset_index(drop=True)
        if data1_chunk.empty:
            continue

        n = len(data1_chunk)
        local_index = pd.RangeIndex(start=0, stop=n, step=1)
        global_index = pd.Series(range(data1_global_index, data1_global_index + n), index=local_index)
        data1_global_index += n

        remove_mask = pd.Series(False, index=local_index)

        if (
            "CID" in data1_chunk.columns
            and "activity_value" in data1_chunk.columns
            and bounds
        ):
            cid_str = data1_chunk["CID"].astype(str)
            act_num = pd.to_numeric(data1_chunk["activity_value"], errors="coerce")

            lower_map = {cid: lu[0] for cid, lu in bounds.items()}
            upper_map = {cid: lu[1] for cid, lu in bounds.items()}

            lowers = cid_str.map(lower_map)
            uppers = cid_str.map(upper_map)

            candidate_mask = lowers.notna() & uppers.notna() & act_num.notna()
            outlier_mask = candidate_mask & ((act_num < lowers) | (act_num > uppers))
            remove_mask = outlier_mask.fillna(False)

        if remove_mask.any():
            error_index.extend(global_index[remove_mask].tolist())

            if "pubmed_ID" in data1_chunk.columns:
                removed_pubmed_ids.extend(
                    data1_chunk.loc[remove_mask, "pubmed_ID"].dropna().astype(str).tolist()
                )

        data2_chunk = data1_chunk.loc[~remove_mask].copy()

        if not data2_chunk.empty:
            mode = "w" if not header_written else "a"
            data2_chunk.to_csv(output_file, sep="\t", index=False, mode=mode, header=not header_written)
            header_written = True

    # 若没有任何可写入数据，输出空表头文件
    if not header_written:
        empty_cols = source_columns if source_columns is not None else []
        pd.DataFrame(columns=empty_cols).to_csv(output_file, sep="\t", index=False)

    print(f"[完成] {input_file} -> {output_file}")
    print(f"  - error_index 数量: {len(error_index)}")
    print(f"  - 删除的 pubmed_ID 数量: {len(removed_pubmed_ids)}")


def main() -> None:
    root_dir = os.getcwd()

    for current_dir, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith("re1.csv"):
                input_path = os.path.join(current_dir, file_name)
                clean_re1_to_re2(input_path)


if __name__ == "__main__":
    main()
