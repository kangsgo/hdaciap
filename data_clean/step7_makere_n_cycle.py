import os

import pandas as pd

"""
递归遍历当前脚本所在目录及其所有子目录，查找所有以 re3.csv 结尾的文件，
按照以下业务逻辑进行数据清洗与特征计算，并输出 re4.csv：

1. 提取所有不重复的 CID 值，存入 datas 列表。
2. 双重循环遍历 datas：
   - 外层 temp_CID，内层 data2。
   - 筛选 CID == data2 且 relative_cid == temp_CID 的记录。
   - 若存在，计算这些记录的 relative_fold 中位数 median_value。
   - 将所有 CID == data2 的行的 relative_fold 设为 median_value，
     relative_cid 设为 temp_CID。
3. 删除 relative_cid != datas[0] 的所有行，保存为 re4.csv。
"""


def process_re3_to_re4(input_file: str) -> None:
    """处理单个 re3.csv 文件，生成 re4.csv。"""
    output_file = os.path.join(os.path.dirname(input_file), "re4.csv")

    # 读取数据
    df = pd.read_csv(input_file, sep="\t", low_memory=False)

    if df.empty or "CID" not in df.columns:
        print(f"[跳过] {input_file} — 无数据或无 CID 列")
        return

    # 确保必要列存在
    if "relative_cid" not in df.columns or "relative_fold" not in df.columns:
        print(f"[跳过] {input_file} — 缺少 relative_cid 或 relative_fold 列")
        return

    # 1. 提取所有不重复的 CID
    datas = df["CID"].dropna().astype(str).str.strip().unique().tolist()
    datas = [c for c in datas if c != ""]

    if len(datas) < 2:
        print(f"[跳过] {input_file} — 唯一 CID 数量不足 ({len(datas)})")
        return

    print(f"[处理] {input_file} — 唯一 CID 数: {len(datas)}")

    # 转换为字符串类型以便精确匹配
    df["CID"] = df["CID"].astype(str).str.strip()
    df["relative_cid"] = df["relative_cid"].astype(str).str.strip()

    # 2. 双重循环计算
    for temp_CID in datas[:100]:
        for data2 in datas:
            # 筛选 CID == data2 且 relative_cid == temp_CID 的记录
            mask = (df["CID"] == data2) & (df["relative_cid"] == temp_CID)
            filtered = df.loc[mask]

            if filtered.empty:
                continue

            # 计算 relative_fold 的中位数
            median_value = filtered["relative_fold"].median()

            # 更新所有 CID == data2 的行
            update_mask = df["CID"] == data2
            df.loc[update_mask, "relative_fold"] = median_value
            df.loc[update_mask, "relative_cid"] = temp_CID

    # 3. 删除 relative_cid != datas[0] 的所有行
    df = df[df["relative_cid"] == datas[0]].copy()

    # 保存结果
    df.to_csv(output_file, sep="\t", index=False)
    print(f"[完成] 输出 -> {output_file}，剩余行数: {len(df)}")


def main() -> None:
    """递归查找所有 re3.csv 并处理。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))

    for root, dirs, files in os.walk(script_dir):
        for f in files:
            if f.endswith("re3.csv"):
                full_path = os.path.join(root, f)
                try:
                    process_re3_to_re4(full_path)
                except Exception as e:
                    print(f"[错误] {full_path}: {e}")


if __name__ == "__main__":
    main()
