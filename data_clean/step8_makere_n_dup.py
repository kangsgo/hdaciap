import os

import pandas as pd


"""
使用 python 和 pandas 写一个 step8_makere_n_dup.py 的脚本，遍历所有目录，
如果包含 re4.csv 结尾的文件则依次执行：

1. 对 CID 重复的数据，保留索引最小（即首次出现）的那一行。
2. 结果保存为对应目录下 re5.csv，分隔符为 \\t。
"""


def process_re4_to_re5(input_file: str) -> None:
    """处理单个 re4.csv 文件：按 CID 去重保留首条，输出 re5.csv。"""
    output_file = os.path.join(os.path.dirname(input_file), "re5.csv")

    df = pd.read_csv(input_file, sep="\t", low_memory=False)

    if df.empty:
        print(f"  [SKIP] {input_file}: 空文件")
        return

    if "CID" not in df.columns:
        print(f"  [SKIP] {input_file}: 缺少 CID 列")
        return

    before = len(df)

    # 按 CID 去重，保留首次出现的行（即原始索引最小的行）
    df = df.drop_duplicates(subset="CID", keep="first")

    after = len(df)
    removed = before - after
    print(f"  {os.path.dirname(input_file)}: {before} -> {after} 行 (去除 {removed} 条重复)")

    df.to_csv(output_file, sep="\t", index=False)
    print(f"  已保存至 {output_file}")


def main() -> None:
    root_dir = os.path.dirname(os.path.abspath(__file__))

    for current_dir, _, files in os.walk(root_dir):
        if "re4.csv" in files:
            input_path = os.path.join(current_dir, "re4.csv")
            print(f"处理: {input_path}")
            process_re4_to_re5(input_path)


if __name__ == "__main__":
    main()
