import os
import pandas as pd


def filter_activity_value(input_file: str) -> None:
    """
    读取 *_molecules.csv，保留 activity_value 不为空的记录，
    并将结果保存为同目录下的 re1.csv。
    """
    output_file = os.path.join(os.path.dirname(input_file), "re1.csv")

    kept_chunks = []
    columns = None

    for chunk in pd.read_csv(input_file,sep='\t', chunksize=50000):
        if columns is None:
            columns = chunk.columns

        if "activity_value" not in chunk.columns:
            print(f"[跳过] {input_file} 不包含 activity_value 列")
            return

        filtered = chunk[chunk["activity_value"].notna()]
        filtered = filtered[filtered["activity_value"].astype(str).str.strip() != ""]

        if not filtered.empty:
            kept_chunks.append(filtered)

    if kept_chunks:
        result = pd.concat(kept_chunks, ignore_index=True)
        result.to_csv(output_file,sep='\t', index=False)
    else:
        # 没有满足条件的数据时，仍输出一个仅含表头的 re1.csv
        empty_df = pd.DataFrame(columns=columns if columns is not None else [])
        empty_df.to_csv(output_file, index=False)

    print(f"[完成] {input_file} -> {output_file}")


def main() -> None:
    root_dir = os.getcwd()

    for current_dir, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith("_molecules.csv"):
                input_path = os.path.join(current_dir, file_name)
                filter_activity_value(input_path)


if __name__ == "__main__":
    main()
