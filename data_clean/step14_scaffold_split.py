import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ===== 配置 =====
SMILES_COL = "smiles"
VALID_SIZE = 0.2
RANDOM_SEED = 42


def get_scaffold(smiles: str):
    """从 SMILES 生成 Bemis-Murcko scaffold，解析失败返回 None。"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def light_scaffold_split(data: pd.DataFrame, valid_size: float = 0.2):
    """
    按 scaffold 进行简单划分：
    - 同一 scaffold 不会被拆散
    - 输出 train_df, valid_df（包含 split 列）
    """
    scaffolds = defaultdict(list)
    for idx, row in data.iterrows():
        scaffolds[row["scaffold"]].append(idx)

    scaffold_list = list(scaffolds.values())
    random.shuffle(scaffold_list)

    n_total_valid = int(np.floor(valid_size * len(data)))

    valid_idx = []
    train_idx = []

    for scaffold_set in scaffold_list:
        if len(valid_idx) + len(scaffold_set) <= n_total_valid:
            valid_idx.extend(scaffold_set)
        else:
            train_idx.extend(scaffold_set)

    out = data.copy()
    out["split"] = "train"
    out.loc[valid_idx, "split"] = "valid"

    train_df = out.loc[train_idx].copy()
    valid_df = out.loc[valid_idx].copy()
    return train_df, valid_df


def process_reg_csv(input_path: str, output_path: str):
    """对单个 reg.csv 执行 scaffold split，并写出 reg_scaffold.csv。"""
    df = pd.read_csv(input_path, sep=None, engine="python")
    if SMILES_COL not in df.columns:
        raise ValueError(f"{input_path} 缺少必要列: {SMILES_COL}")

    raw_n = len(df)
    df["scaffold"] = df[SMILES_COL].apply(get_scaffold)

    # 去掉无法解析 SMILES 的样本
    df = df.dropna(subset=["scaffold"]).copy()
    invalid_n = raw_n - len(df)

    if len(df) < 2:
        raise ValueError(f"{input_path}: 有效样本不足 2 条，无法进行骨架拆分。")

    # 固定随机种子，保证可复现
    random.seed(RANDOM_SEED)
    train_df, valid_df = light_scaffold_split(df, valid_size=VALID_SIZE)

    merged = pd.concat([train_df, valid_df], axis=0).sort_index()
    merged = merged.drop(columns=["name"], errors="ignore")
    merged.to_csv(output_path, index=False)

    print(f"[{os.path.dirname(input_path)}] 完成")
    print(f"  有效样本数: {len(df)}, 无效 SMILES 数: {invalid_n}")
    print(f"  train: {len(train_df)}, valid: {len(valid_df)}")
    print(f"  输出: {output_path}")
    print()


def main():
    base_dir = Path(".")
    reg_files = sorted(base_dir.glob("*/re5.csv"))

    if not reg_files:
        print("未找到任何 */re5.csv 文件。")
        return

    print(f"找到 {len(reg_files)} 个 re5.csv 文件\n")

    for reg_path in reg_files:
        parent_dir = reg_path.parent
        out_path = parent_dir / "reg_scaffold.csv"
        try:
            process_reg_csv(str(reg_path), str(out_path))
        except Exception as e:
            print(f"[{parent_dir}] 处理失败: {e}\n")

    print("全部处理完成。")


if __name__ == "__main__":
    main()
