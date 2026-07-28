import math
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

# ===== 配置 =====
SMILES_COL = "smiles"
TARGET_COL = "y"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def get_scaffold(smiles: str):
    """从 SMILES 生成 Bemis-Murcko scaffold，解析失败返回 None。"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def build_strata(df: pd.DataFrame) -> pd.Series:
    """
    构造分层标签：scaffold + y
    并处理低频 strata，避免 train_test_split(stratify=...) 报错。
    """
    strata = df["scaffold"].astype(str) + "_" + df[TARGET_COL].astype(str)

    # 至少保证每个 strata 有 2 个样本
    counts = strata.value_counts()
    strata = strata.apply(lambda x: "other" if counts.get(x, 0) < 2 else x)

    # 还需保证 test 集样本数 >= strata 类别数
    n_test = math.ceil(len(df) * TEST_SIZE)
    while True:
        class_counts = strata.value_counts()
        if len(class_counts) <= n_test:
            break

        non_other = class_counts.drop(labels=["other"], errors="ignore")
        if non_other.empty:
            break

        # 合并最小类别到 other
        rarest = non_other.idxmin()
        strata = strata.apply(lambda x: "other" if x == rarest else x)

    return strata


def process_classic_csv(input_path, train_out, test_out):
    """对单个 classic.csv 执行 scaffold split。"""
    # 1) 读取数据
    df = pd.read_csv(input_path)
    if SMILES_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"{input_path} 缺少必要列: {SMILES_COL}, {TARGET_COL}")

    # 2) 计算 scaffold
    raw_n = len(df)
    df["scaffold"] = df[SMILES_COL].apply(get_scaffold)

    # 3) 去掉无法解析 SMILES 的数据
    df = df.dropna(subset=["scaffold"]).copy()
    invalid_n = raw_n - len(df)
    if len(df) < 2:
        raise ValueError(f"{input_path}: 有效样本不足 2 条，无法划分训练/测试集。")

    # 4) scaffold + y 分层标签
    df["strata"] = build_strata(df)

    # 5) 分层切分 80/20
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["strata"],
    )

    # 6) 删除辅助列并保存（保留 scaffold，删除 strata 和 name）
    train_df = train_df.drop(columns=["strata", "name"], errors="ignore")
    test_df = test_df.drop(columns=["strata", "name"], errors="ignore")

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"[{os.path.dirname(input_path)}] 完成")
    print(f"  有效样本数: {len(df)}, 无效 SMILES 数: {invalid_n}")
    print(f"  训练集: {len(train_df)} -> {train_out}")
    print(f"  测试集: {len(test_df)} -> {test_out}")
    print()


def main():
    base_dir = Path(".")
    classic_files = sorted(base_dir.glob("*/classic.csv"))

    if not classic_files:
        print("未找到任何 */classic.csv 文件。")
        return

    print(f"找到 {len(classic_files)} 个 classic.csv 文件\n")

    for classic_path in classic_files:
        parent_dir = classic_path.parent
        train_out = parent_dir / "train_classic.csv"
        test_out = parent_dir / "test_classic.csv"

        try:
            process_classic_csv(str(classic_path), str(train_out), str(test_out))
        except Exception as e:
            print(f"[{parent_dir}] 处理失败: {e}\n")

    print("全部处理完成。")


if __name__ == "__main__":
    main()
