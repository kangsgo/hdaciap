import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

# ===== Configuration =====
SMILES_COL = "smiles"
VALID_SIZE = 0.2
RANDOM_SEED = 42


def get_scaffold(smiles: str):
    """Generate Bemis-Murcko scaffold from SMILES; return None on failure."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def light_scaffold_split(data: pd.DataFrame, valid_size: float = 0.2):
    """
    Simple scaffold-based split:
    - Molecules sharing the same scaffold stay together
    - Returns train_df, valid_df (with a 'split' column)
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
    """Perform scaffold split on a single reg.csv and write reg_scaffold.csv."""
    df = pd.read_csv(input_path, sep=None, engine="python")
    if SMILES_COL not in df.columns:
        raise ValueError(f"{input_path} missing required column: {SMILES_COL}")

    raw_n = len(df)
    df["scaffold"] = df[SMILES_COL].apply(get_scaffold)

    # Remove samples with unparseable SMILES
    df = df.dropna(subset=["scaffold"]).copy()
    invalid_n = raw_n - len(df)

    if len(df) < 2:
        raise ValueError(f"{input_path}: fewer than 2 valid samples; cannot perform scaffold split.")

    # Fix random seed for reproducibility
    random.seed(RANDOM_SEED)
    train_df, valid_df = light_scaffold_split(df, valid_size=VALID_SIZE)

    merged = pd.concat([train_df, valid_df], axis=0).sort_index()
    merged = merged.drop(columns=["name"], errors="ignore")
    merged.to_csv(output_path, index=False)

    print(f"[{os.path.dirname(input_path)}] Done")
    print(f"  Valid samples: {len(df)}, Invalid SMILES: {invalid_n}")
    print(f"  Train: {len(train_df)}, Valid: {len(valid_df)}")
    print(f"  Output: {output_path}")
    print()


def main():
    base_dir = Path(".")
    reg_files = sorted(base_dir.glob("*/re5.csv"))

    if not reg_files:
        print("No */re5.csv files found.")
        return

    print(f"Found {len(reg_files)} re5.csv file(s)\n")

    for reg_path in reg_files:
        parent_dir = reg_path.parent
        out_path = parent_dir / "reg_scaffold.csv"
        try:
            process_reg_csv(str(reg_path), str(out_path))
        except Exception as e:
            print(f"[{parent_dir}] Processing failed: {e}\n")

    print("All processing completed.")


if __name__ == "__main__":
    main()
