import math
import os
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import train_test_split

# ===== Configuration =====
SMILES_COL = "smiles"
TARGET_COL = "y"
TEST_SIZE = 0.2
RANDOM_STATE = 42


def get_scaffold(smiles: str):
    """Generate Bemis-Murcko scaffold from SMILES; return None on failure."""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol)


def build_strata(df: pd.DataFrame) -> pd.Series:
    """
    Construct stratification labels: scaffold + y.
    Merge low-frequency strata to avoid train_test_split(stratify=...) errors.
    """
    strata = df["scaffold"].astype(str) + "_" + df[TARGET_COL].astype(str)

    # Ensure at least 2 samples per stratum
    counts = strata.value_counts()
    strata = strata.apply(lambda x: "other" if counts.get(x, 0) < 2 else x)

    # Ensure test set size >= number of stratum classes
    n_test = math.ceil(len(df) * TEST_SIZE)
    while True:
        class_counts = strata.value_counts()
        if len(class_counts) <= n_test:
            break

        non_other = class_counts.drop(labels=["other"], errors="ignore")
        if non_other.empty:
            break

        # Merge the rarest class into "other"
        rarest = non_other.idxmin()
        strata = strata.apply(lambda x: "other" if x == rarest else x)

    return strata


def process_classic_csv(input_path, train_out, test_out):
    """Perform scaffold split on a single classic.csv file."""
    # 1) Load data
    df = pd.read_csv(input_path)
    if SMILES_COL not in df.columns or TARGET_COL not in df.columns:
        raise ValueError(f"{input_path} missing required columns: {SMILES_COL}, {TARGET_COL}")

    # 2) Compute scaffold
    raw_n = len(df)
    df["scaffold"] = df[SMILES_COL].apply(get_scaffold)

    # 3) Remove samples with unparseable SMILES
    df = df.dropna(subset=["scaffold"]).copy()
    invalid_n = raw_n - len(df)
    if len(df) < 2:
        raise ValueError(f"{input_path}: fewer than 2 valid samples; cannot split into train/test.")

    # 4) Build scaffold + y stratification labels
    df["strata"] = build_strata(df)

    # 5) Stratified 80/20 split
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["strata"],
    )

    # 6) Drop auxiliary columns and save (keep scaffold, remove strata and name)
    train_df = train_df.drop(columns=["strata", "name"], errors="ignore")
    test_df = test_df.drop(columns=["strata", "name"], errors="ignore")

    train_df.to_csv(train_out, index=False)
    test_df.to_csv(test_out, index=False)

    print(f"[{os.path.dirname(input_path)}] Done")
    print(f"  Valid samples: {len(df)}, Invalid SMILES: {invalid_n}")
    print(f"  Train set: {len(train_df)} -> {train_out}")
    print(f"  Test set:  {len(test_df)} -> {test_out}")
    print()


def main():
    base_dir = Path(".")
    classic_files = sorted(base_dir.glob("*/classic.csv"))

    if not classic_files:
        print("No */classic.csv files found.")
        return

    print(f"Found {len(classic_files)} classic.csv file(s)\n")

    for classic_path in classic_files:
        parent_dir = classic_path.parent
        train_out = parent_dir / "train_classic.csv"
        test_out = parent_dir / "test_classic.csv"

        try:
            process_classic_csv(str(classic_path), str(train_out), str(test_out))
        except Exception as e:
            print(f"[{parent_dir}] Processing failed: {e}\n")

    print("All processing completed.")


if __name__ == "__main__":
    main()
