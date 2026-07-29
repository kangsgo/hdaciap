import os
import pandas as pd


def process_classic2_file(file_path: str) -> None:
    """
    rename activity to y, map Active/Inactive to 1/0, and save as classic.csv
    in the same directory.
    """
    try:
        df = pd.read_csv(
            file_path,
            sep="\t",
            dtype=str,
            low_memory=False,
        )
    except Exception as e:
        print(f"[ERROR] Failed to read {file_path}: {e}")
        return

    required_cols = ["CID", "name", "smiles", "activity"]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        print(f"[SKIP] {file_path} missing columns: {missing_cols}")
        return

    out_df = df[["CID", "name", "smiles", "activity"]].copy()
    out_df = out_df.rename(columns={"activity": "y"})
    out_df["y"] = out_df["y"].map({"Active": 1, "Inactive": 0})

    out_path = os.path.join(os.path.dirname(file_path), "classic.csv")
    out_df.to_csv(out_path, index=False)
    print(f"[OK] Saved: {out_path}")


def main() -> None:
    base_dir = "."
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith("classic2.csv"):
                full_path = os.path.join(root, filename)
                process_classic2_file(full_path)


if __name__ == "__main__":
    main()
