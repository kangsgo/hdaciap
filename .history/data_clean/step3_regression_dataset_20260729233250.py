import os
import pandas as pd


def filter_activity_value(input_file: str) -> None:
    """Filter molecules with non-empty activity_value from a *_molecules.csv file.

    Reads the input TSV file in chunks, retains rows where the ``activity_value``
    column is not null and not blank, and writes the result as ``re1.csv`` in the
    same directory.

    Parameters
    ----------
    input_file : str
        Path to a ``*_molecules.csv`` (tab-separated) file.
    """
    output_file = os.path.join(os.path.dirname(input_file), "re1.csv")

    kept_chunks = []
    columns = None

    for chunk in pd.read_csv(input_file,sep='\t', chunksize=50000):
        if columns is None:
            columns = chunk.columns

        if "activity_value" not in chunk.columns:
            print(f"[SKIP] {input_file} does not contain 'activity_value' column")
            return

        filtered = chunk[chunk["activity_value"].notna()]
        filtered = filtered[filtered["activity_value"].astype(str).str.strip() != ""]

        if not filtered.empty:
            kept_chunks.append(filtered)

    if kept_chunks:
        result = pd.concat(kept_chunks, ignore_index=True)
        result.to_csv(output_file,sep='\t', index=False)
    else:
        # When no records satisfy the filter, still emit a header-only re1.csv
        empty_df = pd.DataFrame(columns=columns if columns is not None else [])
        empty_df.to_csv(output_file, index=False)

    print(f"[DONE] {input_file} -> {output_file}")


def main() -> None:
    root_dir = os.getcwd()

    for current_dir, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith("_molecules.csv"):
                input_path = os.path.join(current_dir, file_name)
                filter_activity_value(input_path)


if __name__ == "__main__":
    main()
