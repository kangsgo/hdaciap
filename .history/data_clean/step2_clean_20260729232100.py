#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clean all *molecules.csv files and produce classic1.csv and classic2.csv.

Phase 1 — Activity cleaning (classic1.csv)
------------------------------------------
1. Activity column logic:
   - Rows with activity "Active" or "Inactive" are retained as-is.
   - Otherwise, only rows whose activity_name is in {IC50, EC50, Ki} are
     retained; all others are discarded.
2. For retained rows whose activity is NOT "Active"/"Inactive":
   - If activity_value is empty or non-numeric, the row is dropped.
   - activity_value >= 10  →  activity = "Inactive"
   - activity_value < 10   →  activity = "Active"

Phase 2 — CID deduplication (classic2.csv)
------------------------------------------
1. Group records by CID in classic1.csv.
2. For CIDs appearing ≥ 2 times, determine the majority activity:
   - Active count > Inactive count  →  "Active"
   - Otherwise                        →  "Inactive"
3. Keep only the first occurrence of each CID; drop the rest.
4. Write deduplicated results as classic2.csv.

Usage
-----
    python step2_clean.py
"""

import os
import pandas as pd


ALLOWED_ACTIVITY = {"Active", "Inactive"}
ALLOWED_ACTIVITY_NAME = {"IC50", "EC50", "Ki"}


def read_table_auto(input_path: str) -> pd.DataFrame:
    """
    Read a delimited file with tab-separator preference; fall back to comma.
    """
    try:
        return pd.read_csv(input_path, sep="\t")
    except Exception:
        return pd.read_csv(input_path)


def process_file(input_path: str, output_path: str) -> None:
    """
    Clean a single *molecules.csv file and write classic1.csv.
    """
    try:
        df = read_table_auto(input_path)
    except Exception as e:
        print(f"  [ERROR] Failed to read, skipping: {input_path} | {e}")
        return

    required_cols = {"activity", "activity_name", "activity_value"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        print(f"  [ERROR] Missing required columns {missing}, skipping: {input_path}")
        return

    total_rows = len(df)

    # Cast to string and strip whitespace to avoid type/null issues
    activity = df["activity"].astype(str).str.strip()
    activity_name = df["activity_name"].astype(str).str.strip()

    # Rule 1: retain rows with standard activity directly
    mask_valid_activity = activity.isin(ALLOWED_ACTIVITY)
    df_standard = df.loc[mask_valid_activity].copy()

    # For non-standard activity, keep only rows with allowed activity_name
    mask_non_standard = ~mask_valid_activity
    mask_valid_name = activity_name.isin(ALLOWED_ACTIVITY_NAME)
    df_non_standard = df.loc[mask_non_standard & mask_valid_name].copy()

    # Convert activity_value to numeric and drop non-numeric rows
    df_non_standard["activity_value"] = pd.to_numeric(
        df_non_standard["activity_value"], errors="coerce"
    )
    df_non_standard = df_non_standard.dropna(subset=["activity_value"])

    # Relabel activity based on activity_value threshold
    df_non_standard["activity"] = "Active"
    df_non_standard.loc[df_non_standard["activity_value"] >= 10, "activity"] = "Inactive"

    # Merge standard rows with processed non-standard rows
    df_kept = pd.concat([df_standard, df_non_standard], ignore_index=True)

    # Drop rows where activity is "Active" but activity_value is missing
    mask_empty_value_active = df_kept["activity"].eq("Active") & df_kept["activity_value"].isna()
    df_kept = df_kept.loc[~mask_empty_value_active]

    kept_rows = len(df_kept)
    deleted_rows = total_rows - kept_rows

    # Write classic1.csv as tab-delimited, consistent with the source format
    df_kept.to_csv(output_path, index=False, sep="\t", encoding="utf-8")
    print(f"    Kept: {kept_rows} rows, Removed: {deleted_rows} rows -> {output_path}")


def dedup_cid(input_path: str, output_path: str) -> None:
    """
    Deduplicate a classic1.csv by CID and write classic2.csv.

    For CIDs that appear multiple times, the final activity is set to the
    majority label (Active vs. Inactive). Only the first occurrence of each
    CID is retained.
    """
    try:
        df = pd.read_csv(input_path, sep="\t", dtype=str, encoding="utf-8")
    except Exception as e:
        print(f"  [ERROR] Failed to read: {input_path} -> {e}")
        return

    required_cols = ["CID", "activity"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"  [ERROR] Missing required columns {missing_cols}, skipping: {input_path}")
        return

    columns = list(df.columns)

    # Normalize key fields
    df["CID"] = df["CID"].fillna("").astype(str).str.strip()
    df["activity"] = df["activity"].fillna("").astype(str).str.strip()

    # Filter out rows with empty CID
    valid_df = df[df["CID"] != ""].copy()
    if valid_df.empty:
        print(f"  [INFO] No valid CID data, skipping: {input_path}")
        return

    # Count occurrences per CID
    cid_total_count = valid_df.groupby("CID").size()

    # Count Active / Inactive per CID
    activity_stat = (
        valid_df[valid_df["activity"].isin(["Active", "Inactive"])]
        .groupby(["CID", "activity"])
        .size()
        .unstack(fill_value=0)
    )

    # Determine final activity for duplicate CIDs
    cid_final_activity = {}
    for cid, total in cid_total_count.items():
        if total >= 2:
            active_cnt = (
                int(activity_stat.loc[cid, "Active"])
                if "Active" in activity_stat.columns and cid in activity_stat.index
                else 0
            )
            inactive_cnt = (
                int(activity_stat.loc[cid, "Inactive"])
                if "Inactive" in activity_stat.columns and cid in activity_stat.index
                else 0
            )
            cid_final_activity[cid] = (
                "Active" if active_cnt > inactive_cnt else "Inactive"
            )

    # Keep the first occurrence of each CID
    dedup_df = valid_df.drop_duplicates(subset=["CID"], keep="first").copy()

    # Update activity for duplicate CIDs
    duplicated_mask = dedup_df["CID"].isin(cid_final_activity)
    dedup_df.loc[duplicated_mask, "activity"] = (
        dedup_df.loc[duplicated_mask, "CID"].map(cid_final_activity)
    )

    # Write output preserving original column order
    out_df = dedup_df[columns]
    try:
        out_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")
    except Exception as e:
        print(f"  [ERROR] Failed to write: {output_path} -> {e}")
        return

    rows_written = len(out_df)
    duplicate_cids = len(cid_final_activity)
    print(
        f"    Dedup: {rows_written} unique CIDs, "
        f"{duplicate_cids} had duplicates -> {os.path.basename(output_path)}"
    )


def find_target_files(base_dir: str, pattern: str = "molecules.csv"):
    """
    Recursively find files whose names end with *pattern*.

    Examples of matches for pattern='molecules.csv':
        molecules.csv, HDAC1_molecules.csv, SIRT2_molecules.csv
    """
    targets = []
    for root, _, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(pattern):
                targets.append(os.path.join(root, filename))
    return sorted(targets)


def main() -> None:
    base_dir = os.path.dirname(os.path.abspath(__file__)) or "."
    all_files = find_target_files(base_dir)

    if not all_files:
        print("[INFO] No files ending with molecules.csv found.")
        return

    print(f"[INFO] Found {len(all_files)} file(s) to process.\n")

    # Phase 1: activity cleaning → classic1.csv
    for input_path in all_files:
        output_path = os.path.join(os.path.dirname(input_path), "classic1.csv")
        print(f"[Phase 1] Processing: {input_path}")
        process_file(input_path, output_path)

    # Phase 2: CID deduplication → classic2.csv
    print("\n[INFO] Starting Phase 2: CID deduplication.\n")
    classic1_files = find_target_files(base_dir, pattern="classic1.csv")
    if not classic1_files:
        print("[INFO] No classic1.csv files found for deduplication.")
        return

    for input_path in classic1_files:
        output_path = os.path.join(os.path.dirname(input_path), "classic2.csv")
        print(f"[Phase 2] Processing: {input_path}")
        dedup_cid(input_path, output_path)

    print("\n[DONE] All files processed.")


if __name__ == "__main__":
    main()
