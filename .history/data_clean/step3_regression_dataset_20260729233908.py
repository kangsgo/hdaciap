"""
step3_regression_dataset.py — Full regression dataset construction pipeline.

Workflow
--------
Phase 1  *_molecules.csv  ──►  re1.csv          (non-empty activity_value)
Phase 2  re1.csv          ──►  re2.csv          (IC50 filter + IQR outlier removal)
Phase 3  re1.csv          ──►  re2_n.csv        (activity_qualifier == '=', CID-sorted)
Phase 4  re2_n.csv        ──►  re3.csv          (AID-grouped relative fold & log)
Phase 5  re3.csv          ──►  re4.csv          (pairwise CID median cycle)

Phase 2 (IC50 pipeline) and Phase 3–5 (qualifier pipeline) are independent;
both start from ``re1.csv``.
"""

import os
from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

CHUNK_SIZE = 50000

# =============================================================================
# Phase 1 — Filter molecules with non-empty activity_value
# =============================================================================


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

    for chunk in pd.read_csv(input_file, sep='\t', chunksize=CHUNK_SIZE):
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
        result.to_csv(output_file, sep='\t', index=False)
    else:
        # When no records satisfy the filter, still emit a header-only re1.csv
        empty_df = pd.DataFrame(columns=columns if columns is not None else [])
        empty_df.to_csv(output_file, index=False)

    print(f"[DONE] {input_file} -> {output_file}")


# =============================================================================
# Phase 2 — IC50 filter + IQR outlier removal  (re1.csv → re2.csv)
# =============================================================================


def _iter_ic50_chunks(input_file: str):
    """Yield chunks of *input_file* where ``activity_name == 'IC50'``."""
    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
        if "activity_name" not in chunk.columns:
            continue
        filtered = chunk[chunk["activity_name"] == "IC50"].copy()
        if not filtered.empty:
            yield filtered


def _get_valid_cids(input_file: str) -> Set[str]:
    """Return CIDs that appear more than 10 times in the IC50 subset."""
    cid_counter = Counter()

    for ic50_chunk in _iter_ic50_chunks(input_file):
        if "CID" not in ic50_chunk.columns:
            continue
        cid_series = ic50_chunk["CID"].dropna().astype(str).str.strip()
        cid_series = cid_series[cid_series != ""]
        cid_counter.update(cid_series.tolist())

    return {cid for cid, cnt in cid_counter.items() if cnt > 10}


def _calc_iqr_bounds_for_valid_cids(
    input_file: str, valid_cids: Set[str]
) -> Dict[str, Tuple[float, float]]:
    """Compute IQR-based outlier bounds (lower, upper) for each CID in *valid_cids*."""
    cid_values: Dict[str, List[float]] = defaultdict(list)

    for ic50_chunk in _iter_ic50_chunks(input_file):
        if "CID" not in ic50_chunk.columns or "activity_value" not in ic50_chunk.columns:
            continue

        work = ic50_chunk[ic50_chunk["CID"].astype(str).isin(valid_cids)].copy()
        if work.empty:
            continue

        work["CID"] = work["CID"].astype(str)
        work["activity_value_num"] = pd.to_numeric(work["activity_value"], errors="coerce")
        work = work.dropna(subset=["activity_value_num"])

        for cid, group in work.groupby("CID"):
            cid_values[cid].extend(group["activity_value_num"].tolist())

    bounds: Dict[str, Tuple[float, float]] = {}
    for cid, values in cid_values.items():
        s = pd.Series(values, dtype="float64")
        if s.empty:
            continue
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        bounds[cid] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

    return bounds


def clean_re1_to_re2(input_file: str) -> None:
    """Process a single re1.csv through the IC50 + IQR pipeline → re2.csv.

    1. Keep only rows with ``activity_name == 'IC50'``.
    2. For CIDs with >10 occurrences, flag IQR outliers on ``activity_value``.
    3. Record removed ``pubmed_ID`` values; drop outlier rows.
    4. Write the cleaned result as ``re2.csv`` (tab-separated).
    """
    output_file = os.path.join(os.path.dirname(input_file), "re2.csv")

    valid_cids = _get_valid_cids(input_file)
    bounds = _calc_iqr_bounds_for_valid_cids(input_file, valid_cids) if valid_cids else {}

    error_index: List[int] = []
    removed_pubmed_ids: List[str] = []

    data1_global_index = 0
    header_written = False
    source_columns = None

    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
        if source_columns is None:
            source_columns = chunk.columns.tolist()

        if "activity_name" not in chunk.columns:
            continue

        data1_chunk = chunk[chunk["activity_name"] == "IC50"].copy().reset_index(drop=True)
        if data1_chunk.empty:
            continue

        n = len(data1_chunk)
        local_index = pd.RangeIndex(start=0, stop=n, step=1)
        global_index = pd.Series(
            range(data1_global_index, data1_global_index + n), index=local_index
        )
        data1_global_index += n

        remove_mask = pd.Series(False, index=local_index)

        if "CID" in data1_chunk.columns and "activity_value" in data1_chunk.columns and bounds:
            cid_str = data1_chunk["CID"].astype(str)
            act_num = pd.to_numeric(data1_chunk["activity_value"], errors="coerce")

            lowers = cid_str.map({cid: lu[0] for cid, lu in bounds.items()})
            uppers = cid_str.map({cid: lu[1] for cid, lu in bounds.items()})

            candidate_mask = lowers.notna() & uppers.notna() & act_num.notna()
            remove_mask = candidate_mask & ((act_num < lowers) | (act_num > uppers))

        if remove_mask.any():
            error_index.extend(global_index[remove_mask].tolist())
            if "pubmed_ID" in data1_chunk.columns:
                removed_pubmed_ids.extend(
                    data1_chunk.loc[remove_mask, "pubmed_ID"].dropna().astype(str).tolist()
                )

        data2_chunk = data1_chunk.loc[~remove_mask].copy()

        if not data2_chunk.empty:
            mode = "w" if not header_written else "a"
            data2_chunk.to_csv(
                output_file, sep="\t", index=False, mode=mode, header=not header_written
            )
            header_written = True

    if not header_written:
        empty_cols = source_columns if source_columns is not None else []
        pd.DataFrame(columns=empty_cols).to_csv(output_file, sep="\t", index=False)

    print(f"[DONE] {input_file} -> {output_file}")
    print(f"  - error_index count: {len(error_index)}")
    print(f"  - removed pubmed_ID count: {len(removed_pubmed_ids)}")


# =============================================================================
# Phase 3 — Qualifier filter + CID-frequency sort  (re1.csv → re2_n.csv)
# =============================================================================


def process_re1_qualifier(input_file: str) -> None:
    """Filter re1.csv for exact activity values and sort by CID frequency → re2_n.csv.

    1. Keep rows where ``activity_value`` is not null and ``activity_qualifier == '='``.
    2. Sort rows in descending order of CID occurrence count.
    """
    output_file = os.path.join(os.path.dirname(input_file), "re2_n.csv")

    filtered_chunks = []
    for chunk in pd.read_csv(input_file, sep="\t", chunksize=CHUNK_SIZE, low_memory=False):
        if "activity_value" not in chunk.columns or "activity_qualifier" not in chunk.columns:
            continue
        mask = chunk["activity_value"].notna() & (chunk["activity_qualifier"] == "=")
        filtered = chunk[mask].copy()
        if not filtered.empty:
            filtered_chunks.append(filtered)

    if not filtered_chunks:
        print(f"[SKIP] {input_file}: no rows match filter condition")
        return

    data = pd.concat(filtered_chunks, ignore_index=True)
    print(f"  Filtered {len(data)} rows with activity_value and activity_qualifier '='")

    if "CID" not in data.columns:
        print(f"[SKIP] {input_file}: no CID column found")
        return

    cid_counts = Counter(data["CID"].dropna().astype(str).str.strip())
    if not cid_counts:
        print(f"[SKIP] {input_file}: no CID data")
        return

    cid_order = sorted(cid_counts.keys(), key=lambda c: cid_counts[c], reverse=True)
    data["CID_sort_key"] = pd.Categorical(
        data["CID"].astype(str).str.strip(), categories=cid_order, ordered=True
    )
    result = data.sort_values("CID_sort_key").drop(columns=["CID_sort_key"])

    result.to_csv(output_file, sep="\t", index=False)
    print(f"[DONE] {input_file} -> {output_file} ({len(result)} rows, {len(cid_counts)} unique CIDs)")


# =============================================================================
# Phase 4 — AID-grouped relative fold  (re2_n.csv → re3.csv)
# =============================================================================


def process_re2_n_to_re3(input_file: str) -> None:
    """Compute log_value, relative_cid, and relative_fold grouped by AID → re3.csv.

    1. Add ``log_value`` = ln(activity_value) rounded to 2 decimals.
    2. Within each AID group, use the row with the smallest positional index
       as the baseline; broadcast its CID as ``relative_cid`` and compute
       ``relative_fold`` = activity_value / baseline_activity_value (2 decimals).
    """
    output_file = os.path.join(os.path.dirname(input_file), "re3.csv")

    df = pd.read_csv(input_file, sep="\t", low_memory=False)
    if df.empty:
        print(f"[SKIP] {input_file}: empty file")
        return

    required_cols = ["activity_value", "AID", "CID"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[SKIP] {input_file}: missing columns: {missing}")
        return

    df["activity_value"] = pd.to_numeric(df["activity_value"], errors="coerce")
    df = df[df["activity_value"].notna() & (df["activity_value"] > 0)].copy()
    df["log_value"] = np.log(df["activity_value"]).round(2)
    df = df.reset_index(drop=True)

    baseline = df.loc[df.groupby("AID", group_keys=False).apply(
        lambda g: g.index.min(), include_groups=False
    )]

    baseline_map_cid = dict(zip(baseline["AID"], baseline["CID"]))
    baseline_map_value = dict(zip(baseline["AID"], baseline["activity_value"]))

    df["relative_cid"] = df["AID"].map(baseline_map_cid)
    df["relative_fold"] = df.apply(
        lambda row: round(row["activity_value"] / baseline_map_value[row["AID"]], 2),
        axis=1,
    )

    df.to_csv(output_file, sep="\t", index=False)
    print(f"[DONE] {input_file} -> {output_file} ({len(df)} rows)")


# =============================================================================
# Phase 5 — Pairwise CID median cycle  (re3.csv → re4.csv)
# =============================================================================


def process_re3_to_re4(input_file: str) -> None:
    """Apply pairwise CID median imputation on relative_fold → re4.csv.

    1. For every pair (temp_CID, data2) of unique CIDs, compute the median
       ``relative_fold`` of rows where CID == data2 and relative_cid == temp_CID.
    2. Assign the median to all rows with CID == data2.
    3. Keep only rows whose ``relative_cid`` equals the first unique CID.
    """
    output_file = os.path.join(os.path.dirname(input_file), "re4.csv")

    df = pd.read_csv(input_file, sep="\t", low_memory=False)
    if df.empty or "CID" not in df.columns:
        print(f"[SKIP] {input_file} — no data or missing CID column")
        return

    if "relative_cid" not in df.columns or "relative_fold" not in df.columns:
        print(f"[SKIP] {input_file} — missing relative_cid or relative_fold column")
        return

    datas = df["CID"].dropna().astype(str).str.strip().unique().tolist()
    datas = [c for c in datas if c != ""]
    if len(datas) < 2:
        print(f"[SKIP] {input_file} — insufficient unique CIDs ({len(datas)})")
        return

    print(f"[PROCESS] {input_file} — unique CIDs: {len(datas)}")

    df["CID"] = df["CID"].astype(str).str.strip()
    df["relative_cid"] = df["relative_cid"].astype(str).str.strip()

    for temp_CID in datas[:100]:
        for data2 in datas:
            mask = (df["CID"] == data2) & (df["relative_cid"] == temp_CID)
            filtered = df.loc[mask]
            if filtered.empty:
                continue
            median_value = filtered["relative_fold"].median()
            update_mask = df["CID"] == data2
            df.loc[update_mask, "relative_fold"] = median_value
            df.loc[update_mask, "relative_cid"] = temp_CID

    df = df[df["relative_cid"] == datas[0]].copy()
    df.to_csv(output_file, sep="\t", index=False)
    print(f"[DONE] {input_file} -> {output_file} ({len(df)} rows remaining)")


# =============================================================================
# Main — walk directories and run all phases in sequence
# =============================================================================


def main() -> None:
    """Walk the working directory and execute the full regression pipeline.

    Phases are run in order:
      1. *_molecules.csv → re1.csv
      2. re1.csv → re2.csv   (IC50 + IQR)
      3. re1.csv → re2_n.csv (qualifier + CID-sort)
      4. re2_n.csv → re3.csv (relative fold)
      5. re3.csv → re4.csv   (median cycle)
    """
    root_dir = os.getcwd()

    # ---- Phase 1: *_molecules.csv → re1.csv ----
    for current_dir, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.endswith("_molecules.csv"):
                input_path = os.path.join(current_dir, file_name)
                filter_activity_value(input_path)

    # ---- Phase 2: re1.csv → re2.csv (IC50 pipeline) ----
    for current_dir, _, files in os.walk(root_dir):
        if "re1.csv" in files:
            input_path = os.path.join(current_dir, "re1.csv")
            clean_re1_to_re2(input_path)

    # ---- Phase 3: re1.csv → re2_n.csv (qualifier pipeline) ----
    for current_dir, _, files in os.walk(root_dir):
        if "re1.csv" in files:
            input_path = os.path.join(current_dir, "re1.csv")
            process_re1_qualifier(input_path)

    # ---- Phase 4: re2_n.csv → re3.csv ----
    for current_dir, _, files in os.walk(root_dir):
        if "re2_n.csv" in files:
            input_path = os.path.join(current_dir, "re2_n.csv")
            process_re2_n_to_re3(input_path)

    # ---- Phase 5: re3.csv → re4.csv ----
    for current_dir, _, files in os.walk(root_dir):
        if "re3.csv" in files:
            input_path = os.path.join(current_dir, "re3.csv")
            try:
                process_re3_to_re4(input_path)
            except Exception as e:
                print(f"[ERROR] {input_path}: {e}")


if __name__ == "__main__":
    main()

