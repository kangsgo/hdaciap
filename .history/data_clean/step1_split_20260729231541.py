#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split molecules.csv by the 'target_a' column into per-group subdirectories.

Each group is written as a tab-separated CSV file under a directory named after
the corresponding target_a value.

Usage
-----
    conda activate prolif
    python step1_split.py

Requirements
------------
    molecules.csv must reside in the same directory as this script, be
    tab-delimited, and contain a 'target_a' column.
"""

import csv
import os
from collections import defaultdict

def main():
    input_file = "molecules.csv"
    output_base = "."
    target_col = "target_a"

    # Phase 1: collect rows by target_a
    groups = defaultdict(list)
    header = None

    print(f"[INFO] Reading: {input_file}")
    with open(input_file, newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter="\t")
        for idx, row in enumerate(reader):
            if idx == 0:
                # Save header and locate column index of target_a
                header = row
                if target_col not in header:
                    raise SystemExit(
                        f"Column '{target_col}' not found in header. "
                        f"Available columns: {header}"
                    )
                target_idx = header.index(target_col)
                continue

            # Treat rows with insufficient columns or empty target_a as "unknown"
            if len(row) <= target_idx:
                group_key = "unknown"
            else:
                group_key = row[target_idx].strip()
                if group_key == "":
                    group_key = "unknown"

            groups[group_key].append(row)

    total_groups = len(groups)
    total_rows_written = 0
    print(f"[INFO] Found {total_groups} distinct target_a values.")

    # Phase 2: write each group to a subdirectory
    for group_name, rows in groups.items():
        # Sanitize directory name: replace common illegal characters
        safe_name = group_name.replace("/", "_").replace("\\", "_").strip()
        if not safe_name:
            safe_name = "unknown"

        out_dir = os.path.join(output_base, safe_name)
        os.makedirs(out_dir, exist_ok=True)

        out_file = os.path.join(out_dir, f"{safe_name}_molecules.csv")
        with open(out_file, "w", newline="", encoding="utf-8") as f_out:
            writer = csv.writer(f_out, delimiter="\t")
            writer.writerow(header)
            writer.writerows(rows)

        n_rows = len(rows)
        total_rows_written += n_rows
        print(f"  → {safe_name}: {n_rows} rows -> {out_file}")

    print(
        f"[DONE] Wrote {total_rows_written} rows "
        f"across {total_groups} directories."
    )

if __name__ == "__main__":
    main()
