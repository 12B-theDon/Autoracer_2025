#!/usr/bin/env python3
"""
Utility to thin a dense waypoint CSV down to a fixed number of rows.

Example:
    python3 scripts/downsample_csv.py \
        --input global_path/trajectory.csv \
        --output global_path/trajectory_100.csv \
        --count 100 \
        --has-header

Only standard library modules are used so it runs in a minimal ROS workspace.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Downsample a waypoint CSV to a target number of rows."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to the dense input CSV.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination path for the thinned CSV.",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=100,
        help="Desired number of rows in the output (default: 100).",
    )
    parser.add_argument(
        "--has-header",
        action="store_true",
        help="Set if the CSV contains a header row that should be preserved.",
    )
    return parser.parse_args()


def load_rows(path: Path, has_header: bool) -> tuple[List[str], List[List[str]]]:
    with path.open(newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None) if has_header else None
        rows = [row for row in reader if row]
    if not rows:
        raise ValueError(f"No data rows found in {path}")
    return header, rows


def pick_indices(n_rows: int, target: int) -> List[int]:
    if target <= 0:
        raise ValueError("Target row count must be positive.")
    if target >= n_rows:
        return list(range(n_rows))

    # Spread indices evenly, always including first and last row.
    last_idx = n_rows - 1
    denom = max(target - 1, 1)
    idxs = {round(i * last_idx / denom) for i in range(target)}
    ordered = sorted(idxs)

    # Guarantee first/last rows even if rounding collapsed them.
    if ordered[0] != 0:
        ordered.insert(0, 0)
    if ordered[-1] != last_idx:
        ordered.append(last_idx)

    # If duplicates reduced the count, fill greedily.
    while len(ordered) < target:
        for candidate in range(last_idx + 1):
            if candidate not in ordered:
                ordered.append(candidate)
                if len(ordered) == target:
                    break
    return sorted(ordered)


def write_rows(path: Path, header: List[str] | None, rows: List[List[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(header)
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    header, rows = load_rows(args.input, args.has_header)
    indices = pick_indices(len(rows), args.count)
    sampled = [rows[i] for i in indices]
    write_rows(args.output, header, sampled)
    print(
        f"Wrote {len(sampled)} rows to {args.output} "
        f"(from {len(rows)} rows in {args.input})"
    )


if __name__ == "__main__":
    main()
