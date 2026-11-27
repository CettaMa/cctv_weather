"""Membandingkan CSV dua folder hasil (mengabaikan kolom waktu proses).

    python -m pkji.compare <folder_acuan> <folder_baru>
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

TIMING_COLUMNS = {"average_inference_ms", "processing_fps"}


def read(path: Path) -> list:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def as_float(text: str):
    try:
        return float(text)
    except ValueError:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path)
    ap.add_argument("candidate", type=Path)
    args = ap.parse_args()
    different = False
    for ref_path in sorted(args.reference.glob("*.csv")):
        new_path = args.candidate / ref_path.name
        if not new_path.is_file():
            print(f"{ref_path.name}: TIDAK ADA di kandidat")
            different = True
            continue
        a, b = read(ref_path), read(new_path)
        if len(a) != len(b):
            print(f"{ref_path.name}: jumlah baris {len(a)} vs {len(b)}")
            different = True
            continue
        columns = {}  # kolom -> [baris beda, selisih absolut, total absolut acuan]
        for row_a, row_b in zip(a, b):
            for key, value in row_a.items():
                if key in TIMING_COLUMNS:
                    continue
                other = row_b.get(key, "")
                x, y = as_float(value), as_float(other)
                stat = columns.setdefault(key, [0, 0.0, 0.0])
                if x is not None and y is not None:
                    stat[1] += abs(x - y)
                    stat[2] += abs(x)
                    stat[0] += x != y
                else:
                    stat[0] += value != other
        changed = {k: v for k, v in columns.items() if v[0]}
        different |= bool(changed)
        print(f"{ref_path.name}: {len(a)} baris, " + ("identik" if not changed else "berbeda:"))
        for key, (n, delta, base) in changed.items():
            rel = f", selisih relatif {delta / base:.2%}" if base else ""
            print(f"    {key}: {n} baris{rel}")
    return int(different)


if __name__ == "__main__":
    raise SystemExit(main())
