#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Mirror CCTV stream links and coordinates from the live Pelindung Bandung API.

Sync semantics (mirror):
  - The live API is the source of truth for which cameras exist, their stream
    URLs, coordinates (Location), and provider.
  - Cameras present in the live API but missing from local data are added.
  - Existing cameras matching a live entry keep their previously captured sample
    metadata if their m3u8_URL has not changed; if the URL changed, sample
    metadata is reset. Location and Provider are kept up to date with the site.
  - Cameras that no longer appear in the live API are dropped.

Run from the repo root: python script_tools/update_cctv_links.py
"""
from __future__ import annotations

import csv
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from scrape_bandung_cctv import API_URL, CSV_HEADERS, CCTV, fetch_cctv_data  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent.parent
VALIDATED_CSV = BASE_DIR / "csv" / "cctv_bandung_data_validated.csv"
BASE_CSV = BASE_DIR / "csv" / "cctv_bandung_data.csv"

BASE_HEADERS = ["CCTV_Name", "Location", "Provider", "m3u8_URL"]


def normalize_name(name: str) -> str:
    return re.sub(r"\s+", " ", name.strip().upper())


def current_utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def load_existing_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sync_rows(existing_rows: list[dict], scraped: list[CCTV]) -> tuple[list[dict], dict[str, int]]:
    """
    Mirror scraped CCTV entries against existing rows.

    Returns:
        (synced_rows, {"added": int, "updated": int, "removed": int, "unchanged": int})
    """
    existing_by_name: dict[str, dict] = {}
    for r in existing_rows:
        key = normalize_name(r.get("CCTV_Name", ""))
        if key:
            existing_by_name.setdefault(key, r)

    seen_keys: set[str] = set()
    synced_rows: list[dict] = []
    added = 0
    updated = 0
    unchanged = 0

    for live in scraped:
        key = normalize_name(live.name)
        if not key or key in seen_keys:
            continue
        seen_keys.add(key)

        old = existing_by_name.get(key)
        if old is None:
            synced_rows.append({
                "CCTV_Name": live.name,
                "Location": live.location,
                "Provider": live.provider,
                "m3u8_URL": live.stream_url,
                "Sample_Downloaded": "NO",
                "Sample_File": "",
                "Sample_Size_Bytes": "",
                "Sample_Error": "Not sampled (added via CCTV link scrape)",
                "Validated_At_UTC": current_utc_timestamp(),
            })
            added += 1
        else:
            old_projected = {h: old.get(h, "") for h in CSV_HEADERS}
            row = dict(old_projected)
            row["CCTV_Name"] = live.name
            row["Location"] = live.location
            row["Provider"] = live.provider

            if old.get("m3u8_URL", "").strip() != live.stream_url.strip():
                logging.info(
                    "Updating link for %s: %s -> %s",
                    live.name, old.get("m3u8_URL"), live.stream_url,
                )
                row["m3u8_URL"] = live.stream_url
                row["Sample_Downloaded"] = "NO"
                row["Sample_File"] = ""
                row["Sample_Size_Bytes"] = ""
                row["Sample_Error"] = "Link updated from live API; sample not re-validated"
                row["Validated_At_UTC"] = current_utc_timestamp()

            if row != old_projected:
                updated += 1
            else:
                unchanged += 1
            synced_rows.append(row)

    removed = len(existing_by_name) - len(seen_keys.intersection(existing_by_name.keys()))
    stats = {
        "added": added,
        "updated": updated,
        "removed": removed,
        "unchanged": unchanged,
    }
    return synced_rows, stats


def write_validated_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in CSV_HEADERS})
    os.replace(tmp_path, path)


def write_base_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=BASE_HEADERS)
        writer.writeheader()
        for row in rows:
            writer.writerow({h: row.get(h, "") for h in BASE_HEADERS})
    os.replace(tmp_path, path)


def sync_catalog_files(request_timeout: float = 20.0) -> tuple[list[dict], dict]:
    """
    Fetch live CCTV records from Pelindung, mirror them against local CSVs,
    and persist updates atomically if changed.
    """
    scraped = fetch_cctv_data(API_URL, request_timeout)
    if not scraped:
        raise ValueError("Pelindung API returned no CCTV records; keeping existing catalog")

    existing = load_existing_rows(VALIDATED_CSV)
    rows, stats = sync_rows(existing, scraped)

    existing_projected = [{h: r.get(h, "") for h in CSV_HEADERS} for r in existing]
    has_changed = rows != existing_projected
    stats["changed"] = has_changed

    if has_changed:
        try:
            write_validated_csv(rows, VALIDATED_CSV)
            write_base_csv(rows, BASE_CSV)
        except OSError as exc:
            logging.warning("Could not write synced catalog CSVs: %s", exc)
            stats["write_error"] = str(exc)

    return rows, stats


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    try:
        rows, stats = sync_catalog_files()
    except Exception:
        logging.exception("Failed to sync CCTV links from Pelindung API (%s)", API_URL)
        return 1

    suffix = "" if stats.get("changed") else " — no changes, files untouched"
    logging.info(
        "Done: %d rows (+%d added, %d updated, -%d removed, %d unchanged)%s",
        len(rows), stats["added"], stats["updated"], stats["removed"], stats["unchanged"], suffix,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
