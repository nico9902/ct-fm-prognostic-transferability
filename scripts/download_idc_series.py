#!/usr/bin/env python3
"""Download an explicit list of IDC SeriesInstanceUIDs."""

from __future__ import annotations

import argparse
from pathlib import Path

from idc_index import IDCClient


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-uids", type=Path, required=True)
    parser.add_argument("--download-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    uids = sorted(
        {
            line.strip()
            for line in args.series_uids.read_text().splitlines()
            if line.strip()
        }
    )
    if not uids:
        raise SystemExit(f"No SeriesInstanceUIDs found in {args.series_uids}")
    args.download_dir.mkdir(parents=True, exist_ok=True)
    print(f"Series requested: {len(uids)}")
    IDCClient().download_from_selection(
        seriesInstanceUID=uids,
        downloadDir=str(args.download_dir),
        dry_run=args.dry_run,
        quiet=False,
        show_progress_bar=True,
    )


if __name__ == "__main__":
    main()
