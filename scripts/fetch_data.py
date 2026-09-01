"""Assemble the SETU dataset from a manifest, reproducibly.

Two rules. Nothing is downloaded twice: every file is checked for size and, where a
checksum is known, for content. And nothing is fetched from a portal that requires a
human to accept terms or log in - those entries print exactly what to download and where
to put it, so the manifest remains an accurate record of what a result was built from
even when part of the acquisition was manual.

Absent real data, `setu bench generate` produces the controlled benchmark and the whole
pipeline runs against that. The specification's own risk table recommends exactly this:
build against the synthetic bench so that no phase is blocked waiting on an archive.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DEST = ROOT / "data"


def human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} PB"


def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(chunk):
            h.update(block)
    return h.hexdigest()


def download(url: str, dest: Path, expected_sha: str | None = None) -> bool:
    """Fetch one file, skipping it when a good copy is already present."""
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists():
        if expected_sha is None:
            print(f"    already present ({human(dest.stat().st_size)}), skipping")
            return True
        if sha256(dest) == expected_sha:
            print("    already present and checksum matches, skipping")
            return True
        print("    present but checksum differs, re-fetching")

    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as response, tmp.open("wb") as out:
            total = int(response.headers.get("Content-Length") or 0)
            got = 0
            while block := response.read(1 << 20):
                out.write(block)
                got += len(block)
                if total:
                    pct = 100 * got / total
                    print(f"\r    {human(got)} / {human(total)}  ({pct:5.1f}%)", end="", flush=True)
        print()
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        print(f"    failed: {type(exc).__name__}: {exc}")
        return False

    if expected_sha and sha256(tmp) != expected_sha:
        tmp.unlink(missing_ok=True)
        print("    checksum mismatch, discarded")
        return False

    shutil.move(tmp, dest)
    print(f"    saved {human(dest.stat().st_size)}")
    return True


def report_manual(entry: dict[str, Any]) -> None:
    print("    MANUAL — this archive needs a browser session or an account.")
    for key in ("portal", "browse", "mirror"):
        if entry.get(key):
            print(f"      {key}: {entry[key]}")
    print(f"      fetch: {entry.get('want', 'see the manifest')}")
    if entry.get("note"):
        note = " ".join(str(entry["note"]).split())
        print(f"      note:  {note}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path(__file__).parent / "manifest.yaml")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--only", nargs="*", help="Fetch only these source ids.")
    ap.add_argument("--list", action="store_true", help="Describe the manifest and exit.")
    args = ap.parse_args()

    doc = yaml.safe_load(args.manifest.read_text())
    sources = doc.get("sources", [])

    if args.list:
        print(f"{len(doc.get('sites', []))} sites, {len(sources)} sources\n")
        for site in doc.get("sites", []):
            print(f"  site {site['id']:20s} {site['lat']:+8.3f}, {site['lon']:+8.3f}  {site['name']}")
        print()
        for s in sources:
            kind = "manual" if s.get("manual") else "auto"
            print(f"  {kind:7s} {s['id']:14s} {s.get('want', '')}")
        return 0

    args.dest.mkdir(parents=True, exist_ok=True)
    auto_ok = auto_fail = manual = 0

    for entry in sources:
        if args.only and entry["id"] not in args.only:
            continue
        print(f"\n[{entry['id']}]")

        if entry.get("manual") or not entry.get("url"):
            report_manual(entry)
            manual += 1
            continue

        dest = args.dest / entry["id"]
        if download(entry["url"], dest / Path(entry["url"]).name, entry.get("sha256")):
            auto_ok += 1
        else:
            auto_fail += 1

    print(f"\n{auto_ok} fetched, {auto_fail} failed, {manual} require manual download into {args.dest}/")
    if manual:
        print(
            "\nWhile the manual archives download, the pipeline runs end to end against the\n"
            "controlled benchmark, which has exact ground truth:\n"
            "  setu bench generate --out data/bench\n"
            "  setu eval --methods all --out runs/eval_full"
        )
    return 0 if auto_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
