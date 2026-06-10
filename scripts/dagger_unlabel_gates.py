"""Strip all appended gate streams from .pkl.xz files in a labeled failure dir.

The labeler (dagger_label_gates.py) appends each gate record as an additional
xz stream concatenated onto the main pickle. This script detects multi-stream
xz files and truncates them back to just the first stream (the main pickle).

The matching .preview.mp4 mtime is then bumped to be newer than the truncated
pkl, so dagger_label_gates.py's `pkl.mtime > preview.mtime` heuristic correctly
re-classifies the file as unlabeled on the next run.

Usage:
    python scripts/dagger_unlabel_gates.py dataset/raw/sim/one_leg/dagger_iter0_ah1_nm/low/failure
    python scripts/dagger_unlabel_gates.py <dir> --dry-run
"""

import argparse
import lzma
import os
from pathlib import Path


def _first_stream_end(pkl_path: Path):
    """Decompress the first xz stream and return its end-byte offset in the file.

    Returns None if the file is single-stream (nothing to strip) or malformed.
    """
    data = pkl_path.read_bytes()
    dec = lzma.LZMADecompressor()
    try:
        dec.decompress(data)
    except lzma.LZMAError as e:
        print(f"  {pkl_path.name}: LZMA error ({e}); skipping")
        return None
    if not dec.eof:
        print(f"  {pkl_path.name}: first stream did not complete; skipping")
        return None
    unused = len(dec.unused_data)
    if unused == 0:
        return None  # single-stream, no label appended
    return len(data) - unused


def _preview_for(pkl_path: Path) -> Path:
    name = pkl_path.name
    stem = name[: -len(".pkl.xz")] if name.endswith(".pkl.xz") else pkl_path.stem
    return pkl_path.parent / f"{stem}.preview.mp4"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("failure_dir", help="Path to dagger_iterN/{randomness}/failure/")
    parser.add_argument("--dry-run", action="store_true", help="Report what would be stripped without modifying files.")
    args = parser.parse_args()

    failure_dir = Path(args.failure_dir).expanduser().resolve()
    if not failure_dir.exists():
        print(f"ERROR: {failure_dir} does not exist.")
        return

    pkls = sorted(failure_dir.glob("*.pkl.xz"))
    if not pkls:
        print(f"No .pkl.xz files in {failure_dir}.")
        return

    n_stripped = 0
    for pkl in pkls:
        end = _first_stream_end(pkl)
        if end is None:
            continue
        before = pkl.stat().st_size
        if args.dry_run:
            print(f"  {pkl.name}: would truncate {before} -> {end} bytes (strip {before - end})")
        else:
            with open(pkl, "r+b") as f:
                f.truncate(end)
            # Bump preview mtime past the truncated pkl's mtime so the labeler
            # re-classifies as unlabeled on the next run.
            preview = _preview_for(pkl)
            if preview.exists():
                new_mtime = pkl.stat().st_mtime + 1
                os.utime(preview, (new_mtime, new_mtime))
            print(f"  {pkl.name}: stripped {before - end} bytes ({before} -> {end})")
        n_stripped += 1

    verb = "would strip" if args.dry_run else "stripped"
    print(f"\nDone. {verb} labels from {n_stripped} of {len(pkls)} pkls.")


if __name__ == "__main__":
    main()
