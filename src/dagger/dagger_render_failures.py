"""
DAgger Step 2a (cluster-side): render MP4 previews from failure pickles.

For each *.pkl[.xz] under the given failure dir, writes a sibling
*.preview.mp4 of `color_image1 | color_image2` with the frame index burned
into every frame. The user reads the gate frame directly off the video using
any local video player — no GUI is needed on the cluster.

Idempotent: previews that already exist are skipped unless --overwrite.

Usage (on the SLURM cluster, after dagger_collect_failures.py):
    python src/dagger/dagger_render_failures.py dataset/raw/sim/one_leg/dagger_iter0/low/failure
"""

import argparse
import lzma
import pickle
from pathlib import Path

import cv2
import imageio
import numpy as np


def _load_failure(pkl_path: Path) -> dict:
    if pkl_path.suffix == ".xz":
        with lzma.open(pkl_path, "rb") as f:
            return pickle.load(f)
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def _preview_path_for(pkl_path: Path) -> Path:
    name = pkl_path.name
    if name.endswith(".pkl.xz"):
        stem = name[: -len(".pkl.xz")]
    elif name.endswith(".pkl"):
        stem = name[: -len(".pkl")]
    else:
        stem = pkl_path.stem
    return pkl_path.parent / f"{stem}.preview.mp4"


def render_one(pkl_path: Path, out_path: Path, fps: int = 20) -> int:
    """Render a single preview MP4. Returns the number of frames written."""
    data = _load_failure(pkl_path)
    obs = data["observations"]
    # gate must point to a valid snapshot, so quote the snapshot range.
    n_snap = len(data.get("snapshots", []))
    T_max = max(0, n_snap - 1)
    short = pkl_path.name

    # Keep .mp4 at the end so imageio's extension-based format detection still
    # routes through the ffmpeg backend; "<stem>.preview.partial.mp4" is then
    # renamed atomically to "<stem>.preview.mp4". The rsync filter pattern
    # `*.preview.mp4` does NOT match `*.preview.partial.mp4`, so partial files
    # never leak to the labeling laptop.
    tmp_path = out_path.parent / f"{out_path.stem}.partial.mp4"
    with imageio.get_writer(tmp_path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for i, o in enumerate(obs):
            # color_image1, color_image2 are uint8 RGB (240, 320, 3).
            # imageio expects RGB; cv2.putText writes in-place regardless of channel order.
            img = np.concatenate([o["color_image1"], o["color_image2"]], axis=1).copy()  # (240, 640, 3)
            # Frame counter gets a black background rect so it stays legible
            # against the white parts of the scene (table top, robot links).
            counter = f"frame {i:4d} / {T_max}"
            font, scale, thick = cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
            (tw, th), baseline = cv2.getTextSize(counter, font, scale, thick)
            x, y, pad = 10, 30, 4
            cv2.rectangle(img, (x - pad, y - th - pad), (x + tw + pad, y + baseline + pad), (0, 0, 0), thickness=-1)
            cv2.putText(img, counter, (x, y), font, scale, (255, 255, 255), thick)

            cv2.putText(img, short, (10, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            writer.append_data(img)
    tmp_path.rename(out_path)  # atomic so partial files never leak to the rsync pull
    return len(obs)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("failure_dir", help="Path to dagger_iterN/{randomness}/failure/")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render previews even when the *.preview.mp4 already exists.",
    )
    args = parser.parse_args()

    failure_dir = Path(args.failure_dir)
    pkls = sorted(failure_dir.glob("*.pkl.xz")) + sorted(failure_dir.glob("*.pkl"))
    if not pkls:
        print(f"No failure pkls found in {failure_dir}")
        return

    n_done = 0
    n_skip = 0
    for pkl in pkls:
        out = _preview_path_for(pkl)
        if out.exists() and not args.overwrite:
            n_skip += 1
            continue
        print(f"  rendering {pkl.name}  ->  {out.name}", flush=True)
        frames = render_one(pkl, out, fps=args.fps)
        print(f"    wrote {frames} frames")
        n_done += 1
    print(f"\nDone. rendered {n_done}, skipped {n_skip} ({len(pkls)} total).")


if __name__ == "__main__":
    main()
