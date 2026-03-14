"""Interactively view episodes from a processed zarr dataset."""

import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Interactively view episodes from a processed zarr dataset.")
    parser.add_argument("zarr_path", help="Path to the .zarr dataset file")
    parser.add_argument(
        "--episode",
        "-e",
        type=int,
        default=0,
        help="Episode index to start at (default: 0)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Playback speed in frames per second (default: 10)")
    args = parser.parse_args()

    # src/dataset/zarr.py shadows the installed zarr package when this file is
    # run as a script (Python inserts the script's directory first on sys.path).
    # We must evict the cached local module and re-import with sys.path pruned.
    _script_dir = str(Path(__file__).parent.resolve())
    _saved_path = sys.path[:]
    sys.path = [p for p in sys.path if Path(p).resolve() != Path(_script_dir).resolve()]
    if "zarr" in sys.modules:
        del sys.modules["zarr"]
    zarr_lib = importlib.import_module("zarr")
    sys.path = _saved_path

    dataset = zarr_lib.open(args.zarr_path, mode="r")
    episode_ends = dataset["episode_ends"][:]
    n_episodes = len(episode_ends)

    print(f"Dataset: {args.zarr_path}")
    print(f"Episodes: {n_episodes}  |  Total frames: {episode_ends[-1]}")
    print()
    print(f"Structure:\n{zarr_lib.open_group(args.zarr_path, mode='r').tree()}")
    print()
    print("Controls:")
    print("  k / l     step 1 / 10 frames forward")
    print("  j / h     step 1 / 10 frames backward")
    print("  n / p     next / previous episode")
    print("  Space     toggle play/pause")
    print("  q         quit")
    print()

    ep_idx = max(0, min(args.episode, n_episodes - 1))
    frame_delay_ms = max(1, 1000 // args.fps)
    playing = False

    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Viewer", 1280, 480)

    def load_episode_frames(ep):
        start = 0 if ep == 0 else episode_ends[ep - 1]
        end = episode_ends[ep]
        imgs1 = dataset["color_image1"][start:end]  # (T, H, W, 3)
        imgs2 = dataset["color_image2"][start:end]
        success = bool(dataset["success"][ep])
        return np.concatenate([imgs1, imgs2], axis=2), success

    frames, success = load_episode_frames(ep_idx)
    frame_idx = 0

    while True:
        frame = frames[frame_idx]
        # OpenCV expects BGR; zarr images are stored as RGB
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        label = (
            f"Ep {ep_idx}/{n_episodes - 1}  |  "
            f"Frame {frame_idx}/{len(frames) - 1}  |  "
            f"{'SUCCESS' if success else 'FAILURE'}"
        )
        cv2.putText(display, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Dataset Viewer", display)

        wait_ms = frame_delay_ms if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("k") or (playing and key == 255):
            # advance one frame (also used during auto-play)
            if frame_idx < len(frames) - 1:
                frame_idx += 1
            else:
                playing = False
        elif key == ord("l"):
            frame_idx = min(frame_idx + 10, len(frames) - 1)
        elif key == ord("j"):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord("h"):
            frame_idx = max(frame_idx - 10, 0)
        elif key == ord("n"):
            ep_idx = min(ep_idx + 1, n_episodes - 1)
            frames, success = load_episode_frames(ep_idx)
            frame_idx = 0
            playing = False
        elif key == ord("p"):
            ep_idx = max(ep_idx - 1, 0)
            frames, success = load_episode_frames(ep_idx)
            frame_idx = 0
            playing = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
