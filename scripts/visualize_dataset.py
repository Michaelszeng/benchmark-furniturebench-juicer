"""Interactively view episodes from a processed zarr dataset."""

import argparse
import importlib
import io
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pytorch3d.transforms as pt
import torch

matplotlib.use("Agg")  # off-screen rendering; must be set before pyplot import
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 — registers 3d projection


def rot6d_to_mat(r6d: np.ndarray) -> np.ndarray:
    """Convert a (6,) 6D rotation vector to a (3, 3) rotation matrix."""
    t = torch.from_numpy(r6d).float().unsqueeze(0)  # (1, 6)
    mat = pt.rotation_6d_to_matrix(t)  # (1, 3, 3)
    return mat.squeeze(0).numpy()  # (3, 3)


# ---------------------------------------------------------------------------
# Per-episode state/action data loader
# ---------------------------------------------------------------------------


def load_episode_state_action(dataset, episode_ends, ep):
    start = 0 if ep == 0 else episode_ends[ep - 1]
    end = episode_ends[ep]

    rs = dataset["robot_state"][start:end]  # (T, 16)
    ee_pos = rs[:, 0:3]  # (T, 3)  x, y, z  (relative to base)
    ee_rot6d = rs[:, 3:9]  # (T, 6)

    # Try to load action/delta (non-translated zarr) or action (translated)
    if "action" in dataset and hasattr(dataset["action"], "shape"):
        # translated format: flat array under root or data/
        act = dataset["action"][start:end]
    elif "action/delta" in dataset:
        act = dataset["action/delta"][start:end]
    else:
        act = None

    delta_pos = act[:, 0:3] if act is not None else None  # (T, 3)
    delta_rot6d = act[:, 3:9] if act is not None else None  # (T, 6)
    gripper = act[:, 9] if act is not None else None  # (T,)

    return ee_pos, ee_rot6d, delta_pos, delta_rot6d, gripper


# ---------------------------------------------------------------------------
# Matplotlib panel renderer
# ---------------------------------------------------------------------------


def render_state_panel(
    ee_pos: np.ndarray,  # (T, 3) full episode trajectory
    ee_rot6d: np.ndarray,  # (T, 6)
    delta_pos: np.ndarray,  # (T, 3) or None
    delta_rot6d: np.ndarray,  # (T, 6) or None
    gripper: np.ndarray,  # (T,) or None
    frame_idx: int,
    panel_h: int = 480,
    panel_w: int = 480,
) -> np.ndarray:
    """Render a matplotlib panel for the given episode state/action data.

    Returns an (panel_h, panel_w, 3) uint8 RGB image.
    """
    T = len(ee_pos)
    fig = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100)

    # ---- 3D trajectory plot (top, larger) ----
    ax3d = fig.add_axes([0.05, 0.38, 0.90, 0.58], projection="3d")

    # Full trajectory as a faint gray line
    ax3d.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], color="lightgray", linewidth=0.8, zorder=1)

    # Colour the trajectory from blue (start) to red (end) to show time
    for t in range(0, T - 1, max(1, T // 60)):
        frac = t / max(T - 1, 1)
        color = (frac, 0.0, 1.0 - frac)
        ax3d.plot(
            ee_pos[t : t + 2, 0], ee_pos[t : t + 2, 1], ee_pos[t : t + 2, 2], color=color, linewidth=1.5, zorder=2
        )

    # Current position
    cx, cy, cz = ee_pos[frame_idx]
    ax3d.scatter([cx], [cy], [cz], color="yellow", s=60, zorder=5, edgecolors="black", linewidths=0.5)

    # Current orientation axes (scaled to ~5 % of the trajectory range)
    scale = max(np.ptp(ee_pos, axis=0).max() * 0.07, 0.01)
    rot = rot6d_to_mat(ee_rot6d[frame_idx])
    axis_colors = ["red", "green", "blue"]
    axis_labels = ["X", "Y", "Z"]
    for i, (col, lbl) in enumerate(zip(axis_colors, axis_labels)):
        dx, dy, dz = rot[:, i] * scale
        ax3d.quiver(cx, cy, cz, dx, dy, dz, color=col, linewidth=1.5, arrow_length_ratio=0.3)

    # Current action delta arrow
    if delta_pos is not None:
        dp = delta_pos[frame_idx]
        mag = np.linalg.norm(dp)
        if mag > 1e-6:
            dp_scaled = dp / mag * min(mag * 5, scale * 1.5)
            ax3d.quiver(
                cx,
                cy,
                cz,
                dp_scaled[0],
                dp_scaled[1],
                dp_scaled[2],
                color="orange",
                linewidth=2,
                linestyle="dashed",
                arrow_length_ratio=0.3,
            )

    ax3d.set_xlabel("X", fontsize=7, labelpad=0)
    ax3d.set_ylabel("Y", fontsize=7, labelpad=0)
    ax3d.set_zlabel("Z", fontsize=7, labelpad=0)
    ax3d.tick_params(labelsize=6, pad=0)
    ax3d.set_title(f"EE trajectory  (frame {frame_idx}/{T - 1})", fontsize=8, pad=2)

    # ---- Time-series plot (bottom) ----
    ax_ts = fig.add_axes([0.10, 0.04, 0.85, 0.28])
    t_axis = np.arange(T)

    ax_ts.plot(t_axis, ee_pos[:, 0], color="red", linewidth=0.8, label="X")
    ax_ts.plot(t_axis, ee_pos[:, 1], color="green", linewidth=0.8, label="Y")
    ax_ts.plot(t_axis, ee_pos[:, 2], color="blue", linewidth=0.8, label="Z")

    if gripper is not None:
        ax_ts.plot(t_axis, gripper * 0.03, color="purple", linewidth=0.8, linestyle="dotted", label="grip×0.03")

    ax_ts.axvline(frame_idx, color="yellow", linewidth=1.2, zorder=5)
    ax_ts.set_xlim(0, T - 1)
    ax_ts.tick_params(labelsize=6)
    ax_ts.set_ylabel("ee_pos (m)", fontsize=7)
    ax_ts.legend(fontsize=6, loc="upper right", ncol=2)
    ax_ts.set_facecolor("#1a1a1a")
    ax_ts.grid(color="gray", linewidth=0.3)

    fig.patch.set_facecolor("#1a1a1a")
    ax3d.set_facecolor("#1a1a1a")
    ax3d.grid(True, linewidth=0.3)

    # Render to numpy array via PNG encode/decode
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#1a1a1a", dpi=100)
    plt.close(fig)
    buf.seek(0)
    import cv2 as _cv2

    arr = np.frombuffer(buf.getvalue(), np.uint8)
    panel_bgr = _cv2.imdecode(arr, _cv2.IMREAD_COLOR)
    panel_rgb = _cv2.cvtColor(panel_bgr, _cv2.COLOR_BGR2RGB)

    # Resize to exact panel dimensions
    if panel_rgb.shape[:2] != (panel_h, panel_w):
        panel_rgb = _cv2.resize(panel_rgb, (panel_w, panel_h))

    return panel_rgb


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


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
    parser.add_argument(
        "--state",
        action="store_true",
        help="Show the 3D state/action matplotlib panel below the camera feeds.",
    )
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

    # Detect whether state/action data is available
    has_state = args.state and "robot_state" in dataset

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

    # Camera images are 240×320 each; concatenated strip is 240×640.
    # Scale 2× → 480×1280. State panel (if shown) is stacked below at full width.
    cam_h = 480
    cam_w = 1280  # 640 * 2 (2× uniform scale)
    state_h = 720  # height of the state/action panel when shown
    win_w = cam_w
    win_h = cam_h + state_h if has_state else cam_h
    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Viewer", win_w, win_h)

    def load_episode(ep):
        start = 0 if ep == 0 else episode_ends[ep - 1]
        end = episode_ends[ep]
        imgs1 = dataset["color_image1"][start:end]
        imgs2 = dataset["color_image2"][start:end]
        success = bool(dataset["success"][ep])
        camera_frames = np.concatenate([imgs1, imgs2], axis=2)  # (T, H, 640, 3)

        state_data = None
        if has_state:
            state_data = load_episode_state_action(dataset, episode_ends, ep)

        return camera_frames, success, state_data

    camera_frames, success, state_data = load_episode(ep_idx)
    frame_idx = 0

    # Cache rendered matplotlib panels per episode to avoid re-rendering each frame.
    # Panels are rendered lazily and cached in a list indexed by frame.
    panel_cache: list = []

    def get_panel(fi):
        nonlocal panel_cache
        if not panel_cache:
            panel_cache = [None] * len(camera_frames)
        if panel_cache[fi] is None:
            ee_pos, ee_rot6d, delta_pos, delta_rot6d, gripper = state_data
            panel_cache[fi] = render_state_panel(
                ee_pos,
                ee_rot6d,
                delta_pos,
                delta_rot6d,
                gripper,
                fi,
                panel_h=state_h,
                panel_w=cam_w,
            )
        return panel_cache[fi]

    def on_episode_change(ep):
        nonlocal camera_frames, success, state_data, panel_cache
        camera_frames, success, state_data = load_episode(ep)
        panel_cache = []

    while True:
        cam = camera_frames[frame_idx]
        # Camera images are stored RGB; convert to BGR for OpenCV display
        cam_bgr = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
        # Scale uniformly: 240×640 → 480×1280 (2× in both dimensions)
        cam_bgr = cv2.resize(cam_bgr, (cam_w, cam_h))

        if has_state:
            panel_rgb = get_panel(frame_idx)
            panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
            display = np.concatenate([cam_bgr, panel_bgr], axis=0)
        else:
            display = cam_bgr

        label = (
            f"Ep {ep_idx + 1}/{n_episodes}  |  "
            f"Frame {frame_idx}/{len(camera_frames) - 1}  |  "
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
            if frame_idx < len(camera_frames) - 1:
                frame_idx += 1
            else:
                playing = False
        elif key == ord("l"):
            frame_idx = min(frame_idx + 10, len(camera_frames) - 1)
        elif key == ord("j"):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord("h"):
            frame_idx = max(frame_idx - 10, 0)
        elif key == ord("n"):
            ep_idx = min(ep_idx + 1, n_episodes - 1)
            on_episode_change(ep_idx)
            frame_idx = 0
            playing = False
        elif key == ord("p"):
            ep_idx = max(ep_idx - 1, 0)
            on_episode_change(ep_idx)
            frame_idx = 0
            playing = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
