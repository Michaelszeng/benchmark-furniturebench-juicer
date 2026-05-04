"""Interactively view episodes from pkl/pkl.xz files.

Works with both the original DataCollector output and the DART-rollback
output from test_state_rollback_v2.py.  Only the noisy-trajectory
observations are displayed; action_chunks (if present) are ignored.

Usage:
    python scripts/visualize_pkl.py /path/to/pkl/dir
    python scripts/visualize_pkl.py /path/to/pkl/dir --state  # show EE panel
    python scripts/visualize_pkl.py /path/to/pkl/dir -e 3     # start at ep 3

Controls:
  k / l     step 1 / 10 frames forward
  j / h     step 1 / 10 frames backward
  n / p     next / previous episode
  Space     toggle play/pause
  q         quit
"""

import argparse
import io
import lzma
import pickle
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  — registers 3d projection


# ── Data loading ──────────────────────────────────────────────────────────────


def _load_pkl(path: Path) -> dict:
    if path.suffix == ".xz":
        with lzma.open(path, "rb") as f:
            return pickle.load(f)
    with open(path, "rb") as f:
        return pickle.load(f)


def _extract(data: dict):
    """Return (camera_frames, ee_pos, gripper) from a pkl dict.

    camera_frames : (T, H, W*2, 3) uint8 RGB  — images1 and images2 side-by-side
    ee_pos        : (T, 3) float               — end-effector world position
    gripper       : (T,)   float               — gripper width
    """
    obs_list = data["observations"]
    imgs1 = np.stack([o["color_image1"] for o in obs_list])  # (T, H, W, 3)
    imgs2 = np.stack([o["color_image2"] for o in obs_list])  # (T, H, W, 3)
    camera = np.concatenate([imgs1, imgs2], axis=2)           # (T, H, W*2, 3)

    rs_list = [o.get("robot_state", {}) for o in obs_list]
    if rs_list and "ee_pos" in rs_list[0]:
        ee_pos = np.stack([rs["ee_pos"].flatten()[:3] for rs in rs_list])
        gripper = np.stack([np.asarray(rs["gripper_width"]).flatten()[0] for rs in rs_list])
    else:
        ee_pos = np.zeros((len(obs_list), 3))
        gripper = np.zeros(len(obs_list))

    return camera, ee_pos, gripper


# ── State panel ───────────────────────────────────────────────────────────────


def _render_state_panel(ee_pos, gripper, frame_idx, panel_h, panel_w):
    """Matplotlib panel: 3D EE trajectory (left) + time-series (right)."""
    T = len(ee_pos)
    fig = plt.figure(figsize=(panel_w / 100, panel_h / 100), dpi=100)

    # 3D trajectory
    ax3d = fig.add_axes([0.03, 0.08, 0.42, 0.87], projection="3d")
    ax3d.plot(ee_pos[:, 0], ee_pos[:, 1], ee_pos[:, 2], color="lightgray", linewidth=0.8, zorder=1)
    for t in range(0, T - 1, max(1, T // 60)):
        frac = t / max(T - 1, 1)
        ax3d.plot(
            ee_pos[t : t + 2, 0], ee_pos[t : t + 2, 1], ee_pos[t : t + 2, 2],
            color=(frac, 0.0, 1.0 - frac), linewidth=1.5, zorder=2,
        )
    cx, cy, cz = ee_pos[frame_idx]
    ax3d.scatter([cx], [cy], [cz], color="yellow", s=60, zorder=5, edgecolors="black", linewidths=0.5)
    ax3d.set_xlabel("X", fontsize=7, labelpad=0)
    ax3d.set_ylabel("Y", fontsize=7, labelpad=0)
    ax3d.set_zlabel("Z", fontsize=7, labelpad=0)
    ax3d.tick_params(labelsize=6, pad=0)
    ax3d.set_title(f"EE trajectory  (frame {frame_idx}/{T - 1})", fontsize=8, pad=2)
    ax3d.set_facecolor("#1a1a1a")

    # Time series
    ax_ts = fig.add_axes([0.52, 0.12, 0.44, 0.78])
    t_axis = np.arange(T)
    ax_ts.plot(t_axis, ee_pos[:, 0], color="red",   linewidth=0.8, label="X")
    ax_ts.plot(t_axis, ee_pos[:, 1], color="green", linewidth=0.8, label="Y")
    ax_ts.plot(t_axis, ee_pos[:, 2], color="blue",  linewidth=0.8, label="Z")
    ax_ts.plot(t_axis, gripper * 3,  color="purple", linewidth=0.8, linestyle="dotted", label="grip×3")
    ax_ts.axvline(frame_idx, color="yellow", linewidth=1.2, zorder=5)
    ax_ts.set_xlim(0, T - 1)
    ax_ts.tick_params(labelsize=6)
    ax_ts.set_ylabel("ee_pos (m)", fontsize=7)
    ax_ts.legend(fontsize=6, loc="upper right", ncol=2)
    ax_ts.set_facecolor("#1a1a1a")
    ax_ts.grid(color="gray", linewidth=0.3)

    fig.patch.set_facecolor("#1a1a1a")
    buf = io.BytesIO()
    fig.savefig(buf, format="png", facecolor="#1a1a1a", dpi=100)
    plt.close(fig)
    buf.seek(0)
    arr = np.frombuffer(buf.getvalue(), np.uint8)
    panel_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    panel_rgb = cv2.cvtColor(panel_bgr, cv2.COLOR_BGR2RGB)
    if panel_rgb.shape[:2] != (panel_h, panel_w):
        panel_rgb = cv2.resize(panel_rgb, (panel_w, panel_h))
    return panel_rgb


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pkl_dir", help="Directory containing .pkl or .pkl.xz files (searched recursively)")
    parser.add_argument("--episode", "-e", type=int, default=0, help="Episode index to start at (default: 0)")
    parser.add_argument("--fps", type=int, default=10, help="Playback speed in fps (default: 10)")
    parser.add_argument("--state", action="store_true", help="Show 3D EE trajectory + time-series panel")
    args = parser.parse_args()

    pkl_dir = Path(args.pkl_dir)
    pkl_files = sorted(list(pkl_dir.rglob("*.pkl.xz")) + list(pkl_dir.rglob("*.pkl")))
    if not pkl_files:
        print(f"No .pkl or .pkl.xz files found under {pkl_dir}")
        sys.exit(1)

    print(f"Found {len(pkl_files)} episode(s) under {pkl_dir}")
    for i, p in enumerate(pkl_files):
        print(f"  [{i}] {p.relative_to(pkl_dir)}")
    print()
    print("Controls:  k/l  step ±1/10  |  j/h  step ±1/10  |  n/p  episode  |  Space  play/pause  |  q  quit")

    ep_idx = max(0, min(args.episode, len(pkl_files) - 1))
    frame_delay_ms = max(1, 1000 // args.fps)
    playing = False

    def load_ep(idx):
        data = _load_pkl(pkl_files[idx])
        cam, ee_pos, gripper = _extract(data)
        return cam, data.get("success", False), ee_pos, gripper

    camera_frames, success, ee_pos, gripper = load_ep(ep_idx)
    frame_idx = 0
    panel_cache: list = []

    # Determine display size from actual image dimensions.
    # Images are stored as (H, W, 3); two concatenated → (H, W*2, 3).
    src_h, src_w = camera_frames.shape[1], camera_frames.shape[2]
    # Scale so the camera strip is ~1280 px wide (2× for 224×224 → 448→896, or
    # native for 240×640 → match existing viewer).  Keep an integer scale factor.
    scale = max(1, 1280 // src_w)
    disp_w = src_w * scale
    disp_h = src_h * scale
    state_h = disp_h  # state panel same height as camera strip
    win_h = disp_h + state_h if args.state else disp_h

    cv2.namedWindow("PKL Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("PKL Viewer", disp_w, win_h)

    def get_panel(fi):
        nonlocal panel_cache
        if not panel_cache:
            panel_cache = [None] * len(camera_frames)
        if panel_cache[fi] is None:
            panel_cache[fi] = _render_state_panel(ee_pos, gripper, fi, state_h, disp_w)
        return panel_cache[fi]

    while True:
        cam = camera_frames[frame_idx]
        cam_bgr = cv2.cvtColor(cam, cv2.COLOR_RGB2BGR)
        cam_bgr = cv2.resize(cam_bgr, (disp_w, disp_h))

        if args.state:
            panel_rgb = get_panel(frame_idx)
            panel_bgr = cv2.cvtColor(panel_rgb, cv2.COLOR_RGB2BGR)
            display = np.concatenate([cam_bgr, panel_bgr], axis=0)
        else:
            display = cam_bgr

        label = (
            f"Ep {ep_idx + 1}/{len(pkl_files)}  |  "
            f"Frame {frame_idx}/{len(camera_frames) - 1}  |  "
            f"{'SUCCESS' if success else 'FAILURE'}  |  "
            f"{pkl_files[ep_idx].name}"
        )
        cv2.putText(display, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow("PKL Viewer", display)

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
            ep_idx = min(ep_idx + 1, len(pkl_files) - 1)
            camera_frames, success, ee_pos, gripper = load_ep(ep_idx)
            panel_cache = []
            frame_idx = 0
            playing = False
        elif key == ord("p"):
            ep_idx = max(ep_idx - 1, 0)
            camera_frames, success, ee_pos, gripper = load_ep(ep_idx)
            panel_cache = []
            frame_idx = 0
            playing = False

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
