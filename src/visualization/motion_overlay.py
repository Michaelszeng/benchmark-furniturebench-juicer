"""Render onion-skin motion overlay figures from raw trajectory pickles.

Loads a raw .pkl / .pkl.xz trajectory, resets a single FurnitureSim env to the
exact recorded state (Franka joint positions + gripper width + part poses) at
each requested timestep, renders each state from a fixed high-resolution
camera, and alpha-composites the robot/part pixels into a single image where
older poses are more translucent than recent ones.

Isaac Gym cannot render translucent materials, so the overlay is done in image
space: each pose is rendered separately along with a segmentation mask of the
robot + furniture parts, and the masked pixels are blended oldest-first with
increasing opacity. The final pose is drawn fully opaque on top.

Workflow (first run — interactive camera framing):
    python -m src.visualization.motion_overlay \
        dataset/raw/sim/one_leg/scripted_non_markovian/low/success/2026-05-08T20:13:58.014276.pkl.xz \
        --exclude-parts square_table_top --timesteps 175 177 179 181 183 185 187 190 193 --alpha-min 0.2 --alpha-max 0.8

    A viewer window opens showing the last requested pose. Pan/orbit to frame
    the shot, press N to cycle through the selected poses to check framing,
    then press C to capture the view and render. The camera pose is saved to
    <out>.camera.json so the shot can be re-rendered without the GUI.

Re-render headless with a saved camera (e.g. different timesteps/alphas):
    python -m src.visualization.motion_overlay <pickle> \
        --timesteps 0 100 200 -1 --camera-pose one_leg_overlay.camera.json

The camera JSON uses a look-at form and may also be hand-written:
    {"pos": [x, y, z], "target": [x, y, z], "fov": 90.0}

Note: keep --fov at 90 (the viewer's default horizontal FOV) if you want the
rendered framing to match what you see in the GUI.
"""

import argparse
import json
import sys
from pathlib import Path

import isaacgym  # noqa: F401  (must be imported before torch)
import numpy as np
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv
from isaacgym import gymapi
from PIL import Image

from src.visualization.render_mp4 import unpickle_data

# Known furniture task names, used to infer the task from the pickle path.
FURNITURE_NAMES = ("one_leg", "lamp", "round_table", "desk", "square_table", "cabinet", "chair", "stool")

# Segmentation IDs: 0 = background/table/obstacle, 1 = robot, 2+ = furniture parts.
SEG_ROBOT = 1
SEG_PARTS_START = 2


class OverlayRenderEnv(FurnitureSimEnv):
    """FurnitureSimEnv with an extra high-resolution capture camera in env 0.

    The camera must be created before prepare_sim() is called, hence the
    set_camera() override rather than creating it after construction.
    """

    overlay_cam_width = 1920
    overlay_cam_height = 1080
    overlay_cam_fov = 90.0  # matches the viewer's default horizontal FOV

    def set_camera(self):
        super().set_camera()
        props = gymapi.CameraProperties()
        props.width = self.overlay_cam_width
        props.height = self.overlay_cam_height
        props.horizontal_fov = self.overlay_cam_fov
        props.near_plane = 0.001
        props.far_plane = 5.0
        self.overlay_cam = self.isaac_gym.create_camera_sensor(self.envs[0], props)


def assign_segmentation_ids(env):
    """Tag every rigid body of the Franka and the furniture parts so
    IMAGE_SEGMENTATION separates them from the static scene (seg id 0).

    Returns a dict mapping part name -> segmentation id.
    """
    gym_ = env.isaac_gym
    e0 = env.envs[0]

    franka_handle = env.franka_handles[0]
    for body_idx in range(gym_.get_actor_rigid_body_count(e0, franka_handle)):
        gym_.set_rigid_body_segmentation_id(e0, franka_handle, body_idx, SEG_ROBOT)

    part_seg_ids = {}
    for part_idx, part in enumerate(env.furniture.parts):
        part_handle = env.handles[part.name]
        part_seg_ids[part.name] = SEG_PARTS_START + part_idx
        for body_idx in range(gym_.get_actor_rigid_body_count(e0, part_handle)):
            gym_.set_rigid_body_segmentation_id(e0, part_handle, body_idx, SEG_PARTS_START + part_idx)
    return part_seg_ids


def to_env_state(obs):
    """Convert a recorded observation into the dict reset_env_to expects.

    Pickles store gripper_width as a 1-element array; reset_env_to needs a
    scalar to build the finger DOF targets.
    """
    robot_state = obs["robot_state"]
    return {
        "robot_state": {
            "joint_positions": np.asarray(robot_state["joint_positions"]).reshape(-1),
            "gripper_width": float(np.asarray(robot_state["gripper_width"]).reshape(-1)[0]),
        },
        "parts_poses": np.asarray(obs["parts_poses"]).reshape(-1),
    }


def set_state(env, state):
    """Set env 0 to a recorded observation.

    refresh() (one simulate + fetch_results + step_graphics) is required for
    the state tensors written by reset_env_to to propagate to the renderer.
    Velocities are zeroed by the reset, so the single physics step does not
    visibly move the pose away from the recorded one.
    """
    env.reset_env_to(0, state)
    env.refresh()


def render_state(env):
    """Read the overlay camera images for env 0 (rendered by the last refresh())."""
    gym_ = env.isaac_gym
    h, w = env.overlay_cam_height, env.overlay_cam_width
    color = gym_.get_camera_image(env.sim, env.envs[0], env.overlay_cam, gymapi.IMAGE_COLOR)
    color = color.reshape(h, w, 4)[..., :3].copy()
    seg = gym_.get_camera_image(env.sim, env.envs[0], env.overlay_cam, gymapi.IMAGE_SEGMENTATION)
    seg = seg.reshape(h, w).copy()
    return color, seg


def pick_camera_interactively(env, states, timesteps):
    """Run the viewer loop until the user frames the shot and presses C.

    Returns the viewer camera transform (env-0 frame). N cycles through the
    selected poses so the whole motion can be checked against the framing.
    """
    gym_ = env.isaac_gym
    viewer = env.viewer
    e0 = env.envs[0]

    gym_.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_C, "capture")
    gym_.subscribe_viewer_keyboard_event(viewer, gymapi.KEY_N, "next_pose")

    pose_i = len(states) - 1
    set_state(env, states[pose_i])

    print(
        "\n=== Interactive camera framing ===\n"
        "  Pan/orbit with the mouse (viewer window must have focus).\n"
        "  [N]  cycle through the selected timesteps to check framing\n"
        f"       (showing timestep {timesteps[pose_i]})\n"
        "  [C]  capture this view and render the overlay\n"
        "  Close the window to abort.\n"
    )

    while True:
        if gym_.query_viewer_has_closed(viewer):
            print("Viewer closed — aborting without rendering.")
            sys.exit(1)

        for evt in gym_.query_viewer_action_events(viewer):
            if evt.value <= 0:
                continue
            if evt.action == "capture":
                print("View captured.")
                return gym_.get_viewer_camera_transform(viewer, e0)
            if evt.action == "next_pose":
                pose_i = (pose_i + 1) % len(states)
                set_state(env, states[pose_i])
                print(f"Showing timestep {timesteps[pose_i]}")

        gym_.step_graphics(env.sim)
        gym_.draw_viewer(viewer, env.sim, True)
        gym_.sync_frame_time(env.sim)


def viewer_to_lookat(tf):
    """Convert a viewer camera transform to a (pos, target) look-at pair.

    The viewer camera looks along its local +z axis (verified empirically).
    Using set_camera_location instead of set_camera_transform sidesteps the
    sensor camera's quaternion axis convention entirely; the sim up-axis
    becomes image-up, which matches the roll-free viewer camera.
    """
    forward = tf.r.rotate(gymapi.Vec3(0.0, 0.0, 1.0))
    pos = [tf.p.x, tf.p.y, tf.p.z]
    target = [tf.p.x + forward.x, tf.p.y + forward.y, tf.p.z + forward.z]
    return pos, target


def apply_camera_json(env, cam_json):
    """Point the overlay camera using a saved/hand-written camera JSON."""
    if "target" not in cam_json or "pos" not in cam_json:
        raise ValueError("Camera JSON must contain 'pos' and 'target'.")
    env.isaac_gym.set_camera_location(
        env.overlay_cam, env.envs[0], gymapi.Vec3(*cam_json["pos"]), gymapi.Vec3(*cam_json["target"])
    )


def composite(colors, masks, alphas):
    """Blend older poses (per-frame boolean masks) onto the final pose's render.

    Ghosts are drawn oldest-first so newer poses occlude older ones; the final
    pose's masked pixels are re-pasted fully opaque on top so ghosts never
    bleed over it.
    """
    canvas = colors[-1].astype(np.float32)
    for color, mask, alpha in zip(colors[:-1], masks[:-1], alphas):
        mask = mask[..., None]
        canvas = np.where(mask, alpha * color + (1 - alpha) * canvas, canvas)
    final_mask = masks[-1][..., None]
    canvas = np.where(final_mask, colors[-1].astype(np.float32), canvas)
    return canvas.round().astype(np.uint8)


def infer_furniture(pickle_path: Path):
    for part in pickle_path.parts:
        if part in FURNITURE_NAMES:
            return part
    return None


def pickle_stem(pickle_path: Path):
    name = pickle_path.name
    for suffix in (".pkl.xz", ".pkl.gz", ".pkl"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return pickle_path.stem


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pickle", type=Path, help="Raw trajectory pickle (.pkl, .pkl.xz, .pkl.gz)")
    parser.add_argument(
        "--timesteps",
        type=int,
        nargs="+",
        required=True,
        help="Timesteps to overlay (negative indices count from the end, e.g. -1 is the last)",
    )
    parser.add_argument("--furniture", type=str, default=None, help="Furniture task (inferred from path if omitted)")
    parser.add_argument("--out", type=Path, default=None, help="Output PNG path (default: <pickle-stem>_overlay.png)")
    parser.add_argument(
        "--camera-pose",
        type=Path,
        default=None,
        help="Camera JSON from a previous run; if given, renders headless without the GUI",
    )
    parser.add_argument("--width", type=int, default=1920, help="Render width")
    parser.add_argument("--height", type=int, default=1080, help="Render height")
    parser.add_argument("--fov", type=float, default=90.0, help="Horizontal FOV; 90 matches the GUI viewer framing")
    parser.add_argument("--alpha-min", type=float, default=0.25, help="Opacity of the oldest ghost pose")
    parser.add_argument(
        "--alpha-max", type=float, default=0.7, help="Opacity of the newest ghost pose (final pose is always opaque)"
    )
    parser.add_argument(
        "--exclude-parts",
        type=str,
        nargs="*",
        default=None,
        help="Part names to exclude from the ghost/overlay masks. Default: parts ending in '_top' "
        "(e.g. square_table_top), so the final tabletop does not cover older leg poses. "
        "Pass with no values to include every part.",
    )
    parser.add_argument("--save-frames", action="store_true", help="Also save per-timestep color/mask PNGs")
    parser.add_argument("--gpu", type=int, default=0, help="Compute/graphics GPU device id")
    args = parser.parse_args()

    data = unpickle_data(args.pickle)
    observations = data["observations"]
    n_obs = len(observations)

    timesteps = [t if t >= 0 else n_obs + t for t in args.timesteps]
    for t in timesteps:
        if not 0 <= t < n_obs:
            raise ValueError(f"Timestep {t} out of range for trajectory of length {n_obs}")
    timesteps = sorted(set(timesteps))
    states = [to_env_state(observations[t]) for t in timesteps]
    print(f"Loaded {args.pickle} ({n_obs} timesteps); rendering timesteps {timesteps}")

    furniture = args.furniture or infer_furniture(args.pickle)
    if furniture is None:
        raise ValueError("Could not infer furniture task from the pickle path; pass --furniture")

    out_path = args.out or Path(f"{pickle_stem(args.pickle)}_overlay.png")
    camera_json_path = out_path.with_suffix(".camera.json")

    cam_json = None
    if args.camera_pose is not None:
        with open(args.camera_pose) as f:
            cam_json = json.load(f)

    OverlayRenderEnv.overlay_cam_width = args.width
    OverlayRenderEnv.overlay_cam_height = args.height
    OverlayRenderEnv.overlay_cam_fov = cam_json.get("fov", args.fov) if cam_json else args.fov

    env = OverlayRenderEnv(
        furniture=furniture,
        num_envs=1,
        headless=cam_json is not None,
        resize_img=False,
        randomness="low",
        no_noise=True,
        compute_device_id=args.gpu,
        graphics_device_id=args.gpu,
    )
    part_seg_ids = assign_segmentation_ids(env)

    if args.exclude_parts is None:
        exclude_parts = [name for name in part_seg_ids if name.endswith("_top")]
    else:
        unknown = [name for name in args.exclude_parts if name not in part_seg_ids]
        if unknown:
            raise ValueError(f"Unknown part name(s) {unknown}; available parts: {sorted(part_seg_ids)}")
        exclude_parts = args.exclude_parts
    excluded_seg_ids = np.array([part_seg_ids[name] for name in exclude_parts], dtype=np.int64)
    if exclude_parts:
        print(f"Excluding parts from overlay masks: {exclude_parts}")

    if cam_json is not None:
        apply_camera_json(env, cam_json)
    else:
        tf = pick_camera_interactively(env, states, timesteps)
        pos, target = viewer_to_lookat(tf)
        cam_json = {"pos": pos, "target": target, "fov": args.fov}
        apply_camera_json(env, cam_json)
        with open(camera_json_path, "w") as f:
            json.dump(cam_json, f, indent=2)
        print(f"Camera pose saved to {camera_json_path} (reuse with --camera-pose {camera_json_path})")

    colors, masks = [], []
    for t, state in zip(timesteps, states):
        set_state(env, state)
        color, seg = render_state(env)
        mask = (seg > 0) & ~np.isin(seg, excluded_seg_ids)
        colors.append(color)
        masks.append(mask)
        print(f"Rendered timestep {t} (mask covers {mask.mean():.1%} of the image)")

    if args.save_frames:
        frames_dir = out_path.parent / f"{out_path.stem}_frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        for t, color, mask in zip(timesteps, colors, masks):
            Image.fromarray(color).save(frames_dir / f"t{t:05d}_color.png")
            Image.fromarray((mask * 255).astype(np.uint8)).save(frames_dir / f"t{t:05d}_mask.png")
        print(f"Per-frame renders saved to {frames_dir}/")

    n_ghosts = len(states) - 1
    alphas = np.linspace(args.alpha_min, args.alpha_max, n_ghosts) if n_ghosts > 0 else []
    overlay = composite(colors, masks, alphas)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(out_path)
    print(f"Overlay saved to {out_path}")


if __name__ == "__main__":
    main()
