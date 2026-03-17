"""
Re-render robust-rearrangement zarr datasets with AprilTags visible.

The source zarr was collected without AprilTags. This script replays each
trajectory frame-by-frame by:
  1. Solving IK (iterative damped Jacobian pseudoinverse) to find joint
     positions matching the stored EE pose, warm-started from the previous
     frame's result.
  2. Resetting the simulator to those joint positions + stored parts poses.
  3. Rendering with AprilTags enabled (default furniture-bench asset root).

The output zarr has the same structure as the input, with color_image1 and
color_image2 replaced by the re-rendered images.

Usage:
    python scripts/process_robust_rearrangement_data.py \
        --input-dir dataset/processed/diffik/sim/one_leg/teleop/low/success.zarr \
        --output-dir dataset/processed/diffik/sim/one_leg/teleop/low/success_processed.zarr \
        --furniture one_leg \
        --randomness low
"""

# isort: skip_file
import argparse
from pathlib import Path

from isaacgym import gymtorch, gymapi  # must come before torch
import gym
import furniture_bench  # noqa: F401 - registers FurnitureSim envs with gym
from furniture_bench.envs.furniture_sim_env import FurnitureSimEnv, ASSET_ROOT
from furniture_bench.utils.pose import get_mat

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torch

from furniture_bench.sim_config import sim_config
from src.common.geometry import np_rot_6d_to_isaac_quat
from src.data_processing.utils import resize, resize_crop


# ---------------------------------------------------------------------------
# IK helpers
# ---------------------------------------------------------------------------


def ik_refresh(env_inner, pos_target: torch.Tensor):
    """Simulate one step + refresh kinematics/Jacobian, skip camera rendering."""
    env_inner.isaac_gym.set_dof_position_target_tensor(env_inner.sim, gymtorch.unwrap_tensor(pos_target))
    env_inner.isaac_gym.simulate(env_inner.sim)
    env_inner.isaac_gym.fetch_results(env_inner.sim, True)
    env_inner.isaac_gym.refresh_dof_state_tensor(env_inner.sim)
    env_inner.isaac_gym.refresh_rigid_body_state_tensor(env_inner.sim)
    env_inner.isaac_gym.refresh_jacobian_tensors(env_inner.sim)


def solve_ik(
    env_inner,
    target_ee_pos_np,
    target_ee_quat_xyzw_np,
    init_q=None,
    parts_poses_42d=None,
    max_iter=500,
    pos_thresh=1e-5,
    rot_thresh=0.05 * np.pi / 180,
    damping=0.05,
):
    """Return 9-DOF joint positions (arm 7 + gripper 2) via damped Jacobian IK.

    Args:
        env_inner: Unwrapped FurnitureSimEnv (must use ctrl_mode="diffik").
        target_ee_pos_np: (3,) ndarray, target EE position (hand_pos - base_pos).
        target_ee_quat_xyzw_np: (4,) ndarray, target EE quaternion (x,y,z,w).
        init_q: (7,) ndarray arm joint warm-start (defaults to default_dof_pos).
        parts_poses_42d: (42,) ndarray parts poses in AprilTag frame.  When
            provided, parts are reset to these poses before every IK simulate
            step so they never drift far under gravity.  Without this, a
            cold-start (500 iterations) lets parts fall ~300 m, which can
            confuse the renderer's frustum culling and produce black frames.
        max_iter: Maximum IK iterations.
        pos_thresh: Position convergence threshold in metres.
        rot_thresh: Rotation convergence threshold in radians.
        damping: Damped-least-squares regulariser lambda.

    Returns:
        (9,) ndarray: arm joints (7) + gripper open position (2).
    """
    device = env_inner.device

    if init_q is None:
        init_q = env_inner.default_dof_pos[:7]

    current_q = torch.tensor(init_q, dtype=torch.float32, device=device)

    target_pos = torch.tensor(target_ee_pos_np, dtype=torch.float32, device=device)
    rot_target = R.from_quat(target_ee_quat_xyzw_np)  # scipy uses xyzw

    # Persistent position-target tensor (arm + gripper).
    pos_target = torch.zeros_like(env_inner.dof_pos)
    pos_target[0, 7:9] = torch.tensor(env_inner.default_dof_pos[7:9], dtype=torch.float32, device=device)

    eye6 = torch.eye(6, dtype=torch.float32, device=device)

    for _ in range(max_iter):
        # Reset parts and obstacles each iteration so they don't drift.
        # Obstacle-furniture collisions are enabled (filter 1 & 0 = 0), so
        # _reset_parts placing furniture against the obstacle generates contact
        # forces that cumulatively tilt the obstacles without this reset.
        if parts_poses_42d is not None:
            env_inner._reset_parts(0, parts_poses_42d)
            reset_obstacle_pose(env_inner, parts_poses_42d)
        # Set arm to current guess and tick physics (position-controlled, so
        # the arm stays put rather than falling under gravity).
        dof_np = np.concatenate([current_q.cpu().numpy(), env_inner.default_dof_pos[7:9]])
        env_inner._reset_franka(0, dof_np)
        pos_target[0, :7] = current_q
        ik_refresh(env_inner, pos_target)

        ee_pos, ee_quat = env_inner.get_ee_pose()  # (num_envs, 3/4)
        ee_pos = ee_pos[0]  # (3,)
        ee_quat_np = ee_quat[0].cpu().numpy()  # xyzw

        pos_error = target_pos - ee_pos  # (3,) in device

        rot_current = R.from_quat(ee_quat_np)
        rot_error_rv = (rot_target * rot_current.inv()).as_rotvec()
        rot_error = torch.tensor(rot_error_rv, dtype=torch.float32, device=device)

        pos_err_norm = pos_error.norm().item()
        rot_err_norm = rot_error.norm().item()
        if pos_err_norm < pos_thresh and rot_err_norm < rot_thresh:
            break

        # Damped least-squares: dq = J^T (J J^T + lambda*I)^{-1} ee_vel
        J = env_inner.jacobian_eef[0]  # (6, 7)
        ee_vel = torch.cat([pos_error, rot_error])  # (6,)
        JJT = J @ J.T  # (6, 6)
        dq = J.T @ torch.linalg.solve(JJT + damping * eye6, ee_vel)  # (7,)

        current_q = current_q + dq

    arm_q = current_q.cpu().numpy()
    return np.concatenate([arm_q, env_inner.default_dof_pos[7:9]]), pos_err_norm, rot_err_norm


# ---------------------------------------------------------------------------
# Obstacle repositioning
# ---------------------------------------------------------------------------


def reset_obstacle_pose(env_inner, parts_poses_42d: np.ndarray):
    """Reposition the U-shaped obstacle to match the pose stored in the zarr.

    The robust-rearrangement dataset was collected with FurnitureRLSimEnv, which
    randomises the obstacle position per episode (±0.02 m for 'low' randomness)
    and appends the obstacle_front april-frame pose as the 6th element of
    parts_poses (indices [35:42]).  This function reads that stored pose,
    converts it to sim coordinates, and repositions all three obstacle actors.

    Must be called after simulate() and before step_graphics() each frame.
    """
    # Build and cache indices/constants on first call (mirrors _reset_parts pattern).
    if not hasattr(env_inner, "_obstacle_actor_idxs"):
        env_inner._obstacle_actor_idxs = torch.tensor(
            [
                env_inner.isaac_gym.find_actor_index(env_inner.envs[0], name, gymapi.DOMAIN_SIM)
                for name in ("obstacle_front", "obstacle_right", "obstacle_left")
            ],
            device=env_inner.device,
            dtype=torch.int32,
        )
        half_sqrt2 = float(np.sqrt(2) / 2)
        # All three obstacles are created with gymapi.Quat.from_axis_angle(z, π/2).
        env_inner._obstacle_quat = torch.tensor(
            [0.0, 0.0, half_sqrt2, half_sqrt2], dtype=torch.float32, device=env_inner.device
        )
        # Positions of right/left relative to front in sim world frame.
        env_inner._obstacle_offsets = torch.tensor(
            [[0.0, 0.0, 0.0], [-0.075, -0.175, 0.0], [-0.075, 0.175, 0.0]],
            dtype=torch.float32,
            device=env_inner.device,
        )

    # parts_poses_42d[35:38] = obstacle_front position in AprilTag frame.
    sim_mat = env_inner.april_coord_to_sim_coord(get_mat(parts_poses_42d[35:38].tolist(), [0, 0, 0]))
    front_pos = torch.tensor(
        [sim_mat[0, 3], sim_mat[1, 3], sim_mat[2, 3]], dtype=torch.float32, device=env_inner.device
    )

    idxs = env_inner._obstacle_actor_idxs.long()
    env_inner.root_tensor[idxs, 0:3] = front_pos + env_inner._obstacle_offsets
    env_inner.root_tensor[idxs, 3:7] = env_inner._obstacle_quat
    env_inner.root_tensor[idxs, 7:13] = 0.0
    env_inner.isaac_gym.set_actor_root_state_tensor_indexed(
        env_inner.sim,
        gymtorch.unwrap_tensor(env_inner.root_tensor),
        gymtorch.unwrap_tensor(env_inner._obstacle_actor_idxs),
        3,
    )


# ---------------------------------------------------------------------------
# Finger split via physics simulation
# ---------------------------------------------------------------------------


def find_finger_split_sim(
    env_inner,
    gripper_width: float,
    parts_poses_42d: np.ndarray,
    arm_q: np.ndarray,
    pos_target: torch.Tensor,
    n_steps: int = 50,
):
    """Return (d1, d2) finger DOF positions by simulating gripper closing.

    Simulates the gripper closing toward gripper_width/2 while kinematically
    locking the arm and all parts in place.  Each finger is driven by a
    closing torque only while it is still above the target (gripper_width/2);
    once it reaches the target the torque is cut.  Contact forces from parts
    naturally stop a finger before the target, producing an asymmetric split.

    The loop exits early once both fingers are at or below gripper_width/2.
    This gives smooth, threshold-free behavior: a fully-open gripper command
    means the fingers are already at the target and converge in one step; a
    grasping command closes the fingers until they hit the part or the target.

    Args:
        env_inner: Unwrapped FurnitureSimEnv.
        gripper_width: Total commanded opening = d1 + d2 (metres).
        parts_poses_42d: (42,) float32 parts poses in AprilTag frame.
        arm_q: (7,) arm joint positions from IK solve.
        pos_target: (1, 9) position-target tensor (shared).
        n_steps: Maximum simulation steps for convergence.

    Returns:
        d1: float, left finger DOF position (metres).
        d2: float, right finger DOF position (metres).
    """
    part_actor_idxs = torch.tensor(env_inner.part_actor_idx_by_env[0], device=env_inner.device, dtype=torch.int32)

    # Arm: hold at IK solution via position target (DOF_MODE_POS).
    pos_target[0, :7] = torch.tensor(arm_q, dtype=torch.float32, device=env_inner.device)
    env_inner.isaac_gym.set_dof_position_target_tensor(env_inner.sim, gymtorch.unwrap_tensor(pos_target))

    # Fingers: DOF_MODE_EFFORT — must use actuation force, not position target.
    closing_torque = -float(sim_config["robot"]["gripper_torque"])
    torque_action = torch.zeros_like(env_inner.dof_pos)

    for _ in range(n_steps):
        finger_dofs = env_inner.dof_pos[0, 7:9].cpu().numpy()

        # Check if finger width has reduced more than the target width in the dataset. If so, stop simulating/closing
        current_gripper_width = finger_dofs[0] + finger_dofs[1]
        if current_gripper_width <= gripper_width:
            break

        # Otherwise, give the grippers the closing torque.
        torque_action[0, 7] = closing_torque
        torque_action[0, 8] = closing_torque

        # Lock parts and obstacles: zero velocities, then reset pos/quat.
        env_inner.root_tensor[part_actor_idxs.long(), 7:13] = 0.0
        env_inner._reset_parts(0, parts_poses_42d)
        reset_obstacle_pose(env_inner, parts_poses_42d)

        # Lock arm, carry current finger positions; zeros all DOF velocities.
        env_inner._reset_franka(0, np.concatenate([arm_q, finger_dofs]))

        # Apply torque (must follow _reset_franka; actuation force is a
        # separate GPU buffer from the DOF state tensor).
        env_inner.isaac_gym.set_dof_actuation_force_tensor(env_inner.sim, gymtorch.unwrap_tensor(torque_action))

        env_inner.isaac_gym.simulate(env_inner.sim)
        env_inner.isaac_gym.fetch_results(env_inner.sim, True)
        env_inner.isaac_gym.refresh_dof_state_tensor(env_inner.sim)

        if env_inner.viewer is not None:
            snap_fingers = env_inner.dof_pos[0, 7:9].cpu().numpy()
            env_inner.root_tensor[part_actor_idxs.long(), 7:13] = 0.0
            env_inner._reset_parts(0, parts_poses_42d)
            env_inner._reset_franka(0, np.concatenate([arm_q, snap_fingers]))
            reset_obstacle_pose(env_inner, parts_poses_42d)
            env_inner.isaac_gym.step_graphics(env_inner.sim)
            env_inner.isaac_gym.draw_viewer(env_inner.viewer, env_inner.sim, False)
            env_inner.isaac_gym.sync_frame_time(env_inner.sim)

    d1 = float(env_inner.dof_pos[0, 7].item())
    d2 = float(env_inner.dof_pos[0, 8].item())
    return d1, d2


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_frame(env, env_inner, joint_9d, parts_poses_42d, pos_target: torch.Tensor):
    """Set sim state and render. Image tensor access must NOT be open on entry.

    Args:
        env: Wrapped gymnasium env (used for get_observation).
        env_inner: env.unwrapped.
        joint_9d: (9,) ndarray arm (7) + gripper (2) joint positions.
            joint_9d[7:9] are the finger DOF positions from find_finger_split_sim.
        parts_poses_42d: (42,) ndarray flat parts poses in AprilTag coords.
        pos_target: (1, 9) position-target tensor; holds the arm during simulate().

    Returns:
        img1: (240, 320, 3) uint8 numpy (wrist camera, resized).
        img2: (240, 320, 3) uint8 numpy (front camera, resized+cropped).
    """
    dof_pos = joint_9d.copy()

    env_inner.furnitures[0].reset()
    env_inner._reset_franka(0, dof_pos)
    env_inner._reset_parts(0, parts_poses_42d)
    env_inner.env_steps[0] = 0

    # Set position targets so the simulate() step holds the arm in place.
    pos_target[0, :7] = torch.tensor(dof_pos[:7], dtype=torch.float32, device=env_inner.device)
    pos_target[0, 7:9] = torch.tensor(dof_pos[7:9], dtype=torch.float32, device=env_inner.device)
    env_inner.isaac_gym.set_dof_position_target_tensor(env_inner.sim, gymtorch.unwrap_tensor(pos_target))

    env_inner.isaac_gym.simulate(env_inner.sim)
    env_inner.isaac_gym.fetch_results(env_inner.sim, True)

    # Snap parts, arm, and obstacles back to exact target positions so that
    # step_graphics renders them at the right place (gravity / contact forces
    # during simulate() can displace them slightly).
    env_inner._reset_parts(0, parts_poses_42d)
    env_inner._reset_franka(0, dof_pos)
    reset_obstacle_pose(env_inner, parts_poses_42d)
    env_inner.isaac_gym.step_graphics(env_inner.sim)
    if env_inner.viewer is not None:
        env_inner.isaac_gym.draw_viewer(env_inner.viewer, env_inner.sim, False)
        env_inner.isaac_gym.sync_frame_time(env_inner.sim)
    env_inner.isaac_gym.refresh_dof_state_tensor(env_inner.sim)
    env_inner.isaac_gym.refresh_dof_force_tensor(env_inner.sim)
    env_inner.isaac_gym.refresh_rigid_body_state_tensor(env_inner.sim)
    env_inner.isaac_gym.refresh_jacobian_tensors(env_inner.sim)
    env_inner.isaac_gym.refresh_mass_matrix_tensors(env_inner.sim)
    env_inner.isaac_gym.render_all_camera_sensors(env_inner.sim)
    env_inner.isaac_gym.start_access_image_tensors(env_inner.sim)

    new_obs = env.get_observation()
    img1 = new_obs["color_image1"].squeeze(0).cpu().numpy()
    img2 = new_obs["color_image2"].squeeze(0).cpu().numpy()

    img1 = resize(img1)
    img2 = resize_crop(img2)
    return img1, img2


# ---------------------------------------------------------------------------
# zarr I/O
# ---------------------------------------------------------------------------


def copy_zarr_skeleton(src: zarr.Group, dst: zarr.Group, skip=("color_image1", "color_image2")):
    """Copy all arrays/groups from src to dst, skipping those we'll rewrite."""
    for key in src:
        if key in skip:
            continue
        item = src[key]
        if isinstance(item, zarr.Array):
            zarr.copy(item, dst, name=key)
        elif isinstance(item, zarr.Group):
            copy_zarr_skeleton(item, dst.require_group(key), skip)


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------


def process_zarr(env, input_zarr_path: Path, output_zarr_path: Path):
    src = zarr.open(str(input_zarr_path), mode="r")
    T = src["robot_state"].shape[0]
    H, W, C = 240, 320, 3

    output_zarr_path.mkdir(parents=True, exist_ok=True)
    dst = zarr.open(str(output_zarr_path), mode="w")

    copy_zarr_skeleton(src, dst)

    chunk_t = min(256, T)
    dst_img1 = dst.zeros("color_image1", shape=(T, H, W, C), chunks=(chunk_t, H, W, C), dtype=np.uint8)
    dst_img2 = dst.zeros("color_image2", shape=(T, H, W, C), chunks=(chunk_t, H, W, C), dtype=np.uint8)

    env_inner = env.unwrapped

    # gym.make() calls refresh() which calls start_access_image_tensors.
    # Close that mapping now so our simulate() calls during IK are safe.
    env_inner.isaac_gym.end_access_image_tensors(env_inner.sim)

    episode_ends = src["episode_ends"][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    robot_state_all = src["robot_state"][:]  # (T, 16) float32
    parts_poses_all = src["parts_poses"][:]  # (T, 42) float32

    # Shared position-target tensor reused across frames.
    pos_target = torch.zeros_like(env_inner.dof_pos)

    image_access_open = False  # tracks whether start_access_image_tensors is active

    for ep_idx, (ep_start, ep_end) in enumerate(zip(episode_starts, episode_ends)):
        print(f"  Episode {ep_idx + 1}/{len(episode_ends)}: frames {ep_start}-{ep_end}")
        current_q = None  # reset IK warm-start at episode boundary

        if image_access_open:
            env_inner.isaac_gym.end_access_image_tensors(env_inner.sim)
            image_access_open = False

        # Set obstacle to this episode's randomised position before any simulation.
        reset_obstacle_pose(env_inner, parts_poses_all[ep_start])

        for frame_idx in tqdm(range(ep_start, ep_end), leave=False):
            rs = robot_state_all[frame_idx]  # (16,)
            parts_poses = parts_poses_all[frame_idx]  # (42,)

            # Decode EE pose from 16D robot_state:
            #   [0:3]   ee_pos  (hand_pos - base_pos, IsaacGym world frame)
            #   [3:9]   ee_rot_6d -> ee_quat xyzw
            #   [9:12]  ee_pos_vel  (unused)
            #   [12:15] ee_ori_vel  (unused)
            #   [15]    gripper_width
            ee_pos = rs[0:3].copy()
            # The RR fork of furniture-bench used ROBOT_HEIGHT=0 (no bench clamp),
            # so ee_pos was recorded with base at Z=0.415.  Juicer's fork uses
            # ROBOT_HEIGHT=0.015, placing the base at Z=0.430.  Correct for this.
            ee_pos[2] -= 0.015
            ee_quat_xyzw = np_rot_6d_to_isaac_quat(rs[3:9][None])[0]  # (4,)
            gripper_width = float(rs[15])

            # End image tensor access BEFORE running IK: simulate() must not
            # be called while image tensors are mapped.
            if image_access_open:
                env_inner.isaac_gym.end_access_image_tensors(env_inner.sim)
                image_access_open = False

            joint_9d, pos_err, rot_err = solve_ik(
                env_inner,
                target_ee_pos_np=ee_pos,
                target_ee_quat_xyzw_np=ee_quat_xyzw,
                init_q=current_q,
                parts_poses_42d=parts_poses,
            )
            if pos_err > 1e-4 or rot_err > 0.1 * np.pi / 180:
                print(
                    f"    [IK] frame {frame_idx}: pos_err={pos_err * 1000:.3f}mm  rot_err={np.degrees(rot_err):.4f}deg"
                )
            current_q = joint_9d[:7]

            # Individual finger positions are not stored; only total gripper_width
            # = dof[7] + dof[8] is saved.  Simulate gripper closing so contact
            # forces naturally produce an asymmetric split.
            joint_9d[7], joint_9d[8] = find_finger_split_sim(
                env_inner,
                gripper_width=gripper_width,
                parts_poses_42d=parts_poses,
                arm_q=joint_9d[:7],
                pos_target=pos_target,
            )

            img1, img2 = render_frame(
                env,
                env_inner,
                joint_9d=joint_9d,
                parts_poses_42d=parts_poses,
                pos_target=pos_target,
            )
            image_access_open = True  # render_frame called start_access_image_tensors

            dst_img1[frame_idx] = img1
            dst_img2[frame_idx] = img2

    if image_access_open:
        env_inner.isaac_gym.end_access_image_tensors(env_inner.sim)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Re-render robust-rearrangement zarr with AprilTags")
    parser.add_argument("--input-dir", type=Path, required=True, help="Path to input .zarr directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Path for output .zarr directory (default: input dir with _apriltags suffix)",
    )
    parser.add_argument("--furniture", type=str, default="one_leg", help="Furniture task name (default: one_leg)")
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU device ID (default: 0)")
    parser.add_argument(
        "--randomness", type=str, default="low", choices=["low", "med", "high"], help="Randomness level (default: low)"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Open a viewer window so you can watch the simulation in real time.",
    )
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if args.output_dir is None:
        stem = input_dir.name
        output_dir = input_dir.with_name(stem + "_apriltags")
    else:
        output_dir = args.output_dir.resolve()

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Obstacles must be dynamic (not fix_base_link) so that
    # set_actor_root_state_tensor_indexed can reposition them for rendering.
    # disable_gravity keeps them stationary without physics support.
    def _import_obstacle_front_dynamic(self):
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        return self.isaac_gym.load_asset(self.sim, ASSET_ROOT, "furniture/urdf/obstacle_front.urdf", asset_options)

    def _import_obstacle_side_dynamic(self):
        asset_options = gymapi.AssetOptions()
        asset_options.disable_gravity = True
        return self.isaac_gym.load_asset(self.sim, ASSET_ROOT, "furniture/urdf/obstacle_side.urdf", asset_options)

    FurnitureSimEnv._import_obstacle_front_asset = _import_obstacle_front_dynamic
    FurnitureSimEnv._import_obstacle_side_asset = _import_obstacle_side_dynamic

    print(f"Creating env (furniture={args.furniture}, gpu={args.gpu_id}, ctrl_mode=diffik, headless={args.headless})")
    env = gym.make(
        "FurnitureSimFull-v0",
        furniture=args.furniture,
        headless=args.headless,
        num_envs=1,
        resize_img=False,
        np_step_out=False,
        channel_first=False,
        randomness=args.randomness,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        ctrl_mode="diffik",
    )

    env_inner = env.unwrapped

    # Disable inter-obstacle collisions: the three pieces' geometry overlaps at
    # the joints, causing contact impulses that spin the side pieces.
    # filter=1 on all obstacles means (1 & 1) != 0 → collision suppressed.
    # Obstacle-furniture pairs remain (1 & 0) == 0 → still collide.
    for obs_name in ("obstacle_front", "obstacle_right", "obstacle_left"):
        h = env_inner.isaac_gym.find_actor_handle(env_inner.envs[0], obs_name)
        props = env_inner.isaac_gym.get_actor_rigid_shape_properties(env_inner.envs[0], h)
        for p in props:
            p.filter = 1
        env_inner.isaac_gym.set_actor_rigid_shape_properties(env_inner.envs[0], h, props)

    process_zarr(env, input_dir, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
