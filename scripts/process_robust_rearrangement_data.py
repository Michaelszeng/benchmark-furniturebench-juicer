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

from isaacgym import gymtorch  # must come before torch
import gym
import furniture_bench  # noqa: F401 - registers FurnitureSim envs with gym

import numpy as np
import zarr
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torch

from src.common.geometry import np_rot_6d_to_isaac_quat
from src.data_processing.utils import resize, resize_crop


# ---------------------------------------------------------------------------
# IK helpers
# ---------------------------------------------------------------------------


def ik_refresh(env_inner, pos_target: torch.Tensor):
    """Simulate one step + refresh kinematics/Jacobian, skip camera rendering.

    env_inner must be created with ctrl_mode="diffik" (DOF_MODE_POS) so the
    position targets hold the arm in place during simulate() instead of letting
    gravity pull it away.
    """
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
# Collision helpers
# ---------------------------------------------------------------------------


def disable_all_collisions(env_inner):
    """Set filter=1 on every shape in every env, suppressing all collisions.

    IsaacGym skips collision between two shapes when (filterA & filterB) != 0.
    Called once at startup so that render_frame's simulate() step is pure FK
    with no physics interactions that would displace parts or fingers.
    """
    for i in range(env_inner.num_envs):
        env_ptr = env_inner.envs[i]
        n = env_inner.isaac_gym.get_actor_count(env_ptr)
        for j in range(n):
            h = env_inner.isaac_gym.get_actor_handle(env_ptr, j)
            props = env_inner.isaac_gym.get_actor_rigid_shape_properties(env_ptr, h)
            if props:
                for p in props:
                    p.filter = 1
                env_inner.isaac_gym.set_actor_rigid_shape_properties(env_ptr, h, props)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def render_frame(env, env_inner, joint_9d, parts_poses_42d, pos_target: torch.Tensor):
    """Set sim state and render. Image tensor access must NOT be open on entry.

    Collisions must already be disabled globally (disable_all_collisions called
    once at startup) so that simulate() here is pure FK with no physics
    interactions displacing parts or fingers.

    Args:
        env: Wrapped gymnasium env (used for get_observation).
        env_inner: env.unwrapped.
        joint_9d: (9,) ndarray arm (7) + gripper (2) joint positions.
            joint_9d[7:9] are the symmetric finger split (gripper_width / 2 each).
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

    # Set position targets = desired DOF pos so the simulate() step holds the arm.
    pos_target[0, :7] = torch.tensor(dof_pos[:7], dtype=torch.float32, device=env_inner.device)
    pos_target[0, 7:9] = torch.tensor(dof_pos[7:9], dtype=torch.float32, device=env_inner.device)
    env_inner.isaac_gym.set_dof_position_target_tensor(env_inner.sim, gymtorch.unwrap_tensor(pos_target))

    env_inner.isaac_gym.simulate(env_inner.sim)
    env_inner.isaac_gym.fetch_results(env_inner.sim, True)

    # Snap both parts and franka back so step_graphics renders exact positions.
    # Parts drift under gravity during simulate() (no table collision since
    # collisions are globally disabled).  Finger DOFs use DOF_MODE_EFFORT so
    # they drift even without contact forces; snapping ensures the rendered
    # positions exactly match the intended joint_9d values.
    env_inner._reset_parts(0, parts_poses_42d)
    env_inner._reset_franka(0, dof_pos)
    env_inner.isaac_gym.step_graphics(env_inner.sim)
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

    img1 = resize(img1)  # -> (240, 320, 3) numpy
    img2 = resize_crop(img2)  # -> (240, 320, 3) numpy
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
            # zarr.copy preserves compressors, filters, and object codecs
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
    # Overwrite if exists
    dst = zarr.open(str(output_zarr_path), mode="w")

    copy_zarr_skeleton(src, dst)

    chunk_t = min(256, T)
    dst_img1 = dst.zeros("color_image1", shape=(T, H, W, C), chunks=(chunk_t, H, W, C), dtype=np.uint8)
    dst_img2 = dst.zeros("color_image2", shape=(T, H, W, C), chunks=(chunk_t, H, W, C), dtype=np.uint8)

    env_inner = env.unwrapped

    episode_ends = src["episode_ends"][:]
    episode_starts = np.concatenate([[0], episode_ends[:-1]])

    robot_state_all = src["robot_state"][:]  # (T, 16) float32
    parts_poses_all = src["parts_poses"][:]  # (T, 42) float32

    # Shared position-target tensor reused across frames.
    pos_target = torch.zeros_like(env_inner.dof_pos)

    image_access_open = False  # tracks whether start_access_image_tensors is active

    for ep_idx, (ep_start, ep_end) in enumerate(zip(episode_starts, episode_ends)):
        print(f"  Episode {ep_idx + 1}/{len(episode_ends)}: frames {ep_start}-{ep_end}")
        current_q = None  # reset warm-start at episode boundary

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
            # so ee_pos was recorded as hand_pos - base_pos with base at Z=0.415.
            # Juicer's fork uses ROBOT_HEIGHT=0.015, placing the base at Z=0.430.
            # Subtract the difference so the IK targets the correct world height.
            ee_pos[2] -= 0.015
            ee_quat_xyzw = np_rot_6d_to_isaac_quat(rs[3:9][None])[0]  # (4,)
            gripper_width = float(rs[15])

            # End image tensor access BEFORE running IK: simulate() must not
            # be called while image tensors are mapped (causes black/duplicate frames).
            if image_access_open:
                env_inner.isaac_gym.end_access_image_tensors(env_inner.sim)
                image_access_open = False

            # Solve IK (warm-started from previous frame's arm joints).
            joint_9d, pos_err, rot_err = solve_ik(
                env_inner,
                target_ee_pos_np=ee_pos,
                target_ee_quat_xyzw_np=ee_quat_xyzw,
                init_q=current_q,
            )
            if pos_err > 1e-4 or rot_err > 0.1 * np.pi / 180:
                print(f"    [IK] frame {frame_idx}: pos_err={pos_err * 1000:.3f}mm  rot_err={np.degrees(rot_err):.4f}deg")
            current_q = joint_9d[:7]

            # Individual finger positions are not stored in the zarr (only the
            # total gripper_width = dof[7] + dof[8] is saved).  Use a symmetric
            # split; any asymmetry from the original teleop is lost information.
            joint_9d[7] = gripper_width / 2.0
            joint_9d[8] = gripper_width / 2.0

            # Render (image tensor access is closed here, safe to call refresh).
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
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if args.output_dir is None:
        stem = input_dir.name
        output_dir = input_dir.with_name(stem + "_apriltags")
    else:
        output_dir = args.output_dir.resolve()

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")

    # Do NOT redirect ASSET_ROOT -> default furniture-bench assets -> AprilTags present.
    # ctrl_mode="diffik": arm DOFs use DOF_MODE_POS (stiffness=1000) so position
    # targets hold the arm in place during IK's simulate() calls.
    print(f"Creating env (furniture={args.furniture}, gpu={args.gpu_id}, ctrl_mode=diffik)")
    env = gym.make(
        "FurnitureSimFull-v0",
        furniture=args.furniture,
        headless=True,
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

    # Disable all collisions globally.  render_frame's simulate() is used only
    # for FK propagation; disabling collisions ensures no physics interactions
    # displace parts or fingers, so step_graphics renders the exact intended state.
    print("Disabling all collisions...")
    disable_all_collisions(env_inner)

    process_zarr(env, input_dir, output_dir)
    print("Done.")


if __name__ == "__main__":
    main()
