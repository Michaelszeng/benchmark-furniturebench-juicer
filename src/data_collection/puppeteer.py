"""
Script to puppeteer all the parts in the environment.
No gravity, no collisions, just freedom to move the parts around.
"""

import argparse
import time

import furniture_bench.utils.transform as T
import gym
import numpy as np
import torch
from furniture_bench.config import config
from furniture_bench.envs.initialization_mode import Randomness
from isaacgym import gymapi, gymtorch

from src.data_collection.collect_enum import CollectEnum
from src.data_collection.keyboard_interface import KeyboardInterface


def main():
    parser = argparse.ArgumentParser(description="Debug furniture part poses")
    parser.add_argument(
        "--furniture", "-f", help="Name of the furniture", choices=list(config["furniture"].keys()), required=True
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI")
    args = parser.parse_args()

    # Initialize environment
    print("Initializing environment...")
    env = gym.make(
        "FurnitureSimFull-v0",
        furniture=args.furniture,
        num_envs=1,
        headless=args.headless,
        randomness=Randomness.LOW,
        compute_device_id=0,
        graphics_device_id=0,
    )

    env.reset()

    # Disable gravity so parts float
    sim_params = env.isaac_gym.get_sim_params(env.sim)
    sim_params.gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    env.isaac_gym.set_sim_params(env.sim, sim_params)
    print("Gravity disabled.")

    # Disable collisions for ALL actors.
    # In IsaacGym, two shapes do NOT collide if (filterA & filterB) != 0.
    # Setting all shape filters to the same non-zero bitmask ensures every pair
    # shares a common bit, suppressing all collision responses.
    print("Disabling collisions for all actors...")
    for i in range(env.num_envs):
        env_ptr = env.envs[i]
        num_actors = env.isaac_gym.get_actor_count(env_ptr)
        for j in range(num_actors):
            actor_handle = env.isaac_gym.get_actor_handle(env_ptr, j)
            shape_props = env.isaac_gym.get_actor_rigid_shape_properties(env_ptr, actor_handle)
            if shape_props is not None:
                for shape_prop in shape_props:
                    shape_prop.filter = 1
                env.isaac_gym.set_actor_rigid_shape_properties(env_ptr, actor_handle, shape_props)

    # Get list of parts including the robot
    parts = ["robot"] + [p.name for p in env.furniture.parts]
    active_part_idx = 0

    # Create views for velocities to zero them out
    # root_tensor shape: (num_envs, num_actors, 13)
    # 0-3: pos, 3-7: quat, 7-10: lin_vel, 10-13: ang_vel
    root_lin_vel = env.root_tensor.view(env.num_envs, -1, 13)[..., 7:10]
    root_ang_vel = env.root_tensor.view(env.num_envs, -1, 13)[..., 10:13]

    print("\n" + "=" * 50)
    print("DEBUG POSES MODE")
    print(f"Loaded furniture: {args.furniture}")
    print("Controls:")
    print("  B (Undo): Cycle active part (Robot -> Part1 -> Part2 ...)")
    print("  W/S: Move X")
    print("  A/D: Move Y")
    print("  E/Q: Move Z")
    print("  I/K: Rotate Pitch (Y-axis)")
    print("  J/L: Rotate Roll (X-axis)")
    print("  U/O: Rotate Yaw (Z-axis)")
    print("  P (Pause): Print current poses")
    print("  R (Reset): Reset environment")
    print("  [/]: Adjust speed")
    print("  ESC: Exit")
    print("=" * 50 + "\n")

    keyboard = KeyboardInterface()

    print(f"Active Part: {parts[active_part_idx]}")

    # Disable gravity to keep parts floating when we move them
    # env.isaac_gym.get_sim_params(env.sim).gravity = gymapi.Vec3(0.0, 0.0, 0.0)
    # Note: Changing sim params after creation might not work for all backends/params,
    # but let's try setting gravity to zero if possible.
    # Actually, we can just reset the part's velocity to 0 every step.

    while True:
        # Get action from keyboard (deltas)
        # use_quat=False returns euler angles for rotation
        action, collect_enum = keyboard.get_action(use_quat=False)

        # Parse action
        # Scale actions down by 0.15 to make increments smaller
        dpos = action[:3] * 0.15
        drot_euler = action[3:6] * 0.15  # (roll, pitch, yaw)
        grasp = action[6]

        if collect_enum == CollectEnum.UNDO:  # 'b'
            active_part_idx = (active_part_idx + 1) % len(parts)
            print(f"Switched to part: {parts[active_part_idx]}")
            time.sleep(0.2)  # Debounce

        elif collect_enum == CollectEnum.RESET:  # 'r'
            env.reset()
            print("Environment reset.")

        elif collect_enum == CollectEnum.PAUSE:  # 'p'
            print(f"\n--- Poses for {args.furniture} ---")

            # Robot Pose
            ee_pos, ee_quat = env.get_ee_pose()
            ee_pos = ee_pos[0].cpu().numpy()
            ee_quat = ee_quat[0].cpu().numpy()  # (x, y, z, w)
            print(f"Robot EE: Pos={ee_pos}, Quat={ee_quat}")

            # Parts Poses
            for p_name in [p.name for p in env.furniture.parts]:
                p_idx = env.parts_handles[p_name]
                pos = env.root_pos[0, p_idx].cpu().numpy()
                quat = env.root_quat[0, p_idx].cpu().numpy()
                print(f"Part '{p_name}': Pos={pos}, Quat={quat}")

            print("\n--- Relative Poses (Target relative to Source) ---")
            for i, p1 in enumerate(env.furniture.parts):
                for j, p2 in enumerate(env.furniture.parts):
                    if i == j:
                        continue

                    idx1 = env.parts_handles[p1.name]
                    idx2 = env.parts_handles[p2.name]

                    pos1 = env.root_pos[0, idx1].cpu().numpy()
                    quat1 = env.root_quat[0, idx1].cpu().numpy()
                    mat1 = T.to_homogeneous(pos1, T.quat2mat(quat1))

                    pos2 = env.root_pos[0, idx2].cpu().numpy()
                    quat2 = env.root_quat[0, idx2].cpu().numpy()
                    mat2 = T.to_homogeneous(pos2, T.quat2mat(quat2))

                    # Relative pose of p2 in p1's frame: T_1_2 = inv(T_w_1) @ T_w_2
                    rel_mat = np.linalg.inv(mat1) @ mat2
                    rel_pos = rel_mat[:3, 3]
                    rel_quat = T.mat2quat(rel_mat[:3, :3])

                    print(f"'{p2.name}' relative to '{p1.name}':\n  Pos={rel_pos}\n  Quat={rel_quat}")
            print("--------------------------\n")
            time.sleep(0.5)  # Debounce

        # Apply movement
        active_part_name = parts[active_part_idx]

        # Prepare robot action (default no-op)
        # Action: [x, y, z, qx, qy, qz, qw, gripper]
        # Delta mode: 0,0,0, 0,0,0,1, -1
        robot_action = torch.tensor([0, 0, 0, 0, 0, 0, 1, -1], device=env.device, dtype=torch.float32).unsqueeze(0)

        if np.linalg.norm(dpos) > 0 or np.linalg.norm(drot_euler) > 0:
            if active_part_name == "robot":
                # Robot control
                # Convert euler delta to quat delta
                dquat = T.mat2quat(T.euler2mat(drot_euler))
                # Action: [x, y, z, qx, qy, qz, qw, gripper]
                env_action = np.concatenate([dpos, dquat, [grasp]])
                # Expand to (num_envs, 8)
                robot_action = torch.tensor(env_action, device=env.device, dtype=torch.float32).unsqueeze(0)

            else:
                # Part control
                p_idx = env.parts_handles[active_part_name]

                # Update position
                curr_pos = env.root_pos[0, p_idx].cpu().numpy()
                new_pos = curr_pos + dpos
                env.root_pos[0, p_idx] = torch.tensor(new_pos, device=env.device)

                # Update rotation
                curr_quat = env.root_quat[0, p_idx].cpu().numpy()  # (x, y, z, w)
                curr_mat = T.quat2mat(curr_quat)
                delta_mat = T.euler2mat(drot_euler)

                # Apply delta in global frame (since keyboard controls are global axes)
                new_mat = delta_mat @ curr_mat

                new_quat = T.mat2quat(new_mat)
                env.root_quat[0, p_idx] = torch.tensor(new_quat, device=env.device)

                # Zero out velocities to prevent drift
                root_lin_vel[0, p_idx] = torch.zeros(3, device=env.device)
                root_ang_vel[0, p_idx] = torch.zeros(3, device=env.device)

                # Find global actor index
                part_list_idx = -1
                for i, p in enumerate(env.furniture.parts):
                    if p.name == active_part_name:
                        part_list_idx = i
                        break

                if part_list_idx != -1:
                    global_actor_idx = env.part_actor_idx_by_env[0][part_list_idx]

                    env.isaac_gym.set_actor_root_state_tensor_indexed(
                        env.sim,
                        gymtorch.unwrap_tensor(env.root_tensor),
                        gymtorch.unwrap_tensor(torch.tensor([global_actor_idx], device=env.device, dtype=torch.int32)),
                        1,
                    )

        # Always step environment to update physics and viewer
        env.step(robot_action)


if __name__ == "__main__":
    main()
