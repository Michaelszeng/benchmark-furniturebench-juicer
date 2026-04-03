"""
Convert raw trajectory pickle files into a processed zarr dataset.

Reads .pkl / .pkl.gz / .pkl.xz files from:
    $DATA_DIR_RAW/raw/{env}/{task}/{source}/{randomness}/

Applies the following transforms to each trajectory:
  - Converts quaternion to 6D rotation representation
  - Converts delta actions from quaternion to 6D rotation
  - Clips z-axis rotation deltas to ±0.35 and xyz position deltas to ±0.025
  - Computes absolute end-effector position actions from robot state + delta

Writes a zarr store to:
    $DATA_DIR_PROCESSED/processed/{env}/{task}/{source}/{randomness}/{outcome}.zarr

Usage:
    python -m src.data_processing.process_pickles \\
        --env sim --furniture one_leg --source scripted \\
        --randomness low --demo-outcome success [--overwrite] [--n-cpus 8]
"""

import argparse
import array
import os

if "DATA_DIR_RAW" not in os.environ:
    os.environ["DATA_DIR_RAW"] = "dataset"

if "DATA_DIR_PROCESSED" not in os.environ:
    os.environ["DATA_DIR_PROCESSED"] = "dataset"

import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import torch
import zarr
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from ipdb import set_trace as bp  # noqa
from numcodecs import JSON, Blosc
from tqdm import tqdm, trange

from src.common.files import get_processed_path, get_raw_paths
from src.common.geometry import (
    np_action_6d_to_quat,
    np_action_quat_to_6d_rotation,
    np_apply_quat,
    np_extract_ee_pose_6d,
    np_proprioceptive_quat_to_6d_rotation,
)
from src.common.types import Trajectory
from src.data_processing.utils import clip_axis_rotation
from src.visualization.render_mp4 import unpickle_data


# === Modified Function to Initialize Zarr Store with Empty Dimensions ===
def initialize_zarr_store_empty(out_path, data_shapes_and_dtypes, chunksize=32):
    """
    Initialize the Zarr store with empty dimensions for each dataset.
    """
    z = zarr.open(str(out_path), mode="w")
    z.attrs["time_created"] = datetime.now().astimezone().isoformat()

    # Define the compressor
    compressor = Blosc(cname="lz4", clevel=5)

    # Initialize datasets with empty first dimension
    for name, shape, dtype in data_shapes_and_dtypes:
        if "color_image" in name:  # Apply compression to image data
            z.create_dataset(
                name,
                shape=(0,) + shape,
                dtype=dtype,
                chunks=(chunksize,) + shape,
                compressor=compressor,
            )
        elif dtype == object:
            z.create_dataset(
                name,
                shape=(0,),
                dtype=dtype,
                chunks=(chunksize,),
                object_codec=JSON(),
            )
        else:
            z.create_dataset(name, shape=(0,) + shape, dtype=dtype, chunks=(chunksize,) + shape)

    return z


def process_pickle_file(
    pickle_path: Path,
    noop_threshold: float,
    calculate_pos_action_from_delta: bool = False,
):
    """
    Process a single pickle file and return processed data.
    """
    data: Trajectory = unpickle_data(pickle_path)
    obs = data["observations"]

    # Extract the observations from the pickle file and convert to 6D rotation
    color_image1 = np.array([o["color_image1"] for o in obs], dtype=np.uint8)[:-1]
    color_image2 = np.array([o["color_image2"] for o in obs], dtype=np.uint8)[:-1]

    if isinstance(obs[0]["robot_state"], dict):
        # Convert the robot state to a numpy array
        all_robot_state_quat = np.array(
            [filter_and_concat_robot_state(o["robot_state"]) for o in obs],
            dtype=np.float32,
        )
    else:
        all_robot_state_quat = np.array([o["robot_state"] for o in obs], dtype=np.float32)

    all_robot_state_6d = np_proprioceptive_quat_to_6d_rotation(all_robot_state_quat)

    robot_state_6d = all_robot_state_6d[:-1]
    parts_poses = np.array([o["parts_poses"] for o in obs], dtype=np.float32)[:-1]

    # Extract the delta actions from the pickle file and convert to 6D rotation
    action_delta = np.array(data["actions"], dtype=np.float32)
    if action_delta.shape[-1] == 8:
        action_delta_quat = action_delta
        action_delta_6d = np_action_quat_to_6d_rotation(action_delta_quat)
    elif action_delta.shape[-1] == 10:
        raise Exception("Expecting 8D actions, not 10D actions.")
        action_delta_6d = action_delta
        action_delta_quat = np_action_6d_to_quat(action_delta_6d)
    else:
        raise ValueError(f"Unexpected action shape: {action_delta.shape}. Expected (N, 8) or (N, 10)")

    # TODO: Make sure this is rectified in the controller-end and
    # figure out what to do with the corrupted raw data
    # For now, clip the z-axis rotation to 0.35
    action_delta_6d = clip_axis_rotation(action_delta_6d, clip_mag=0.35, axis="z")

    # Clip xyz delta position actions to ±0.025
    action_delta_6d[:, :3] = np.clip(action_delta_6d[:, :3], -0.025, 0.025)

    # Calculate the position actions
    if calculate_pos_action_from_delta:
        action_pos = np.concatenate(
            [
                all_robot_state_quat[:-1, :3] + action_delta_quat[:, :3],
                np_apply_quat(all_robot_state_quat[:-1, 3:7], action_delta_quat[:, 3:7]),
                # Append the gripper action
                action_delta_quat[:, -1:],
            ],
            axis=1,
        )
        action_pos_6d = np_action_quat_to_6d_rotation(action_pos)

    else:
        # Extract the position control actions from the pickle file
        # and concat onto the position actions the gripper actions
        action_pos_6d = np_extract_ee_pose_6d(all_robot_state_6d[1:])
        action_pos_6d = np.concatenate([action_pos, action_delta_6d[:, -1:]], axis=1)

    # Extract the rewards, skills, and parts_poses from the pickle file
    reward = np.array(data["rewards"], dtype=np.float32)
    skill = np.array(data["skills"], dtype=np.float32) if "skills" in data else np.zeros_like(reward)
    augment_states = data["augment_states"] if "augment_states" in data else np.zeros_like(reward)

    # Sanity check that all arrays are the same length
    assert len(robot_state_6d) == len(action_delta_6d), (
        f"Mismatch in {pickle_path}, lengths differ by {len(robot_state_6d) - len(action_delta_6d)}"
    )

    # Extract the pickle file name as the path after `raw` in the path
    pickle_file = "/".join(pickle_path.parts[pickle_path.parts.index("raw") + 1 :])

    processed_data = {
        "robot_state": robot_state_6d,
        "color_image1": color_image1,
        "color_image2": color_image2,
        "action/delta": action_delta_6d,
        "action/pos": action_pos_6d,
        "reward": reward,
        "skill": skill,
        "augment_states": augment_states,
        "parts_poses": parts_poses,
        "episode_length": len(action_delta_6d),
        "furniture": data["furniture"],
        "success": 1 if data["success"] == "partial_success" else int(data["success"]),
        "failure_idx": data.get("failure_idx", -1),
        "critical_state_id": data.get("critical_state", -1),
        "pickle_file": pickle_file,
    }

    return processed_data


def _append_episode_to_zarr(z, data, current_episode_end):
    """Append a single episode's processed data to an open Zarr store."""
    for key in [
        "robot_state",
        "color_image1",
        "color_image2",
        "action/delta",
        "action/pos",
        "reward",
        "skill",
        "parts_poses",
        "augment_states",
    ]:
        z[key].append(data[key])

    new_episode_end = current_episode_end + data["episode_length"]
    z["episode_ends"].append(np.array([new_episode_end], dtype=np.uint32))

    for key in ["furniture", "pickle_file"]:
        z[key].append(np.array([data[key]], dtype=object))
    for key in ["success", "failure_idx", "critical_state_id"]:
        z[key].append(np.array([data[key]]))

    return new_episode_end


def stream_process_and_write_to_zarr(
    pickle_paths,
    z,
    noop_threshold,
    num_threads,
    calculate_pos_action_from_delta=False,
    initial_episode_end=0,
):
    """
    Process pickle files and write each episode to Zarr immediately to avoid OOM.

    Tasks are submitted in batches of num_threads. Each batch is fully written and freed
    before the next batch is submitted, bounding peak memory to num_threads episodes.
    """
    current_episode_end = initial_episode_end

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        for batch_start in tqdm(
            range(0, len(pickle_paths), num_threads), desc="Processing and writing files"
        ):
            batch_paths = pickle_paths[batch_start : batch_start + num_threads]
            futures = [
                executor.submit(
                    process_pickle_file, path, noop_threshold, calculate_pos_action_from_delta
                )
                for path in batch_paths
            ]
            for future in futures:
                data = future.result()
                current_episode_end = _append_episode_to_zarr(z, data, current_episode_end)
                del data
            del futures


# === Entry Point of the Script ===
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", "-e", type=str, nargs="+", default=None)
    parser.add_argument("--furniture", "-f", type=str, default=None, nargs="+")
    parser.add_argument(
        "--source",
        "-s",
        type=str,
        choices=["scripted", "rollout", "teleop", "augmentation"],
        default=None,
        nargs="+",
    )
    parser.add_argument(
        "--randomness",
        "-r",
        type=str,
        default="low",
        nargs="+",
    )
    parser.add_argument(
        "--demo-outcome",
        "-d",
        type=str,
        choices=["success", "failure", "partial_success"],
        default="success",
        nargs="+",
    )
    parser.add_argument("--calculate-pos-action-from-delta", action="store_true", default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--randomize-order", action="store_true")
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--n-cpus", type=int, default=1)
    args = parser.parse_args()

    pickle_paths: List[Path] = sorted(
        get_raw_paths(
            environment=args.env,
            task=args.furniture,
            demo_source=args.source,
            randomness=args.randomness,
            demo_outcome=args.demo_outcome,
            # print("NBNB: IMplemented hack for balancing lamp demos")
            # demo_outcome="balanced",
        )
    )

    if args.randomize_order:
        print(f"Using random seed: {args.random_seed}")
        random.seed(args.random_seed)
        random.shuffle(pickle_paths)

    if args.max_files is not None:
        pickle_paths = pickle_paths[: args.max_files]

    print(f"Found {len(pickle_paths)} pickle files")

    output_path = get_processed_path(
        environment=args.env,
        task=args.furniture,
        demo_source=args.source,
        randomness=args.randomness,
        demo_outcome=args.demo_outcome,
    )

    print(f"Output path: {output_path}")

    if output_path.exists() and not args.overwrite:
        raise ValueError(f"Output path already exists: {output_path}. Use --overwrite to overwrite.")

    # Process all pickle files
    chunksize = 1_000
    noop_threshold = 0.0
    n_cpus = min(os.cpu_count(), args.n_cpus)

    print(f"Processing pickle files with {n_cpus} CPUs, chunksize={chunksize}, noop_threshold={noop_threshold}")

    # Process the first file to learn per-timestep shapes, then initialize the Zarr store.
    print("Reading first pickle file to determine data shapes...")
    first_data = process_pickle_file(pickle_paths[0], noop_threshold, args.calculate_pos_action_from_delta)

    data_shapes_and_dtypes = [
        # Per-timestep arrays — pass trailing shape only (first dim is variable length)
        ("robot_state", first_data["robot_state"].shape[1:], np.float32),
        ("color_image1", first_data["color_image1"].shape[1:], np.uint8),
        ("color_image2", first_data["color_image2"].shape[1:], np.uint8),
        ("action/delta", first_data["action/delta"].shape[1:], np.float32),
        ("action/pos", first_data["action/pos"].shape[1:], np.float32),
        ("parts_poses", first_data["parts_poses"].shape[1:], np.float32),
        ("reward", first_data["reward"].shape[1:], np.float32),
        ("skill", first_data["skill"].shape[1:], np.float32),
        ("augment_states", first_data["augment_states"].shape[1:], np.float32),
        # Per-episode scalars/strings
        ("episode_ends", (), np.uint32),
        ("furniture", (), object),
        ("success", (), np.uint8),
        ("failure_idx", (), np.int32),
        ("critical_state_id", (), np.int32),
        ("pickle_file", (), object),
    ]

    # Initialize Zarr store with empty dimensions
    z = initialize_zarr_store_empty(output_path, data_shapes_and_dtypes, chunksize=chunksize)

    # Write the first episode (already processed above), then stream the rest
    current_episode_end = _append_episode_to_zarr(z, first_data, 0)
    del first_data

    stream_process_and_write_to_zarr(
        pickle_paths[1:],
        z,
        noop_threshold,
        n_cpus,
        calculate_pos_action_from_delta=args.calculate_pos_action_from_delta,
        initial_episode_end=current_episode_end,
    )

    # Update final metadata
    z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    z.attrs["noop_threshold"] = noop_threshold
    z.attrs["chunksize"] = chunksize
    z.attrs["rotation_mode"] = "rot_6d"
    z.attrs["n_episodes"] = len(z["episode_ends"])
    z.attrs["n_timesteps"] = len(z["action/delta"])
    z.attrs["mean_episode_length"] = round(len(z["action/delta"]) / len(z["episode_ends"]))
    z.attrs["calculated_pos_action_from_delta"] = args.calculate_pos_action_from_delta
    z.attrs["randomize_order"] = args.randomize_order
    z.attrs["random_seed"] = args.random_seed
    z.attrs["demo_source"] = args.source[0]
    # z.attrs["balanced"] = True
