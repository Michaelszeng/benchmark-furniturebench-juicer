"""
Convert DART-rollback pickle files (scripted_dart.py output) into a zarr dataset.

Reads .pkl / .pkl.xz files from:
    $DATA_DIR_RAW/raw/{env}/{task}/{source}/{randomness}/

Each pickle contains:
  observations   : list of N obs dicts (color_image1, color_image2, robot_state, parts_poses)
  action_chunks  : list of N arrays, each (chunk_size, 8) — clean lookahead actions
  actions        : list of N arrays, each (8,)             — clean single-step actions
  rewards        : list of N floats
  skills         : list of N ints

Applies the same transforms as process_pickles.py:
  - Converts quaternion robot state to 6D rotation
  - Converts 8D quaternion actions to 10D 6D-rotation actions
  - Clips z-axis rotation deltas to ±0.35 and xyz position deltas to ±0.025

Zarr layout written (FurnitureBench convention, compatible with process_zarr.py):

    root/
      # Per-timestep (concat across all episodes, length = sum of episode lengths)
      robot_state     (T, 14)                float32
      color_image1    (T, H, W, 3)           uint8
      color_image2    (T, H, W, 3)           uint8
      action/delta    (T, 10)                float32   ← single clean step (6D rot)
      action/chunk    (T, chunk_size, 10)    float32   ← clean lookahead chunk (6D rot)
      parts_poses     (T, P)                 float32
      reward          (T,)                   float32
      skill           (T,)                   float32

      # Per-episode (length = number of episodes)
      episode_ends    (E,)                   uint32
      furniture       (E,)                   object
      success         (E,)                   uint8
      chunk_size      (E,)                   uint32
      pickle_file     (E,)                   object

After running process_zarr.py the translated keys are:
  action/delta  → action
  action/chunk  → action_chunk

Usage:
    python -m src.data_processing.process_pickles_dart \\
        --env sim --furniture one_leg --source scripted_chunk \\
        --randomness low --demo-outcome success [--overwrite] [--n-cpus 8]
"""

import argparse
import os

if "DATA_DIR_RAW" not in os.environ:
    os.environ["DATA_DIR_RAW"] = "dataset"
if "DATA_DIR_PROCESSED" not in os.environ:
    os.environ["DATA_DIR_PROCESSED"] = "dataset"

import random
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
import zarr
from furniture_bench.robot.robot_state import filter_and_concat_robot_state
from numcodecs import JSON, Blosc
from tqdm import tqdm

from src.common.files import get_processed_path, get_raw_paths
from src.common.geometry import (
    np_action_quat_to_6d_rotation,
    np_proprioceptive_quat_to_6d_rotation,
)
from src.data_processing.utils import clip_axis_rotation
from src.visualization.render_mp4 import unpickle_data


def _initialize_zarr_store(out_path, data_shapes_and_dtypes, chunksize=32):
    z = zarr.open(str(out_path), mode="w")
    z.attrs["time_created"] = datetime.now().astimezone().isoformat()
    compressor = Blosc(cname="lz4", clevel=5)
    for name, shape, dtype in data_shapes_and_dtypes:
        if "color_image" in name:
            z.create_dataset(name, shape=(0,) + shape, dtype=dtype, chunks=(chunksize,) + shape, compressor=compressor)
        elif dtype == object:
            z.create_dataset(name, shape=(0,), dtype=dtype, chunks=(chunksize,), object_codec=JSON())
        else:
            z.create_dataset(name, shape=(0,) + shape, dtype=dtype, chunks=(chunksize,) + shape)
    return z


def _convert_actions(actions_quat: np.ndarray) -> np.ndarray:
    """Convert (..., 8) quaternion actions to (..., 10) 6D-rotation actions with clipping."""
    orig_shape = actions_quat.shape
    flat = actions_quat.reshape(-1, 8)
    flat_6d = np_action_quat_to_6d_rotation(flat)
    flat_6d = clip_axis_rotation(flat_6d, clip_mag=0.35, axis="z")
    flat_6d[:, :3] = np.clip(flat_6d[:, :3], -0.025, 0.025)
    return flat_6d.reshape(orig_shape[:-1] + (10,))


def process_pickle_file(pickle_path: Path):
    data = unpickle_data(pickle_path)
    obs = data["observations"]
    N = len(obs)

    # Images and robot state — no [:-1] needed (no terminal observation in DART format)
    color_image1 = np.array([o["color_image1"] for o in obs], dtype=np.uint8)
    color_image2 = np.array([o["color_image2"] for o in obs], dtype=np.uint8)

    if isinstance(obs[0]["robot_state"], dict):
        robot_state_quat = np.array([filter_and_concat_robot_state(o["robot_state"]) for o in obs], dtype=np.float32)
    else:
        robot_state_quat = np.array([o["robot_state"] for o in obs], dtype=np.float32)

    robot_state_6d = np_proprioceptive_quat_to_6d_rotation(robot_state_quat)
    parts_poses = np.array([o["parts_poses"] for o in obs], dtype=np.float32)

    # Single-step clean actions: (N, 8) → (N, 10)
    action_delta_6d = _convert_actions(np.array(data["actions"], dtype=np.float32))

    # Clean action chunks: (N, chunk_size, 8) → (N, chunk_size, 10)
    action_chunk_6d = _convert_actions(np.array(data["action_chunks"], dtype=np.float32))

    chunk_size = action_chunk_6d.shape[1]

    reward = np.array(data["rewards"], dtype=np.float32)
    skill = np.array(data["skills"], dtype=np.float32) if "skills" in data else np.zeros(N, dtype=np.float32)

    assert len(robot_state_6d) == len(action_delta_6d) == N, (
        f"Length mismatch in {pickle_path}: obs={N}, actions={len(action_delta_6d)}"
    )

    pickle_file = "/".join(pickle_path.parts[pickle_path.parts.index("raw") + 1 :])

    return {
        "robot_state": robot_state_6d,
        "color_image1": color_image1,
        "color_image2": color_image2,
        "action/delta": action_delta_6d,
        "action/chunk": action_chunk_6d,
        "parts_poses": parts_poses,
        "reward": reward,
        "skill": skill,
        "episode_length": N,
        "chunk_size": chunk_size,
        "furniture": data["furniture"],
        "success": 1 if data["success"] == "partial_success" else int(data["success"]),
        "pickle_file": pickle_file,
    }


_TIMESTEP_KEYS = [
    "robot_state",
    "color_image1",
    "color_image2",
    "action/delta",
    "action/chunk",
    "reward",
    "skill",
    "parts_poses",
]


def _check_shapes(z, data) -> List[str]:
    """Return a list of mismatch descriptions, empty if all shapes are compatible."""
    mismatches = []
    for key in _TIMESTEP_KEYS:
        expected = z[key].shape[1:]
        actual = data[key].shape[1:]
        if expected != actual:
            mismatches.append(f"  {key}: zarr expects {expected}, got {actual}")
    return mismatches


def _append_episode_to_zarr(z, data, current_episode_end):
    for key in _TIMESTEP_KEYS:
        z[key].append(data[key])

    new_episode_end = current_episode_end + data["episode_length"]
    z["episode_ends"].append(np.array([new_episode_end], dtype=np.uint32))
    z["chunk_size"].append(np.array([data["chunk_size"]], dtype=np.uint32))
    z["success"].append(np.array([data["success"]], dtype=np.uint8))
    for key in ["furniture", "pickle_file"]:
        z[key].append(np.array([data[key]], dtype=object))

    return new_episode_end


def _stream_process_and_write(pickle_paths, z, num_threads, initial_episode_end=0):
    current_episode_end = initial_episode_end
    n_skipped = 0
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        with tqdm(total=len(pickle_paths), desc="Processing") as pbar:
            for batch_start in range(0, len(pickle_paths), num_threads):
                batch = pickle_paths[batch_start : batch_start + num_threads]
                futures = {executor.submit(process_pickle_file, p): p for p in batch}
                for f, path in futures.items():
                    data = f.result()
                    mismatches = _check_shapes(z, data)
                    if mismatches:
                        tqdm.write(f"  SKIPPED {path.name} — shape mismatch:\n" + "\n".join(mismatches))
                        n_skipped += 1
                    else:
                        current_episode_end = _append_episode_to_zarr(z, data, current_episode_end)
                    del data
                del futures
                pbar.update(len(batch))
    if n_skipped:
        print(f"\nWarning: skipped {n_skipped} pickle(s) due to shape mismatches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env", "-e", type=str, nargs="+", default=None)
    parser.add_argument("--furniture", "-f", type=str, nargs="+", default=None)
    parser.add_argument("--source", "-s", type=str, nargs="+", default=None)
    parser.add_argument("--randomness", "-r", type=str, nargs="+", default="low")
    parser.add_argument(
        "--demo-outcome",
        "-d",
        type=str,
        nargs="+",
        choices=["success", "failure", "partial_success"],
        default="success",
    )
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
        )
    )
    pickle_paths = [p for p in pickle_paths if p.suffix != ".tmp"]

    if args.randomize_order:
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

    chunksize = 1_000
    n_cpus = min(os.cpu_count(), args.n_cpus)
    print(f"Processing with {n_cpus} CPUs, chunksize={chunksize}")

    print("Reading first pickle to determine shapes...")
    first_data = process_pickle_file(pickle_paths[0])

    data_shapes_and_dtypes = [
        ("robot_state", first_data["robot_state"].shape[1:], np.float32),
        ("color_image1", first_data["color_image1"].shape[1:], np.uint8),
        ("color_image2", first_data["color_image2"].shape[1:], np.uint8),
        ("action/delta", first_data["action/delta"].shape[1:], np.float32),
        ("action/chunk", first_data["action/chunk"].shape[1:], np.float32),
        ("parts_poses", first_data["parts_poses"].shape[1:], np.float32),
        ("reward", first_data["reward"].shape[1:], np.float32),
        ("skill", first_data["skill"].shape[1:], np.float32),
        # Per-episode
        ("episode_ends", (), np.uint32),
        ("chunk_size", (), np.uint32),
        ("success", (), np.uint8),
        ("furniture", (), object),
        ("pickle_file", (), object),
    ]

    z = _initialize_zarr_store(output_path, data_shapes_and_dtypes, chunksize=chunksize)
    current_episode_end = _append_episode_to_zarr(z, first_data, 0)
    del first_data

    _stream_process_and_write(pickle_paths[1:], z, n_cpus, initial_episode_end=current_episode_end)

    z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    z.attrs["chunksize"] = chunksize
    z.attrs["rotation_mode"] = "rot_6d"
    z.attrs["n_episodes"] = len(z["episode_ends"])
    z.attrs["n_timesteps"] = len(z["action/delta"])
    z.attrs["mean_episode_length"] = round(len(z["action/delta"]) / len(z["episode_ends"]))
    z.attrs["chunk_size"] = int(z["chunk_size"][0])
    z.attrs["demo_source"] = args.source[0] if isinstance(args.source, list) else args.source
    z.attrs["randomize_order"] = args.randomize_order
    z.attrs["random_seed"] = args.random_seed
