"""
Truncate a processed zarr dataset so that each episode ends at the first
timestep where the furniture is fully assembled (cumulative reward == total_reward).

Episodes in the public FurnitureBench dataset sometimes include idle steps
recorded after full assembly. This script removes those trailing steps.

Episodes where no success reward is found are kept as-is with a warning.

Processes one episode at a time to avoid loading the entire dataset into memory.

Usage:
    python -m src.data_processing.truncate_at_success path/to/source.zarr --output path/to/output.zarr
"""

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import zarr
from furniture_bench.config import config
from numcodecs import JSON, Blosc
from tqdm import tqdm


def _walk_arrays(group, prefix=""):
    """Yield (zarr_path, array) for every leaf array under group, excluding episode_ends."""
    for name in group:
        path = f"{prefix}/{name}" if prefix else name
        item = group[name]
        if isinstance(item, zarr.Array):
            if path != "episode_ends":
                yield path, item
        elif isinstance(item, zarr.Group):
            yield from _walk_arrays(item, path)


def _classify_arrays(src, episode_ends):
    """
    Split leaf arrays into per-timestep and per-episode sets.

    Per-episode arrays have shape[0] == n_episodes AND shape[0] != total_steps
    (to handle edge cases where the two happen to be equal).
    """
    n_episodes = len(episode_ends)
    total_steps = int(episode_ends[-1])
    per_episode = set()
    all_paths = []
    for path, arr in _walk_arrays(src):
        all_paths.append(path)
        if arr.shape[0] == n_episodes and arr.shape[0] != total_steps:
            per_episode.add(path)
    return all_paths, per_episode


def _init_output_zarr(src, out_path, all_paths, per_episode_paths, chunksize=32):
    """Create an empty output zarr with the same dataset schemas as src."""
    compressor = Blosc(cname="lz4", clevel=5)
    z = zarr.open(str(out_path), mode="w")
    z.attrs["time_created"] = datetime.now().astimezone().isoformat()

    z.create_dataset("episode_ends", shape=(0,), dtype=np.uint32, chunks=(chunksize,))

    for zarr_path in all_paths:
        arr = src[zarr_path]
        trailing = arr.shape[1:]
        if arr.dtype == object:
            z.create_dataset(
                zarr_path,
                shape=(0,) + trailing,
                dtype=object,
                chunks=(chunksize,) + trailing,
                object_codec=JSON(),
            )
        elif "color_image" in zarr_path:
            z.create_dataset(
                zarr_path,
                shape=(0,) + trailing,
                dtype=arr.dtype,
                chunks=(chunksize,) + trailing,
                compressor=compressor,
            )
        else:
            z.create_dataset(
                zarr_path,
                shape=(0,) + trailing,
                dtype=arr.dtype,
                chunks=(chunksize,) + trailing,
            )

    return z


def _find_keep_count(reward_slice, total_reward, ep_len):
    """
    Return the number of timesteps to keep from this episode.

    Mirrors the done_when_assembled logic in upstream preprocess_data.py:
    find the first index where cumulative reward reaches total_reward, then
    keep one additional step so the assembled observation is included.

    Returns None if no success reward is found.
    """
    cumulative = 0.0
    reward_idx = None
    for idx, rew in enumerate(reward_slice):
        cumulative += float(rew)
        if cumulative >= total_reward:
            reward_idx = idx
            break

    if reward_idx is None:
        return None

    # Include one step after the final reward if the episode is long enough.
    return reward_idx + 1 if reward_idx + 2 < ep_len else ep_len - 1


def truncate_zarr(source_zarr: Path, output_zarr: Path):
    src = zarr.open(str(source_zarr), mode="r")

    if "episode_ends" not in src:
        raise ValueError(f"No 'episode_ends' array found in {source_zarr}")

    episode_ends = src["episode_ends"][:].astype(np.int64)
    n_episodes = len(episode_ends)
    total_steps = int(episode_ends[-1])

    all_paths, per_episode_paths = _classify_arrays(src, episode_ends)

    print(f"Source:           {source_zarr}")
    print(f"Output:           {output_zarr}")
    print(f"Episodes:         {n_episodes}")
    print(f"Total timesteps:  {total_steps}")
    print(f"Per-episode keys: {[p for p in all_paths if p in per_episode_paths]}")
    print(f"Per-timestep keys:{[p for p in all_paths if p not in per_episode_paths]}")

    z = _init_output_zarr(src, output_zarr, all_paths, per_episode_paths)

    current_end = 0
    n_no_reward = 0
    total_removed = 0

    for ep_idx in tqdm(range(n_episodes), desc="Truncating"):
        start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        end = int(episode_ends[ep_idx])
        ep_len = end - start

        furniture_name = str(src["furniture"][ep_idx])
        total_reward = config["furniture"][furniture_name]["total_reward"]

        reward_slice = src["reward"][start:end]
        keep = _find_keep_count(reward_slice, total_reward, ep_len)

        if keep is None:
            tqdm.write(
                f"  WARNING: ep {ep_idx} ({furniture_name}): no success reward found, keeping all {ep_len} steps"
            )
            keep = ep_len
            n_no_reward += 1
        else:
            removed = ep_len - keep
            total_removed += removed
            if removed > 0:
                tqdm.write(f"  ep {ep_idx}: {ep_len} -> {keep} steps (-{removed})")

        for zarr_path in all_paths:
            if zarr_path in per_episode_paths:
                z[zarr_path].append(src[zarr_path][ep_idx : ep_idx + 1])
            else:
                z[zarr_path].append(src[zarr_path][start : start + keep])

        current_end += keep
        z["episode_ends"].append(np.array([current_end], dtype=np.uint32))

    # Copy source metadata then stamp our own.
    for key, val in src.attrs.items():
        z.attrs[key] = val
    z.attrs["time_finished"] = datetime.now().astimezone().isoformat()
    z.attrs["truncated_at_success"] = True
    z.attrs["n_episodes"] = len(z["episode_ends"])
    z.attrs["n_timesteps"] = int(z["episode_ends"][-1])
    z.attrs["mean_episode_length"] = round(int(z["episode_ends"][-1]) / len(z["episode_ends"]))

    print(
        f"\nDone. {n_episodes} episodes processed, {n_no_reward} had no reward signal. "
        f"Total steps removed: {total_removed}."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Truncate a processed zarr dataset at the point of successful assembly."
    )
    parser.add_argument("source_zarr", type=str, help="Path to the source zarr store.")
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output zarr path. Defaults to <source_stem>_truncated.zarr in the same directory.",
    )
    args = parser.parse_args()

    source = Path(args.source_zarr)
    output = Path(args.output) if args.output else source.parent / f"{source.stem}_truncated.zarr"

    truncate_zarr(source, output)
