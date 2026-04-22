"""
Translate a FurnitureBench zarr store into the diffusion policy format.

FurnitureBench layout (input):
    root/
      episode_ends    (E,)
      action/delta    (T, Da)
      action/pos      (T, Da)
      color_image1    (T, H, W, C)
      color_image2    (T, H, W, C)
      robot_state     (T, Ds)
      ...

Diffusion policy format layout (output):
    root/
      meta/
        episode_ends  (E,)
      data/
        action        (T, Da)       ← from action/delta (explicit remap)
        action_pos    (T, Da)       ← from action/pos  ('/' replaced with '_')
        color_image1  (T, H, W, C)
        color_image2  (T, H, W, C)
        robot_state   (T, Ds)
        ...

Key naming: entries in _KEY_MAP are remapped explicitly; all other zarr paths
with '/' separators have '/' replaced with '_' to keep every output key a flat
string (zarr would interpret '/' as a nested group path).

Usage:
  python src/data_processing/process_zarr.py dataset/imitation-juicer-data-processed-001/processed/sim/one_leg/teleop/low/success.zarr \
    --output dataset/imitation-juicer-data-processed-001/processed/sim/one_leg/teleop/low/success_translated.zarr
"""

import argparse
import os
import pathlib
import sys

import numpy as np
import zarr
from tqdm import tqdm

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT_DIR)

from diffusion_policy.common.replay_buffer import ReplayBuffer

# Explicit remaps take priority; all other slash-separated zarr paths have
# their '/' replaced with '_' to avoid zarr treating them as nested groups
# (e.g. "action/pos" → "action_pos", preventing a conflict with "action").
_KEY_MAP = {"action/delta": "action"}


def _zarr_path_to_key(zarr_path: str) -> str:
    """Return the output key for a given zarr path."""
    if zarr_path in _KEY_MAP:
        return _KEY_MAP[zarr_path]
    return zarr_path.replace("/", "_")


def discover_keys(group):
    """
    Recursively find all leaf arrays in a zarr group.
    Returns a list of (zarr_path, zarr.Array) tuples, excluding episode_ends.
    """
    results = []

    def _walk(g, prefix=""):
        for name in g:
            path = f"{prefix}/{name}" if prefix else name
            item = g[name]
            if isinstance(item, zarr.Array):
                if path != "episode_ends":
                    results.append(path)
            elif isinstance(item, zarr.Group):
                _walk(item, path)

    _walk(group)
    return results


def _classify_arrays(source, episode_ends):
    """
    Classify all leaf arrays as 'per_timestep' or 'per_episode'.

    Per-episode arrays have shape[0] == n_episodes; they store one value per
    episode rather than one per timestep.  All others are treated as
    per-timestep and sliced with [start:end].
    """
    n_episodes = len(episode_ends)
    total_steps = int(episode_ends[-1])

    per_episode_paths = set()
    all_paths = discover_keys(source)
    for path in all_paths:
        arr = source[path]
        if arr.shape[0] == n_episodes and arr.shape[0] != total_steps:
            per_episode_paths.add(path)

    return all_paths, per_episode_paths


def _build_inv_key_map(all_zarr_paths):
    """Build output_key → zarr_path reverse lookup for --keys resolution."""
    return {_zarr_path_to_key(p): p for p in all_zarr_paths}


def translate(source_zarr, output_zarr, keys=None):
    """
    Translate a FurnitureBench zarr into a standard ReplayBuffer zarr.

    Reads one episode at a time and writes it to the output via
    ReplayBuffer.add_episode, so only one episode is in memory at once.

    Per-episode scalar arrays (e.g. furniture, success, pickle_file) whose
    first dimension equals n_episodes are broadcast to every timestep in the
    episode so that ReplayBuffer.add_episode receives uniform-length arrays.
    """
    source = zarr.open(str(source_zarr), mode="r")

    if "episode_ends" not in source:
        raise ValueError(
            f"No 'episode_ends' array found in {source_zarr}. This doesn't look like a FurnitureBench zarr."
        )

    episode_ends = source["episode_ends"][:].astype(np.int64)
    n_episodes = len(episode_ends)

    all_zarr_paths, per_episode_paths = _classify_arrays(source, episode_ends)

    if keys is not None:
        # User specified output key names — resolve back to zarr paths.
        inv_map = _build_inv_key_map(all_zarr_paths)
        selected_zarr_paths = []
        for key in keys:
            zarr_path = inv_map.get(key)
            if zarr_path is None:
                raise ValueError(
                    f"Key '{key}' not found in {source_zarr}. "
                    f"Available keys: {[_zarr_path_to_key(p) for p in all_zarr_paths]}"
                )
            selected_zarr_paths.append(zarr_path)
    else:
        selected_zarr_paths = all_zarr_paths

    # Build zarr_path → output_key mapping
    path_to_key = {p: _zarr_path_to_key(p) for p in selected_zarr_paths}

    if per_episode_paths:
        broadcast_keys = [_zarr_path_to_key(p) for p in per_episode_paths if p in selected_zarr_paths]
        print(f"Per-episode arrays (will be broadcast): {broadcast_keys}")

    print(f"Source:   {source_zarr}")
    print(f"Output:   {output_zarr}")
    print(f"Episodes: {n_episodes}")
    print(f"Keys:     {list(path_to_key.values())}")

    out = ReplayBuffer.create_from_path(str(output_zarr), mode="w")

    for ep_idx in tqdm(range(n_episodes), desc="Translating"):
        start = 0 if ep_idx == 0 else int(episode_ends[ep_idx - 1])
        end = int(episode_ends[ep_idx])
        ep_len = end - start

        episode_data = {}
        for zarr_path, out_key in path_to_key.items():
            if zarr_path in per_episode_paths:
                # Broadcast the single per-episode value to all timesteps.
                val = source[zarr_path][ep_idx]  # scalar or small array
                episode_data[out_key] = np.broadcast_to(val, (ep_len,) + np.shape(val)).copy()
            else:
                episode_data[out_key] = source[zarr_path][start:end]

        compressors = {k: "disk" if v.ndim == 4 else "default" for k, v in episode_data.items()}
        out.add_episode(episode_data, compressors=compressors)

    print(f"\nDone. {out.n_episodes} episodes, {out.n_steps} total frames.")


def main():
    parser = argparse.ArgumentParser(description="Translate a FurnitureBench zarr into standard ReplayBuffer format.")
    parser.add_argument(
        "source_zarr",
        type=str,
        help="Path to the FurnitureBench zarr store.",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Output zarr path. Defaults to <source_dir>/<source_stem>_translated.zarr.",
    )
    parser.add_argument(
        "--keys",
        "-k",
        type=str,
        nargs="+",
        default=None,
        help=(
            "Subset of keys to include in the output (using normalised names, "
            'e.g. "action" not "action/delta"). Default: all keys.'
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output zarr if it already exists.",
    )
    args = parser.parse_args()

    source = pathlib.Path(args.source_zarr)
    if args.output is not None:
        output = pathlib.Path(args.output)
    else:
        output = source.parent / f"{source.stem}_translated.zarr"

    if output.exists() and not args.overwrite:
        raise ValueError(f"Output path already exists: {output}. Use --overwrite to overwrite.")

    translate(source, output, keys=args.keys)


if __name__ == "__main__":
    main()
