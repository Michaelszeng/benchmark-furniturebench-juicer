"""Evaluate a diffusion_policy checkpoint on FurnitureBench simulation.

Usage:
    python src/eval/evaluate_model_custom.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --furniture one_leg \
        --n-rollouts 10 \
        --n-envs 1 \
        --n-action-steps 1
"""

import argparse
import collections
import csv
import datetime
import math
import os
import pickle
import time
from pathlib import Path

import dill
import furniture_bench  # noqa: F401 — registers FurnitureSim envs
import gym
import imageio
import numpy as np
import torch
from furniture_bench.envs.observation import DEFAULT_STATE_OBS, DEFAULT_VISUAL_OBS
from furniture_bench.sim_config import sim_config
from omegaconf import OmegaConf

from src.common.geometry import proprioceptive_quat_to_6d_rotation
from src.common.tasks import task_timeout
from src.data_processing.utils import resize, resize_crop
from src.visualization.render_mp4 import unpickle_data

# Register the "eval" resolver used in diffusion_policy configs.
OmegaConf.register_new_resolver("eval", eval, replace=True)


# def preprocess_images(imgs: torch.Tensor) -> torch.Tensor:
#     """Convert (B, H, W, 3) uint8 → (B, 3, 224, 224) float [0, 1]."""
#     imgs = imgs.float() / 255.0
#     imgs = imgs.permute(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W)
#     imgs = TF.resize(imgs, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
#     return imgs


def preprocess_obs(obs: dict, device: torch.device, obs_keys: set) -> dict:
    """
    Handles both image-based policies (color_image1, color_image2, robot_state)
    and state-based policies (robot_state, parts_poses).

    Args:
        obs: raw observation dict from env.step / env.reset.
        device: torch device to move tensors to.
        obs_keys: set of observation key names the policy expects
                  (from cfg.shape_meta.obs).
    """
    result = {"robot_state": proprioceptive_quat_to_6d_rotation(obs["robot_state"].float().to(device))}
    if "color_image1" in obs_keys:
        # resize() matches data_collector.py: 1280x720 → 320x240, same FOV.
        result["color_image1"] = resize(obs["color_image1"]).float().to(device)
    if "color_image2" in obs_keys:
        # resize_crop() matches data_collector.py: 1280x720 → 320x240 center-crop.
        result["color_image2"] = resize_crop(obs["color_image2"]).float().to(device)
    if "parts_poses" in obs_keys:
        result["parts_poses"] = obs["parts_poses"].float().to(device)
    return result


def build_obs_dict(obs_deque: collections.deque, device: torch.device) -> dict:
    """Stack obs deque into obs_dict for policy.predict_action.

    Each element of obs_deque is a dict {key: (n_envs, ...)}.
    Returns {"obs": {key: (n_envs, T_obs, ...)}} on ``device``.
    """
    keys = obs_deque[0].keys()
    obs_stacked = {k: torch.stack([o[k] for o in obs_deque], dim=1) for k in keys}
    return {"obs": obs_stacked}


def load_policy(checkpoint_path: str, device: torch.device):
    """Load a diffusion_policy workspace + policy from a .ckpt file.

    Also loads (or generates) the paired normalizer.pt that lives next to the
    checkpoint's parent checkpoints/ directory.

    Returns:
        policy: the EMA (or base) model, on ``device``, in eval mode.
        cfg: the OmegaConf config baked into the checkpoint.
    """
    import hydra  # import here to avoid hydra global-init side-effects
    from diffusion_policy.workspace.base_workspace import BaseWorkspace  # noqa

    payload = torch.load(open(checkpoint_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    # TEMPORARY FIX
    # Checkpoints saved with DataParallel have every key prefixed with "module.".
    # Strip it so the state dict aligns with a plain (non-wrapped) model.
    # Also drop training-only submodule keys (e.g. mask_generator) that are not
    # part of the inference-time policy architecture.
    _INFERENCE_ONLY_DROP_PREFIXES = ("mask_generator.",)
    if "state_dicts" in payload:
        for sd_key, sd in payload["state_dicts"].items():
            if any(k.startswith("module.") for k in sd):
                sd = {(k[len("module.") :] if k.startswith("module.") else k): v for k, v in sd.items()}
            payload["state_dicts"][sd_key] = {
                k: v for k, v in sd.items() if not any(k.startswith(pfx) for pfx in _INFERENCE_ONLY_DROP_PREFIXES)
            }
    # END TEMPORARY FIX

    cls = hydra.utils.get_class(cfg._target_)
    workspace: BaseWorkspace = cls(cfg)
    # Exclude optimizer/scheduler state: not needed for inference, and optimizer
    # param-group sizes from DataParallel training don't match the unwrapped model.
    workspace.load_payload(payload, exclude_keys=["optimizer", "lr_scheduler"], include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model

    # --- normalizer ---
    ckpt_path = Path(checkpoint_path)
    normalizer_path = ckpt_path.parent.parent / "normalizer.pt"
    if normalizer_path.exists():
        print(f"Loading normalizer from {normalizer_path}")
        normalizer = torch.load(normalizer_path)
    else:
        print(f"Normalizer not found at {normalizer_path}, generating from dataset…")
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        normalizer = dataset.get_normalizer()
        torch.save(normalizer, normalizer_path)
        print(f"Saved normalizer to {normalizer_path}")

    policy.set_normalizer(normalizer)
    policy.to(device).eval()
    # Explicitly set obs_encoder to eval mode: R3M's ResNet18 uses BatchNorm, which
    # behaves catastrophically in training mode with the small batch sizes (n_obs_steps=2)
    # typical at inference time. policy.eval() should propagate here, but this is a
    # belt-and-suspenders guard in case something in the load path leaves it in train mode.
    if hasattr(policy, "obs_encoder"):
        policy.obs_encoder.eval()
    return policy, cfg


def _write_mp4(frames: list, path: Path, fps: int = 10) -> None:
    """Write a list of (H, W, 3) uint8 numpy frames to an MP4 file."""
    with imageio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p") as writer:
        for frame in frames:
            writer.append_data(frame)


def _write_summary(n_success: int, n_total: int, trial_records: list, summary_path: Path = "summary.txt") -> None:
    n_failure = sum(1 for r in trial_records if r["result"] == "failure")
    n_timeout = sum(1 for r in trial_records if r["result"] == "timeout")
    rate = n_success / n_total if n_total > 0 else 0.0
    avg_steps = sum(r["trial_time"] for r in trial_records) / len(trial_records) if trial_records else 0.0
    with open(summary_path, "w") as f:
        f.write(f"Trials completed : {n_total} / {args.n_rollouts}\n")
        f.write(f"Successes        : {n_success}\n")
        f.write(f"Failures         : {n_failure}\n")
        f.write(f"Timeouts         : {n_timeout}\n")
        f.write(f"Success rate     : {rate:.1%}\n")
        f.write(f"Avg trial steps  : {avg_steps:.1f}\n")


def _repair_csv_and_summary_from_pkl_data(pkl_path: Path):
    """Rewrite results.csv and summary.txt to match results.pkl; return (csv_file, csv_writer)."""
    saved = pickle.load(open(pkl_path, "rb"))
    out_dir, fields = pkl_path.parent, ["trial", "result", "reward", "trial_time"]
    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(saved["trials"])
    _write_summary(saved["n_success"], saved["n_total"], saved["trials"], out_dir / "summary.txt")
    csv_file = open(csv_path, "a", newline="")
    return csv_file, csv.DictWriter(csv_file, fieldnames=fields)


def load_dataset_init_states(dataset_path: str):
    """Load the first observation from pkl file(s) at dataset_path.

    dataset_path may be a single .pkl/.pkl.xz file or a directory containing
    such files.  Returns a list of raw obs dicts (robot_state dict +
    parts_poses array), matching the format expected by env.reset_to().
    """
    p = Path(dataset_path)
    if p.is_file():
        pkl_paths = [p]
    else:
        pkl_paths = sorted(p.glob("*.pkl*"))
    if not pkl_paths:
        raise FileNotFoundError(f"No pkl files found at {dataset_path}")
    states = []
    for pkl_path in pkl_paths:
        data = unpickle_data(pkl_path)
        states.append(data["observations"][0])
        break
    print(f"Loaded {len(states)} initial states from {dataset_path}")
    return states


def reset_to_dataset_states(env, states):
    """Reset all envs to the given dataset states and return observations.

    Replicates the refresh + zero-torque sequence that env.reset() performs,
    which reset_to() itself skips.
    """
    from isaacgym import gymtorch

    # Call env.reset() first so the gym.Wrapper bookkeeping is satisfied
    # (it raises "Cannot call step() before reset()" otherwise), then
    # immediately override the random initial state with our dataset states.
    env.reset()

    # env is a gym.Wrapper; the actual FurnitureSimEnv sits at env.unwrapped.
    sim = env.unwrapped

    sim.reset_to(states)  # sets dof state + root state for each env

    # Mimic env.reset(): apply zero torques then refresh so Isaac Gym registers
    # the new state before we read back observations.
    if sim.ctrl_mode == "osc":
        torque_action = torch.zeros_like(sim.dof_pos)
        sim.isaac_gym.set_dof_actuation_force_tensor(sim.sim, gymtorch.unwrap_tensor(torque_action))
    sim.refresh()
    sim.furniture.reset()
    sim.scripted_timeout = False
    sim.refresh()
    return sim._get_observation()


@torch.no_grad()
def run_rollout(
    env,
    policy,
    n_obs_steps: int,
    rollout_max_steps: int,
    device: torch.device,
    obs_keys: set,
    record_video: bool = False,
    n_action_steps: int = None,
    init_states: list = None,
) -> dict:
    """Run one round of parallel rollouts.

    Returns a dict with per-env arrays:
        success      (n_envs,) bool
        total_reward (n_envs,) float
        result       (n_envs,) str — "success", "timeout", or "failure"
        steps        int — total steps executed this round
        frames       (n_envs,) list[np.ndarray] — only present if record_video=True;
                     each element is a list of (H, W*2, 3) uint8 frames (cam1 | cam2)
    """
    n_envs = env.num_envs
    if init_states is not None:
        obs = reset_to_dataset_states(env, init_states)
    else:
        obs = env.reset()
    preprocessed = preprocess_obs(obs, device, obs_keys)
    obs_deque = collections.deque([preprocessed] * n_obs_steps, maxlen=n_obs_steps)
    action_queue: collections.deque = collections.deque()

    done = torch.zeros((n_envs, 1), dtype=torch.bool, device=device)
    total_reward = torch.zeros(n_envs, device=device)
    # Track the step at which each env first became done (-1 = never done).
    done_step = torch.full((n_envs,), -1, dtype=torch.long, device=device)
    step = 0

    # Per-env frame buffers: list of lists of (H, W*2, 3) uint8 numpy arrays.
    if record_video:
        frame_buffers = [[] for _ in range(n_envs)]

    while not done.all() and step < rollout_max_steps:
        if len(action_queue) == 0:
            obs_dict = build_obs_dict(obs_deque, device)
            result = policy.predict_action(obs_dict, use_DDIM=True)
            actions = result["action_pred"]
            n_steps = n_action_steps if n_action_steps is not None else actions.shape[1]
            for t in range(n_steps):
                action_queue.append(actions[:, t, :])

        action = action_queue.popleft()
        obs, reward, done, _ = env.step(action)
        total_reward += reward.squeeze(-1).float()

        # Record when each env finishes for the first time.
        newly_done = done.squeeze(-1) & (done_step == -1)
        done_step[newly_done] = step

        if record_video:
            # obs images are (n_envs, H, W, 3) uint8 tensors on GPU.
            imgs1 = obs["color_image1"].cpu().numpy()  # (n_envs, H, W, 3)
            imgs2 = obs["color_image2"].cpu().numpy()
            for env_idx in range(n_envs):
                side_by_side = np.concatenate([imgs1[env_idx], imgs2[env_idx]], axis=1)
                frame_buffers[env_idx].append(side_by_side)

        preprocessed = preprocess_obs(obs, device, obs_keys)
        obs_deque.append(preprocessed)
        step += 1

    n_parts = len(env.furniture.should_be_assembled)
    success_mask = (total_reward >= n_parts).cpu().numpy()
    done_step_np = done_step.cpu().numpy()

    results = []
    for env_idx in range(n_envs):
        if success_mask[env_idx]:
            results.append("success")
        elif done_step_np[env_idx] == -1:
            # Never went done — ran out of steps.
            results.append("timeout")
        else:
            results.append("failure")

    # Per-env step count: how many steps each individual trial ran.
    # done_step == -1 means the env never signalled done (timeout), so it ran for all steps.
    steps_per_env = np.where(done_step_np >= 0, done_step_np + 1, step)

    out = {
        "success": success_mask,
        "total_reward": total_reward.cpu().numpy(),
        "result": results,
        "steps": step,
        "steps_per_env": steps_per_env,
    }
    if record_video:
        out["frames"] = frame_buffers
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a diffusion_policy checkpoint on FurnitureBench.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .ckpt file")
    parser.add_argument(
        "--furniture",
        "-f",
        type=str,
        required=True,
        choices=["one_leg", "lamp", "round_table", "desk", "square_table", "cabinet", "chair", "stool"],
    )
    parser.add_argument("--n-rollouts", type=int, default=10)
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--randomness", type=str, default="low")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    parser.add_argument("--save-video", action="store_true", default=True)
    parser.add_argument("--no-save-video", dest="save_video", action="store_false")
    parser.add_argument(
        "--n-video-trials",
        type=int,
        default=20,
        help="Save videos for only the first N trials (default: 20). Set to -1 to save all.",
    )
    parser.add_argument(
        "--record-failures",
        action="store_true",
        default=False,
        help="If set, only save videos of failed trials, and save all of them (overrides --n-video-trials).",
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override action horizon (default: use value from checkpoint config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to write results (default: outputs/<date>/<time>)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume from an existing results.pkl in --output-dir (requires --output-dir)",
    )
    parser.add_argument(
        "--dataset-dir",
        type=str,
        default=None,
        help="(Testing) Path to a single .pkl/.pkl.xz file or a directory of such "
        "files; resets each rollout to their initial observations instead of random "
        "env.reset().",
    )
    args = parser.parse_args()

    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir to be specified")

    dataset_init_states = None
    if args.dataset_dir is not None:
        dataset_init_states = load_dataset_init_states(args.dataset_dir)

    device = torch.device(args.device)

    print(f"Loading policy from {args.checkpoint}")
    policy, cfg = load_policy(args.checkpoint, device)
    n_obs_steps: int = int(cfg.n_obs_steps)
    n_action_steps: int = args.n_action_steps  # None means use the value from the checkpoint config
    print(
        f"n_obs_steps={n_obs_steps}, n_action_steps={'from_cfg' if n_action_steps is None else n_action_steps}, furniture={args.furniture}, n_envs={args.n_envs}"
    )

    # Detect policy modality from shape_meta and configure env obs accordingly.
    policy_obs_keys = set(cfg.shape_meta.obs.keys())
    is_image_based = "color_image1" in policy_obs_keys
    env_obs_keys = DEFAULT_VISUAL_OBS if is_image_based else DEFAULT_STATE_OBS
    print(f"Policy type: {'image-based' if is_image_based else 'state-based'} (obs keys: {sorted(policy_obs_keys)})")

    if args.save_video and not is_image_based:
        print("Warning: --save-video requires camera observations; disabling for state-based policy.")
        args.save_video = False

    # Use the same timeout that scripted data collection uses (sim_config["scripted_timeout"]),
    # falling back to task_timeout for tasks not listed there.
    rollout_max_steps = sim_config["scripted_timeout"].get(args.furniture, task_timeout(args.furniture))
    print(f"Creating env (furniture={args.furniture}, max_steps={rollout_max_steps})")
    np.random.seed(42)
    # Image-based policies were trained on 1280x720 images captured with the wide
    # FOV (69.4°), then manually resized/cropped to 320x240 (see data_collector.py).
    # Setting resize_img=False replicates that setup: the env returns full-res images
    # and preprocess_obs applies the same resize/resize_crop.  State-based policies
    # don't render cameras at all, so resize_img has no effect for them.
    env = gym.make(
        "FurnitureSim-v0",
        furniture=args.furniture,
        max_env_steps=rollout_max_steps,
        headless=args.headless,
        num_envs=args.n_envs,
        obs_keys=env_obs_keys,
        resize_img=False,
        np_step_out=False,
        channel_first=False,
        act_rot_repr="rot_6d",
        action_type="delta",
        ctrl_mode="osc",
        randomness=args.randomness,
        concat_robot_state=True,
    )

    # --- obs diagnostic (only when --dataset-dir is set) ---
    if dataset_init_states is not None and not is_image_based:
        import zarr as _zarr
        _diag_zarr_path = str(cfg.task.dataset.zarr_configs[0].path)
        print(f"\n=== OBS DIAGNOSTIC ===")
        print(f"Training zarr: {_diag_zarr_path}")

        # Print normalizer ranges for obs keys
        for _k in sorted(policy_obs_keys):
            _p = policy.normalizer[_k].params_dict
            print(f"  normalizer[{_k}]: input_min={_p.input_stats.min[:4].tolist()}..., "
                  f"input_max={_p.input_stats.max[:4].tolist()}...")

        # Get first obs from env after reset_to_dataset_states
        _env_obs = reset_to_dataset_states(env, dataset_init_states[:args.n_envs])
        _env_rs = proprioceptive_quat_to_6d_rotation(_env_obs["robot_state"].float().to(device))[0]  # (16,)
        _env_pp = _env_obs["parts_poses"].float().to(device)[0]  # (35,)
        print(f"\n  env robot_state[0][:8]: {_env_rs[:8].tolist()}")
        print(f"  env parts_poses[0][:7]: {_env_pp[:7].tolist()}")

        # Compare to first timestep from the training zarr
        _z = _zarr.open(_diag_zarr_path, "r")
        _zrs = torch.tensor(_z["data/robot_state"][0])   # (16,)
        _zpp = torch.tensor(_z["data/parts_poses"][0])   # (35,)
        print(f"\n  zarr robot_state[0][:8]: {_zrs[:8].tolist()}")
        print(f"  zarr parts_poses[0][:7]: {_zpp[:7].tolist()}")

        _rs_diff = (_env_rs - _zrs).abs().max().item()
        _pp_diff = (_env_pp - _zpp).abs().max().item()
        print(f"\n  max abs diff robot_state: {_rs_diff:.6f}")
        print(f"  max abs diff parts_poses: {_pp_diff:.6f}")

        # Run policy on env obs and print first predicted action
        _obs_deque = collections.deque(
            [{"robot_state": _env_rs.unsqueeze(0), "parts_poses": _env_pp.unsqueeze(0)}] * n_obs_steps,
            maxlen=n_obs_steps,
        )
        with torch.no_grad():
            _pred = policy.predict_action(build_obs_dict(_obs_deque, device), use_DDIM=True)
        print(f"\n  action_pred[0] (t=0): {_pred['action_pred'][0, 0, :].tolist()}")
        print(f"  action[0]      (t=1): {_pred['action'][0, 0, :].tolist()}")
        print(f"=== END DIAGNOSTIC ===\n")

    # --- output directory ---
    if args.output_dir is not None:
        out_dir = Path(args.output_dir)
    else:
        now = datetime.datetime.now()
        out_dir = Path("outputs") / now.strftime("%Y-%m-%d") / now.strftime("%H-%M-%S")
    out_dir.mkdir(parents=True, exist_ok=True)
    videos_dir = out_dir / "videos"
    if args.save_video:
        videos_dir.mkdir(parents=True, exist_ok=True)

    # --- state initialization (fresh or resumed) ---
    n_success = 0
    n_total = 0
    all_trial_records = []  # list of dicts, one per env per round
    resuming = False

    if args.resume:
        pkl_path = out_dir / "results.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                saved = pickle.load(f)
            if saved.get("n_total", 0) > 0:
                n_success = saved["n_success"]
                n_total = saved["n_total"]
                all_trial_records = saved["trials"]
                resuming = True
            else:
                print(f"Found results.pkl at {pkl_path} but no completed trials; starting fresh.")
        else:
            print(f"--resume set but no results.pkl found in {out_dir}; starting fresh.")

    n_rounds = max(1, math.ceil(args.n_rollouts / args.n_envs))
    # Round to start from: skip rounds already covered by resumed state.
    i_start = math.ceil(n_total / args.n_envs)

    csv_path = out_dir / "results.csv"
    csv_fields = ["trial", "result", "reward", "trial_time"]
    summary_path = out_dir / "summary.txt"

    if resuming:
        print(
            f"Resuming: {n_total}/{args.n_rollouts} trials done"
            f" ({n_success} successes, {n_success / n_total:.1%});"
            f" starting from round {i_start + 1}/{n_rounds}"
        )
        csv_file, csv_writer = _repair_csv_and_summary_from_pkl_data(pkl_path)
    else:
        csv_file = open(csv_path, "w", newline="")
        csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
        csv_writer.writeheader()
        csv_file.flush()

    for i in range(i_start, n_rounds):
        t_start = time.time()
        video_budget = args.n_video_trials if args.n_video_trials >= 0 else args.n_rollouts
        if args.record_failures:
            record_this_round = args.save_video
        else:
            record_this_round = args.save_video and (n_total < video_budget)
        # If using dataset initial states, cycle through them round by round.
        round_init_states = None
        if dataset_init_states is not None:
            start = (i * args.n_envs) % len(dataset_init_states)
            round_init_states = [
                dataset_init_states[(start + e) % len(dataset_init_states)] for e in range(args.n_envs)
            ]

        round_result = run_rollout(
            env=env,
            policy=policy,
            n_obs_steps=n_obs_steps,
            rollout_max_steps=rollout_max_steps,
            device=device,
            obs_keys=policy_obs_keys,
            record_video=record_this_round,
            n_action_steps=n_action_steps,
            init_states=round_init_states,
        )
        rollout_time = time.time() - t_start

        # Stop saving trial records once we reach n_rollouts (so we have exactly n_rollouts records)
        for env_idx in range(args.n_envs):
            if n_total >= args.n_rollouts:
                break
            trial_num = n_total + 1
            result_str = round_result["result"][env_idx]
            record = {
                "trial": trial_num,
                "result": result_str,
                "reward": float(round_result["total_reward"][env_idx]),
                "trial_time": int(round_result["steps_per_env"][env_idx]),
            }
            all_trial_records.append(record)
            n_success += int(round_result["success"][env_idx])
            n_total += 1

            csv_writer.writerow(record)
            csv_file.flush()

            if record_this_round:
                save_this_video = False
                if args.record_failures:
                    if result_str != "success":
                        save_this_video = True
                elif trial_num <= video_budget:
                    save_this_video = True
                
                if save_this_video:
                    video_path = videos_dir / f"trial_{trial_num:04d}_{result_str}.mp4"
                    _write_mp4(round_result["frames"][env_idx], video_path)
                    print(f"  Saved video: {video_path.name}")

        # Write text summary file
        _write_summary(n_success, n_total, all_trial_records, summary_path)

        # Write intermediate pickle after every round
        pkl_path = out_dir / "results.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(
                {
                    "trials": all_trial_records,
                    "n_success": n_success,
                    "n_total": n_total,
                    "success_rate": n_success / n_total if n_total > 0 else 0.0,
                    "checkpoint": args.checkpoint,
                    "furniture": args.furniture,
                    "randomness": args.randomness,
                    "n_obs_steps": n_obs_steps,
                    "rollout_max_steps": rollout_max_steps,
                },
                f,
            )

        success_rate = n_success / n_total
        video_tag = "video=on" if record_this_round else "video=off"
        total_time = time.time() - t_start
        print(
            f"Round {i + 1}/{n_rounds} [{video_tag}]: rollout time={rollout_time:.1f}s, total time={total_time:.1f}s, "
            f"result={round_result['result']}  running {n_success}/{n_total} ({success_rate:.1%})"
        )

    csv_file.close()

    final_success_rate = n_success / n_total
    print(f"\nFinal success rate: {n_success}/{n_total} ({final_success_rate:.1%})")
    print(f"Results written to {out_dir}/")

    # IsaacGym's C++ destructors segfault during normal Python shutdown.
    # os._exit bypasses all cleanup and exits immediately with a clean code.
    # Flush stdout/stderr first so buffered print output isn't lost (os._exit
    # skips Python's normal atexit/buffer-flush sequence).
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
