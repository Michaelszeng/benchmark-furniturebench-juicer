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
import torchvision.transforms.functional as TF
from omegaconf import OmegaConf
from torchvision.transforms import InterpolationMode

from src.common.geometry import proprioceptive_quat_to_6d_rotation
from src.common.tasks import task_timeout

# Register the "eval" resolver used in diffusion_policy configs.
OmegaConf.register_new_resolver("eval", eval, replace=True)


def preprocess_images(imgs: torch.Tensor) -> torch.Tensor:
    """Convert (B, H, W, 3) uint8 → (B, 3, 224, 224) float [0, 1]."""
    imgs = imgs.float() / 255.0
    imgs = imgs.permute(0, 3, 1, 2)  # (B, H, W, 3) -> (B, 3, H, W)
    imgs = TF.resize(imgs, [224, 224], interpolation=InterpolationMode.BILINEAR, antialias=True)
    return imgs


def preprocess_obs(obs: dict, device: torch.device) -> dict:
    """Transform one env step obs into the format expected by the policy.

    Returns dict with keys color_image1, color_image2, robot_state — all on
    ``device`` with float dtype.
    """
    img1 = preprocess_images(obs["color_image1"]).to(device)
    img2 = preprocess_images(obs["color_image2"]).to(device)
    robot_state = proprioceptive_quat_to_6d_rotation(obs["robot_state"].float().to(device))
    return {"color_image1": img1, "color_image2": img2, "robot_state": robot_state}


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


@torch.no_grad()
def run_rollout(
    env,
    policy,
    n_obs_steps: int,
    rollout_max_steps: int,
    device: torch.device,
    record_video: bool = False,
    n_action_steps: int = None,
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
    obs = env.reset()
    preprocessed = preprocess_obs(obs, device)
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

        preprocessed = preprocess_obs(obs, device)
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
        "--n-action-steps",
        type=int,
        default=None,
        help="Override action horizon (default: use value from checkpoint config)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory to write results (default: outputs/<date>/<time>)"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading policy from {args.checkpoint}")
    policy, cfg = load_policy(args.checkpoint, device)
    n_obs_steps: int = int(cfg.n_obs_steps)
    n_action_steps: int = args.n_action_steps  # None means use the value from the checkpoint config
    print(
        f"n_obs_steps={n_obs_steps}, n_action_steps={'from_cfg' if n_action_steps is None else n_action_steps}, furniture={args.furniture}, n_envs={args.n_envs}"
    )

    rollout_max_steps = task_timeout(args.furniture)
    print(f"Creating env (furniture={args.furniture}, max_steps={rollout_max_steps})")
    env = gym.make(
        "FurnitureSim-v0",
        furniture=args.furniture,
        max_env_steps=rollout_max_steps,
        headless=args.headless,
        num_envs=args.n_envs,
        np_step_out=False,
        channel_first=False,
        act_rot_repr="rot_6d",
        action_type="delta",
        ctrl_mode="osc",
        randomness=args.randomness,
        concat_robot_state=True,
    )

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

    n_success = 0
    n_total = 0
    n_rounds = max(1, math.ceil(args.n_rollouts / args.n_envs))
    all_trial_records = []  # list of dicts, one per env per round

    csv_path = out_dir / "results.csv"
    csv_fields = ["trial", "result", "reward", "trial_time"]
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(csv_file, fieldnames=csv_fields)
    csv_writer.writeheader()
    csv_file.flush()

    summary_path = out_dir / "summary.txt"

    for i in range(n_rounds):
        t_start = time.time()
        video_budget = args.n_video_trials if args.n_video_trials >= 0 else args.n_rollouts
        record_this_round = args.save_video and (n_total < video_budget)
        round_result = run_rollout(
            env=env,
            policy=policy,
            n_obs_steps=n_obs_steps,
            rollout_max_steps=rollout_max_steps,
            device=device,
            record_video=record_this_round,
            n_action_steps=n_action_steps,
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
            _write_summary(n_success, n_total, all_trial_records, summary_path)

            if record_this_round and trial_num <= video_budget:
                video_path = videos_dir / f"trial_{trial_num:04d}_{result_str}.mp4"
                _write_mp4(round_result["frames"][env_idx], video_path)
                print(f"  Saved video: {video_path.name}")

        rate = n_success / n_total
        video_tag = "video=on" if record_this_round else "video=off"
        total_time = time.time() - t_start
        print(
            f"Round {i + 1}/{n_rounds} [{video_tag}]: rollout time={rollout_time:.1f}s, total time={total_time:.1f}s, result={round_result['result']}  running {n_success}/{n_total} ({rate:.1%})"
        )

    csv_file.close()

    final_rate = n_success / n_total
    print(f"\nFinal success rate: {n_success}/{n_total} ({final_rate:.1%})")

    # --- write pickle ---
    pkl_path = out_dir / "results.pkl"
    with open(pkl_path, "wb") as f:
        pickle.dump(
            {
                "trials": all_trial_records,
                "n_success": n_success,
                "n_total": n_total,
                "success_rate": final_rate,
                "checkpoint": args.checkpoint,
                "furniture": args.furniture,
                "randomness": args.randomness,
                "n_obs_steps": n_obs_steps,
                "rollout_max_steps": rollout_max_steps,
            },
            f,
        )

    print(f"Results written to {out_dir}/")

    # IsaacGym's C++ destructors segfault during normal Python shutdown.
    # os._exit bypasses all cleanup and exits immediately with a clean code.
    # Flush stdout/stderr first so buffered print output isn't lost (os._exit
    # skips Python's normal atexit/buffer-flush sequence).
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
