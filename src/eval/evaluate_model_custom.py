"""Evaluate a diffusion_policy checkpoint on FurnitureBench simulation.

Uses FurnitureSimFull-v0, which mirrors the data-collection environment:
full observations (cameras + all state keys), dict robot_state, act_rot_repr=rot_6d.

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
from furniture_bench.sim_config import sim_config
from omegaconf import OmegaConf

from src.common.geometry import proprioceptive_quat_to_6d_rotation
from src.common.tasks import task_timeout
from src.data_processing.utils import resize, resize_crop
from src.visualization.render_mp4 import unpickle_data

# Register the "eval" resolver used in diffusion_policy configs.
OmegaConf.register_new_resolver("eval", eval, replace=True)

# Keys used to reconstruct the 14-D proprioceptive vector from the full
# robot_state dict that FurnitureSimFull-v0 returns.  Order must match
# filter_and_concat_robot_state / ROBOT_STATES in furniture_bench.
_ROBOT_STATE_KEYS = ["ee_pos", "ee_quat", "ee_pos_vel", "ee_ori_vel", "gripper_width"]

# Per-dimension label for one part's 7-D pose block [x y z qx qy qz qw].
_POSE_DIM_NAMES = ["x", "y", "z", "qx", "qy", "qz", "qw"]


def quat_xyzw_to_6d(quat: torch.Tensor) -> torch.Tensor:
    """
    Use custom function to convert quaternion to 6D rotation to match training conversion function.
    """
    qx, qy, qz, qw = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    col0 = torch.stack(
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy + qz * qw),
            2 * (qx * qz - qy * qw),
        ],
        dim=-1,
    )
    col1 = torch.stack(
        [
            2 * (qx * qy - qz * qw),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz + qx * qw),
        ],
        dim=-1,
    )
    return torch.cat([col0, col1], dim=-1)


def preprocess_obs(obs: dict, device: torch.device, obs_keys: set) -> dict:
    """
    Handles both image-based policies (color_image1, color_image2, robot_state)
    and state-based policies (robot_state, parts_poses).

    FurnitureSimFull-v0 returns robot_state as a dict; this function selects
    and concatenates the canonical 5 keys to form a 14-D quat vector, then
    converts to 16-D rot-6d.

    Args:
        obs: raw observation dict from env.step / env.reset.
        device: torch device to move tensors to.
        obs_keys: set of observation key names the policy expects
                  (from cfg.shape_meta.obs).
    """
    rs = obs["robot_state"]
    if isinstance(rs, dict):
        rs = torch.cat([rs[k] for k in _ROBOT_STATE_KEYS], dim=-1)
    result = {"robot_state": proprioceptive_quat_to_6d_rotation(rs.float().to(device))}
    if "color_image1" in obs_keys:
        # resize() matches data_collector.py: 1280x720 → 320x240, same FOV.
        result["color_image1"] = resize(obs["color_image1"]).float().to(device)
    if "color_image2" in obs_keys:
        # resize_crop() matches data_collector.py: 1280x720 → 320x240 center-crop.
        result["color_image2"] = resize_crop(obs["color_image2"]).float().to(device)
    if "parts_poses" in obs_keys:
        parts_poses = obs["parts_poses"].float().to(device)
        shape = parts_poses.shape
        n_parts = shape[-1] // 7
        parts_poses = parts_poses.view(*shape[:-1], n_parts, 7)
        pos = parts_poses[..., :3]
        quat = parts_poses[..., 3:]
        rot_6d = quat_xyzw_to_6d(quat)
        parts_poses_6d = torch.cat([pos, rot_6d], dim=-1)
        result["parts_poses"] = parts_poses_6d.view(*shape[:-1], n_parts * 9)
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


def load_reference_pkl_parts_poses(pkl_path: str) -> np.ndarray:
    """Load parts_poses from all timesteps of a reference pkl file.

    Returns array of shape (T, 35).
    """
    data = unpickle_data(Path(pkl_path))
    obs = data["observations"]
    parts_poses = np.array([o["parts_poses"] for o in obs], dtype=np.float32)
    # parts_poses may be stored with shape (35,) or (1, 35) per timestep; flatten
    if parts_poses.ndim == 3:
        parts_poses = parts_poses.squeeze(1)
    return parts_poses  # (T, 35)


def _plot_parts_poses_traces(
    rollout_traces: list,
    reference_traces,
) -> None:
    """Display a live parts-poses trace plot.

    Args:
        rollout_traces: list of (T_i, 35) np.ndarray, one per rollout round.
                        Each row is one timestep's flattened parts_poses.
        reference_traces: (T_ref, 35) np.ndarray, or None.
                          Full-episode parts_poses from the reference pkl.
    """
    import matplotlib.pyplot as plt

    n_parts = 5
    # 10 distinguishable colours, one per pose dimension (7 used).
    colors = [
        "#e6194b",
        "#3cb44b",
        "#4363d8",
        "#f58231",
        "#911eb4",
        "#42d4f4",
        "#f032e6",
    ]

    fig, axes = plt.subplots(n_parts, 1, figsize=(14, 4 * n_parts), squeeze=False)
    axes = axes[:, 0]  # (n_parts,)

    for part_idx in range(n_parts):
        ax = axes[part_idx]
        base = part_idx * 7

        # --- rollout traces (solid, slightly transparent when multiple rounds) ---
        alpha = max(0.35, 1.0 - 0.15 * len(rollout_traces))
        for round_idx, trace in enumerate(rollout_traces):
            for dim_idx, dim_name in enumerate(_POSE_DIM_NAMES):
                ax.plot(
                    trace[:, base + dim_idx],
                    color=colors[dim_idx],
                    alpha=alpha,
                    linewidth=1.4,
                    label=f"rollout {dim_name}" if round_idx == 0 else None,
                )

        # --- reference pkl traces (dashed, full opacity) ---
        if reference_traces is not None:
            for dim_idx, dim_name in enumerate(_POSE_DIM_NAMES):
                ax.plot(
                    reference_traces[:, base + dim_idx],
                    color=colors[dim_idx],
                    alpha=1.0,
                    linewidth=2.0,
                    linestyle="--",
                    label=f"pkl {dim_name}",
                )

        ax.set_title(f"Part {part_idx} poses over time", fontsize=11)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.legend(
            loc="upper right",
            fontsize=7,
            ncol=2,
            framealpha=0.7,
        )
        ax.grid(True, alpha=0.25)

    fig.suptitle(
        f"Parts poses: {len(rollout_traces)} rollout(s)"
        + (" vs reference pkl" if reference_traces is not None else ""),
        fontsize=13,
        y=1.002,
    )
    plt.tight_layout()
    plt.show()


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
) -> dict:
    """Run one round of parallel rollouts.

    Returns a dict with per-env arrays:
        success           (n_envs,) bool
        total_reward      (n_envs,) float
        result            (n_envs,) str — "success", "timeout", or "failure"
        steps             int — total steps executed this round
        steps_per_env     (n_envs,) int
        parts_poses_trace (T+1, 35) float32 — env-0 parts_poses at every timestep
                          (initial obs + one entry per env.step call); may be
                          shorter than rollout_max_steps if interrupted mid-rollout
        interrupted       bool — True if a KeyboardInterrupt cut the rollout short
        frames            (n_envs,) list[np.ndarray] — only present if record_video=True;
                          each element is a list of (H, W*2, 3) uint8 frames (cam1 | cam2)
    """
    n_envs = env.num_envs

    # Initialise accumulators before env.reset() so a Ctrl+C during reset still
    # produces a valid (possibly empty) return value.
    parts_poses_trace: list = []
    interrupted = False

    done = torch.zeros((n_envs, 1), dtype=torch.bool, device=device)
    total_reward = torch.zeros(n_envs, device=device)
    done_step = torch.full((n_envs,), -1, dtype=torch.long, device=device)
    step = 0
    if record_video:
        frame_buffers = [[] for _ in range(n_envs)]

    try:
        obs = env.reset()
        parts_poses_trace.append(obs["parts_poses"][0].cpu().numpy())  # initial obs

        preprocessed = preprocess_obs(obs, device, obs_keys)
        obs_deque = collections.deque([preprocessed] * n_obs_steps, maxlen=n_obs_steps)
        action_queue: collections.deque = collections.deque()

        while not done.all() and step < rollout_max_steps:
            if len(action_queue) == 0:
                obs_dict = build_obs_dict(obs_deque, device)
                result = policy.predict_action(obs_dict, use_DDIM=True)
                start = n_obs_steps - 1
                actions = result["action_pred"][:, start:]
                n_steps = n_action_steps if n_action_steps is not None else policy.n_action_steps
                for t in range(n_steps):
                    action_queue.append(actions[:, t, :])

            action = action_queue.popleft()
            obs, reward, done, _ = env.step(action)
            total_reward += reward.squeeze(-1).float()

            newly_done = done.squeeze(-1) & (done_step == -1)
            done_step[newly_done] = step

            preprocessed = preprocess_obs(obs, device, obs_keys)

            if record_video:
                if "color_image1" in preprocessed:
                    imgs1 = preprocessed["color_image1"].cpu().numpy().astype(np.uint8)
                    imgs2 = preprocessed["color_image2"].cpu().numpy().astype(np.uint8)
                else:
                    imgs1 = resize(obs["color_image1"]).cpu().numpy().astype(np.uint8)
                    imgs2 = resize_crop(obs["color_image2"]).cpu().numpy().astype(np.uint8)
                for env_idx in range(n_envs):
                    side_by_side = np.concatenate([imgs1[env_idx], imgs2[env_idx]], axis=1)
                    frame_buffers[env_idx].append(side_by_side)

            parts_poses_trace.append(obs["parts_poses"][0].cpu().numpy())
            obs_deque.append(preprocessed)
            step += 1

    except KeyboardInterrupt:
        interrupted = True

    n_parts = len(env.furniture.should_be_assembled)
    success_mask = (total_reward >= n_parts).cpu().numpy()
    done_step_np = done_step.cpu().numpy()

    results = []
    for env_idx in range(n_envs):
        if success_mask[env_idx]:
            results.append("success")
        elif done_step_np[env_idx] == -1:
            results.append("timeout")
        else:
            results.append("failure")

    steps_per_env = np.where(done_step_np >= 0, done_step_np + 1, step)

    out = {
        "success": success_mask,
        "total_reward": total_reward.cpu().numpy(),
        "result": results,
        "steps": step,
        "steps_per_env": steps_per_env,
        "parts_poses_trace": np.stack(parts_poses_trace) if parts_poses_trace else np.empty((0, 35), dtype=np.float32),
        "interrupted": interrupted,
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
        "--reference-pkl",
        type=str,
        default=None,
        help="Path to a reference pkl file whose parts_poses are overlaid on the trace plot.",
    )
    args = parser.parse_args()

    if args.resume and args.output_dir is None:
        parser.error("--resume requires --output-dir to be specified")

    # Load reference pkl parts_poses if provided.
    reference_parts_poses = None
    if args.reference_pkl is not None:
        print(f"Loading reference pkl parts_poses from {args.reference_pkl}")
        reference_parts_poses = load_reference_pkl_parts_poses(args.reference_pkl)
        print(f"  Reference trajectory length: {len(reference_parts_poses)} steps")

    device = torch.device(args.device)

    print(f"Loading policy from {args.checkpoint}")
    policy, cfg = load_policy(args.checkpoint, device)
    n_obs_steps: int = int(cfg.n_obs_steps)
    n_action_steps: int = args.n_action_steps  # None means use the value from the checkpoint config
    print(
        f"n_obs_steps={n_obs_steps}, n_action_steps={'from_cfg' if n_action_steps is None else n_action_steps}, furniture={args.furniture}, n_envs={args.n_envs}"
    )

    # Detect policy modality from shape_meta.
    policy_obs_keys = set(cfg.shape_meta.obs.keys())
    is_image_based = "color_image1" in policy_obs_keys
    print(f"Policy type: {'image-based' if is_image_based else 'state-based'} (obs keys: {sorted(policy_obs_keys)})")

    # Use the same timeout that scripted data collection uses (sim_config["scripted_timeout"]),
    # falling back to task_timeout for tasks not listed there.
    rollout_max_steps = sim_config["scripted_timeout"].get(args.furniture, task_timeout(args.furniture))
    print(f"Creating env (furniture={args.furniture}, max_steps={rollout_max_steps})")
    np.random.seed(42)
    # FurnitureSimFull-v0 mirrors the data-collection environment:
    # - obs_keys=FULL_OBS (cameras + all state keys), robot_state returned as dict
    # - act_rot_repr="rot_6d" so env.step accepts the policy's 10-D actions directly
    # - resize_img=False: env returns full-res 1280x720 images; preprocess_obs applies
    #   the same resize/resize_crop as data_collector.py.
    env = gym.make(
        "FurnitureSimFull-v0",
        furniture=args.furniture,
        max_env_steps=rollout_max_steps,
        headless=args.headless,
        num_envs=args.n_envs,
        resize_img=False,
        np_step_out=False,
        channel_first=False,
        act_rot_repr="rot_6d",
        action_type="delta",
        ctrl_mode="osc",
        randomness=args.randomness,
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

    # --- state initialization (fresh or resumed) ---
    n_success = 0
    n_total = 0
    all_trial_records = []  # list of dicts, one per env per round
    all_parts_poses_traces = []  # (T+1, 35) array per completed round, env 0 only
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

        round_result = run_rollout(
            env=env,
            policy=policy,
            n_obs_steps=n_obs_steps,
            rollout_max_steps=rollout_max_steps,
            device=device,
            obs_keys=policy_obs_keys,
            record_video=record_this_round,
            n_action_steps=n_action_steps,
        )
        rollout_time = time.time() - t_start

        # Always collect the trace, even if the rollout was cut short.
        if len(round_result["parts_poses_trace"]) > 0:
            all_parts_poses_traces.append(round_result["parts_poses_trace"])

        if round_result["interrupted"]:
            print("\nInterrupted — generating parts-poses trace plot…")
            break

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
        if n_total > 0:
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

        if n_total > 0:
            success_rate = n_success / n_total
            video_tag = "video=on" if record_this_round else "video=off"
            total_time = time.time() - t_start
            print(
                f"Round {i + 1}/{n_rounds} [{video_tag}]: rollout time={rollout_time:.1f}s, total time={total_time:.1f}s, "
                f"result={round_result['result']}  running {n_success}/{n_total} ({success_rate:.1%})"
            )

    csv_file.close()

    if n_total > 0:
        final_success_rate = n_success / n_total
        print(f"\nFinal success rate: {n_success}/{n_total} ({final_success_rate:.1%})")
    print(f"Results written to {out_dir}/")

    # --- parts-poses trace plot ---
    if all_parts_poses_traces:
        _plot_parts_poses_traces(all_parts_poses_traces, reference_parts_poses)
    else:
        print("No rollout data collected; skipping trace plot.")

    # IsaacGym's C++ destructors segfault during normal Python shutdown.
    # os._exit bypasses all cleanup and exits immediately with a clean code.
    # Flush stdout/stderr first so buffered print output isn't lost (os._exit
    # skips Python's normal atexit/buffer-flush sequence).
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
