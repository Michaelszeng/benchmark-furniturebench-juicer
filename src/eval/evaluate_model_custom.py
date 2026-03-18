"""Evaluate a diffusion_policy checkpoint on FurnitureBench simulation.

Usage:
    python src/eval/evaluate_model_custom.py \
        --checkpoint /path/to/checkpoint.ckpt \
        --furniture one_leg \
        --n-rollouts 10 \
        --n-envs 1
"""

import argparse
import collections
from pathlib import Path

import dill
import furniture_bench  # noqa: F401 — registers FurnitureSim envs
import gym
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
    """Convert (B, H, W, 3) uint8 → (B, 3, 224, 224) float [0, 1].

    Applies to both color_image1 and color_image2.
    """
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


def load_policy(checkpoint_path: str, device: torch.device, normalizer_override: str = None):
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
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    policy = workspace.ema_model if cfg.training.use_ema else workspace.model

    # --- normalizer ---
    ckpt_path = Path(checkpoint_path)
    normalizer_path = ckpt_path.parent.parent / "normalizer.pt"
    if normalizer_override is not None:
        print(f"Loading normalizer from {normalizer_override}")
        normalizer = torch.load(normalizer_override)
    elif normalizer_path.exists():
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


@torch.no_grad()
def run_rollout(
    env,
    policy,
    n_obs_steps: int,
    rollout_max_steps: int,
    device: torch.device,
) -> np.ndarray:
    """Run one round of parallel rollouts.  Returns bool array (n_envs,)."""
    obs = env.reset()
    preprocessed = preprocess_obs(obs, device)
    obs_deque = collections.deque([preprocessed] * n_obs_steps, maxlen=n_obs_steps)
    action_queue: collections.deque = collections.deque()

    done = torch.zeros((env.num_envs, 1), dtype=torch.bool, device="cuda")
    total_reward = torch.zeros(env.num_envs, device="cuda")
    step = 0

    while not done.all() and step < rollout_max_steps:
        if len(action_queue) == 0:
            obs_dict = build_obs_dict(obs_deque, device)
            result = policy.predict_action(obs_dict, use_DDIM=True)
            # result["action"]: (n_envs, n_action_steps, action_dim)
            actions = result["action"]
            for t in range(actions.shape[1]):
                action_queue.append(actions[:, t, :])  # (n_envs, action_dim)

        action = action_queue.popleft()  # (n_envs, action_dim)
        obs, reward, done, _ = env.step(action)
        total_reward += reward.squeeze(-1).float()
        preprocessed = preprocess_obs(obs, device)
        obs_deque.append(preprocessed)
        step += 1

    n_parts = len(env.furniture.should_be_assembled)
    success = (total_reward >= n_parts).cpu().numpy()
    return success


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
    parser.add_argument("--normalizer", type=str, default=None, help="Path to normalizer.pt (overrides auto-discovery)")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--headless", action="store_true", default=False)
    parser.add_argument("--no-headless", dest="headless", action="store_false")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading policy from {args.checkpoint}")
    policy, cfg = load_policy(args.checkpoint, device, normalizer_override=args.normalizer)
    n_obs_steps: int = int(cfg.n_obs_steps)
    print(f"n_obs_steps={n_obs_steps}, furniture={args.furniture}, n_envs={args.n_envs}")

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

    n_success = 0
    n_total = 0
    n_rounds = max(1, args.n_rollouts // args.n_envs)

    for i in range(n_rounds):
        success = run_rollout(
            env=env,
            policy=policy,
            n_obs_steps=n_obs_steps,
            rollout_max_steps=rollout_max_steps,
            device=device,
        )
        n_success += int(success.sum())
        n_total += args.n_envs
        rate = n_success / n_total
        print(f"Round {i + 1}/{n_rounds}: success={success.tolist()}  running {n_success}/{n_total} ({rate:.1%})")

    final_rate = n_success / n_total
    print(f"\nFinal success rate: {n_success}/{n_total} ({final_rate:.1%})")
