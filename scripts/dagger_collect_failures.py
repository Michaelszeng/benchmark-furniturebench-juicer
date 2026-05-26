"""
DAgger Step 1: roll out a policy and collect N failed episodes with per-step snapshots.

Loads a diffusion_policy .ckpt via the same path as evaluate_model_custom.py,
rolls out on FurnitureSimFull-v0, and on each failed episode saves a pickle
containing:
  - observations  : list of T+1 dicts {color_image1, color_image2, robot_state, parts_poses}
                    (same shape as data_collector pickles; includes terminal obs)
  - actions       : list of T policy actions (10-D rot_6d delta, as fed to env.step)
  - rewards       : list of T floats
  - snapshots     : list of T (phys_cpu, parts) tuples — physics + FSM state at the
                    start of step i (i.e., the state from which actions[i] was applied)
  - success       : False
  - furniture, iter, checkpoint, process_seed, rollout_seed, runtime_err

Output goes to $DATA_DIR_RAW/raw/sim/{furniture}/dagger_iter{N}/{randomness}/failure/.
The gate-labeling script (Step 2) writes a sibling <stem>.gate.json next to each pkl.

Multi-instance: --num-failures is a shared target across all parallel runs.
Each process polls the disk and stops as soon as the combined count of completed
(.pkl.xz) and in-progress (.pkl.xz.tmp) files reaches the target — same pattern
data_collector.py uses for scripted collection.

Usage:
    python scripts/dagger_collect_failures.py 
        --checkpoint path/to/model.ckpt --furniture one_leg --iter 0 \
        --num-failures 20 --headless
"""

import argparse
import collections
import copy
import datetime
import os
import re
import sys
import uuid

if "DATA_DIR_RAW" not in os.environ:
    os.environ["DATA_DIR_RAW"] = "dataset"

import furniture_bench  # noqa: F401  must come before isaacgym / torch
from isaacgym import gymtorch  # noqa: F401

import torch  # noqa: F401  # isort: skip

import gym
import numpy as np
from furniture_bench.sim_config import sim_config

from src.common.files import trajectory_save_dir
from src.common.tasks import task_timeout
from src.data_processing.utils import resize, resize_crop
from src.eval.evaluate_model_custom import build_obs_dict, load_policy, preprocess_obs
from src.visualization.render_mp4 import pickle_data


def _count_failures_on_disk(failure_dir) -> int:
    """Count completed (.pkl.xz, .pkl) and in-progress (.tmp) failure pickles.

    Mirrors data_collector.py:_count_existing_demos: counts .tmp files too so a
    slot that another parallel job is currently writing is treated as occupied.
    """
    if not failure_dir.exists():
        return 0
    return sum(1 for p in failure_dir.iterdir() if p.suffix in (".pkl", ".xz", ".tmp"))


_NUMBERED_PREFIX_RE = re.compile(r"^(\d+)__")


def _claim_failure_number(failure_dir, tail: str):
    """Atomically claim the smallest unused 'N__' number for a new failure pkl.

    Scans the dir for existing '<N>__...' filenames (pkls, previews, .tmp's, all
    count) and tries to create '<N>__<tail>.tmp' via O_CREAT|O_EXCL starting at
    the smallest unused N. Returns (n, final_path) — pickle_data() will then
    write to <final_path>.tmp (the very file we just claimed, overwriting its
    empty contents) and rename to <final_path> on success.

    The O_EXCL claim is what makes this safe for parallel collectors: if two
    jobs pick the same N simultaneously, only one wins the create() call.
    """
    used = set()
    for p in failure_dir.iterdir():
        m = _NUMBERED_PREFIX_RE.match(p.name)
        if m:
            used.add(int(m.group(1)))
    n = 1
    while True:
        while n in used:
            n += 1
        # 2-digit zero-pad so the dir lex-sorts numerically up to 99 failures.
        final_path = failure_dir / f"{n:02d}__{tail}"
        tmp_path = final_path.with_suffix(final_path.suffix + ".tmp")
        try:
            fd = os.open(str(tmp_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            os.close(fd)
            return n, final_path
        except FileExistsError:
            used.add(n)
            n += 1


# ── Snapshot ──────────────────────────────────────────────────────────────────


def _to_cpu_deep(v):
    """Recursively clone tensors to CPU; deepcopy other Python containers/scalars.

    Preserves nested structure (dicts/lists/tuples). All resulting tensors are
    detached and on CPU so the snapshot pickles cheaply and portably.
    """
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().clone()
    if isinstance(v, dict):
        return {k: _to_cpu_deep(vv) for k, vv in v.items()}
    if isinstance(v, list):
        return [_to_cpu_deep(vv) for vv in v]
    if isinstance(v, tuple):
        return tuple(_to_cpu_deep(vv) for vv in v)
    return copy.deepcopy(v)


def _snap_part(part) -> dict:
    """Reflectively snapshot every field in `part.__dict__` via `_to_cpu_deep`."""
    out = {}
    for k, v in part.__dict__.items():
        try:
            out[k] = _to_cpu_deep(v)
        except Exception as e:
            print(f"  warning: _snap_part skipping {type(part).__name__}.{k}: {e}")
    return out


def _snap_osc(ctrl):
    return {
        "ee_pos_desired": ctrl.ee_pos_desired.data.clone(),
        "ee_quat_desired": ctrl.ee_quat_desired.data.clone(),
        "goal_pos": ctrl.goal_pos.clone(),
        "prev_goal_pos": ctrl.prev_goal_pos.clone(),
        "step_num_pos": ctrl.step_num_pos,
        "goal_ori": ctrl.goal_ori.clone(),
        "prev_goal_ori": ctrl.prev_goal_ori.clone(),
        "step_num_ori": ctrl.step_num_ori,
        "prev_interp_pos": ctrl.prev_interp_pos.clone(),
        "prev_interp_ori": ctrl.prev_interp_ori.clone(),
        "fraction": ctrl.fraction,
        "repeated_torques_counter": ctrl.repeated_torques_counter,
        "prev_torques": ctrl.prev_torques.clone(),
    }


def snapshot(raw_env):
    """Clone physics + FSM state into a plain Python dict.

    Per-part FSM state is captured reflectively via `_snap_part` so all NM
    counters / latent-plan fields / etc. round-trip without having to be
    enumerated here.
    """
    phys = {
        "dof_states": raw_env.dof_states.clone(),
        "rb_states": raw_env.rb_states.clone(),
        "root_tensor": raw_env.root_tensor.clone(),
        "jacobian": raw_env.jacobian.clone(),
        "mm": raw_env.mm.clone(),
        "last_grasp": raw_env.last_grasp.clone(),
        "ctrl_started": raw_env.ctrl_started,
        "last_torque_action": (raw_env.last_torque_action.clone() if raw_env.last_torque_action is not None else None),
        "osc_ctrls": [_snap_osc(c) for c in raw_env.osc_ctrls],
        "env_steps": raw_env.env_steps.clone(),
        "scripted_timeout": list(raw_env.scripted_timeout),
        "assembled_sets": [set(furn.assembled_set) for furn in raw_env.furnitures],
    }
    # Reflectively capture all env-level non-Markovian state by `_nm_*` prefix.
    # Future env-level NM fields are picked up automatically; matches the
    # restore-side discovery in dagger_collect_corrections.py:restore().
    for k, v in vars(raw_env).items():
        if k.startswith("_nm_"):
            phys[k] = _to_cpu_deep(v)
    parts = [[_snap_part(p) for p in furn.parts] for furn in raw_env.furnitures]
    return phys, parts


def _obs_to_numpy_pkl(obs, env_idx: int = 0) -> dict:
    """Mirror DataCollector._obs_to_numpy: extract env_idx and convert to numpy.

    Images are resized/cropped to 240x320 to match the data_collector pickle format.
    """
    out = {}
    for k, v in obs.items():
        if isinstance(v, dict):
            out[k] = {k2: v2[env_idx].cpu().numpy() for k2, v2 in v.items()}
        elif k == "color_image1":
            out[k] = resize(v[env_idx : env_idx + 1]).squeeze().cpu().numpy()
        elif k == "color_image2":
            out[k] = resize_crop(v[env_idx : env_idx + 1]).squeeze().cpu().numpy()
        else:
            out[k] = v[env_idx].cpu().numpy()
    return out


def _snapshot_to_cpu(phys, parts):
    """Move every tensor in a snapshot to CPU so it can be pickled cheaply."""
    phys_cpu = {}
    for k, v in phys.items():
        if isinstance(v, torch.Tensor):
            phys_cpu[k] = v.detach().cpu()
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            phys_cpu[k] = [
                {kk: (vv.detach().cpu() if isinstance(vv, torch.Tensor) else vv) for kk, vv in d.items()} for d in v
            ]
        else:
            phys_cpu[k] = v
    return phys_cpu, parts


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to diffusion_policy .ckpt")
    parser.add_argument("--furniture", "-f", required=True)
    parser.add_argument("--randomness", "-r", default="low")
    parser.add_argument("--iter", type=int, required=True, help="DAgger iteration index (used in output dir name)")
    parser.add_argument("--num-failures", "-n", type=int, default=20)
    parser.add_argument(
        "--max-rollouts",
        type=int,
        default=None,
        help="Safety cap on total rollouts. Default: 20 * num_failures.",
    )
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="Override action horizon (default: read from ckpt config).",
    )
    parser.add_argument(
        "--rollout-max-steps",
        type=int,
        default=None,
        help="Default: sim_config['scripted_timeout'][furniture].",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--non-markovian",
        action="store_true",
        help="Build the env with non_markovian=True and apply the NM latent plan per episode. "
        "Use this when the policy was trained on non-Markovian scripted data so the FSM the "
        "FSM-tick observes matches what produced the training data.",
    )
    parser.add_argument(
        "--dart-amount",
        type=float,
        default=0.0,
        help="DART action-noise scale passed to the env (matches scripted.py). 0.0 = no noise.",
    )
    args = parser.parse_args()

    process_seed = args.seed if args.seed is not None else (uuid.uuid4().int & 0x7FFFFFFF)
    print(f"[seed] process_seed={process_seed}")

    device = torch.device(f"cuda:{args.gpu_id}")

    print(f"Loading policy from {args.checkpoint}")
    policy, cfg = load_policy(args.checkpoint, device)
    n_obs_steps = int(cfg.n_obs_steps)
    policy_obs_keys = set(cfg.shape_meta.obs.keys())
    print(f"Policy n_obs_steps={n_obs_steps}, obs_keys={sorted(policy_obs_keys)}")

    if args.rollout_max_steps:
        rollout_max_steps = args.rollout_max_steps
    elif args.non_markovian:
        rollout_max_steps = sim_config["nm_scripted_timeout"].get(
            args.furniture, sim_config["scripted_timeout"].get(args.furniture, task_timeout(args.furniture))
        )
    else:
        rollout_max_steps = sim_config["scripted_timeout"].get(args.furniture, task_timeout(args.furniture))
    max_rollouts = args.max_rollouts or (20 * args.num_failures)
    print(f"rollout_max_steps={rollout_max_steps}, max_rollouts={max_rollouts}, non_markovian={args.non_markovian}")

    # demo_source encodes the (iter, action_horizon, nm) tuple so different
    # rollout configs for the same iter land in distinct dirs.
    nm_tag = "nm" if args.non_markovian else "m"
    ah_for_path = args.n_action_steps if args.n_action_steps is not None else "default"
    demo_source = f"dagger_iter{args.iter}_ah{ah_for_path}_{nm_tag}"
    data_path = trajectory_save_dir(
        environment="sim",
        task=args.furniture,
        demo_source=demo_source,
        randomness=args.randomness,
    )
    failure_dir = data_path / "failure"
    failure_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {failure_dir}")

    prior_on_disk = _count_failures_on_disk(failure_dir)
    if prior_on_disk:
        print(
            f"Found {prior_on_disk} existing failure pickles on disk (counted toward shared quota of {args.num_failures})."
        )

    # Same env construction as evaluate_model_custom.py: act_rot_repr=rot_6d so the
    # 10-D diffusion_policy action vector is consumed directly by env.step.
    env = gym.make(
        "FurnitureSimFull-v0",
        furniture=args.furniture,
        max_env_steps=rollout_max_steps,
        headless=args.headless,
        num_envs=1,
        resize_img=False,
        np_step_out=False,
        channel_first=False,
        act_rot_repr="rot_6d",
        action_type="delta",
        ctrl_mode="osc",
        randomness=args.randomness,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        non_markovian=args.non_markovian,
        dart_amount=args.dart_amount,
    )
    raw_env = env.unwrapped

    def _configure_nm_episode():
        """Apply NM per-episode config (samples latent plan / persistent offsets).

        Mirrors DataCollector._configure_episode. Must be called after every
        env.reset() in NM mode so part.apply_non_markovian_config() resamples
        the per-episode latent variables (offsets, sticky countdowns, etc.).
        """
        if not args.non_markovian:
            return
        for part in raw_env.furnitures[0].parts:
            part.max_len_offset = part._NM_MAX_PAUSE
            if hasattr(part, "apply_non_markovian_config"):
                part.apply_non_markovian_config()

    n_failures = 0  # contributed by THIS process; the shared total is read from disk
    n_rollouts = 0

    while _count_failures_on_disk(failure_dir) < args.num_failures and n_rollouts < max_rollouts:
        rollout_seed = (process_seed + n_rollouts) % (2**31)
        np.random.seed(rollout_seed)
        torch.manual_seed(rollout_seed)

        obs = env.reset()
        _configure_nm_episode()  # no-op when --non-markovian not set
        preprocessed = preprocess_obs(obs, device, policy_obs_keys)
        obs_deque = collections.deque([preprocessed] * n_obs_steps, maxlen=n_obs_steps)
        action_queue: collections.deque = collections.deque()

        observations: list = []
        actions: list = []
        rewards: list = []
        snapshots: list = []

        step = 0
        done_flag = False
        runtime_err = False

        while not done_flag and step < rollout_max_steps:
            # Tick the scripted-expert FSM once per step (reacts to current physics
            # via env.get_assembly_action()). We discard the expert's action —
            # env.step() below uses the policy's action — but this keeps per-part
            # FSM fields like `pre_assemble_done`, `_last_state`, etc. in sync
            # with reality so the snapshot captures meaningful FSM state.
            # Without this call, dagger_collect_corrections.py would restore the
            # FSM in its post-reset "nothing has happened" state and the expert
            # would start from scratch (e.g., re-running pre-assembly from frame 0
            # even when restored at gate frame 232).
            try:
                env.get_assembly_action()
            except Exception as e:
                print(f"  warning: get_assembly_action() at step {step}: {e}")

            # The expert's FSM `satisfy()` timeouts (e.g. "EE didn't reach goal in
            # max_len steps") latch raw_env.scripted_timeout[env_idx] = True, which
            # furniture_sim_env.step() then forces into done=True. That's the right
            # behavior when the expert is driving; here the expert is only observing,
            # so its timeout is meaningless and would prematurely end the rollout
            # (and label it as failure regardless of what the policy was doing).
            # Clear the flag so only the policy's true termination signals stop us.
            for _i in range(raw_env.num_envs):
                raw_env.scripted_timeout[_i] = False

            # Snapshot BEFORE env.step → snapshots[i] is the state from which actions[i] runs
            # (and after the FSM tick above, so the FSM reflects the expert's view at obs[i]).
            phys, parts = snapshot(raw_env)
            snapshots.append(_snapshot_to_cpu(phys, parts))

            obs_np = _obs_to_numpy_pkl(obs)
            observations.append({k: obs_np[k] for k in ["color_image1", "color_image2", "robot_state", "parts_poses"]})

            if len(action_queue) == 0:
                with torch.no_grad():
                    obs_dict = build_obs_dict(obs_deque, device)
                    result = policy.predict_action(obs_dict, use_DDIM=True)
                start = n_obs_steps - 1
                actions_pred = result["action_pred"][:, start:]
                n_act_steps = args.n_action_steps if args.n_action_steps is not None else policy.n_action_steps
                for t in range(n_act_steps):
                    action_queue.append(actions_pred[:, t, :])

            action = action_queue.popleft()
            try:
                obs, reward, done, _ = env.step(action)
            except RuntimeError as e:
                print(f"  env.step RuntimeError at step {step}: {e}")
                runtime_err = True
                break

            actions.append(action.detach().cpu().numpy().squeeze(0))
            rewards.append(float(reward.squeeze().cpu()))
            preprocessed = preprocess_obs(obs, device, policy_obs_keys)
            obs_deque.append(preprocessed)

            done_flag = bool(done.any())
            step += 1

        # Terminal obs (T+1-th observation; no terminal snapshot).
        obs_np = _obs_to_numpy_pkl(obs)
        observations.append({k: obs_np[k] for k in ["color_image1", "color_image2", "robot_state", "parts_poses"]})

        success = bool(raw_env.furnitures[0].all_assembled())
        n_rollouts += 1
        tag = "SUCCESS" if success else ("RUNTIME_ERR" if runtime_err else "FAILURE")
        print(f"[rollout {n_rollouts}] {tag} after {step} steps")

        if success:
            continue  # discard successes; we only save failures

        # Re-check shared quota right before writing to minimise the race window
        # where multiple parallel jobs finish near-simultaneously.
        on_disk = _count_failures_on_disk(failure_dir)
        if on_disk >= args.num_failures:
            print(f"  shared quota reached ({on_disk}/{args.num_failures}) — discarding this failure.")
            break

        n_failures += 1
        data = {
            "observations": observations,
            "actions": actions,
            "rewards": rewards,
            "snapshots": snapshots,
            "success": False,
            "furniture": args.furniture,
            "iter": args.iter,
            "checkpoint": str(args.checkpoint),
            "process_seed": process_seed,
            "rollout_seed": rollout_seed,
            "runtime_err": runtime_err,
        }
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        tail = f"{ts}_pid{os.getpid()}.pkl.xz"
        n, out_path = _claim_failure_number(failure_dir, tail)  # writes empty <N>__<tail>.tmp
        # pickle_data writes to <out_path>.tmp (the file we just claimed,
        # which it opens in 'wb' mode and truncates) and renames to <out_path>
        # on success — safe for parallel writers; labeler never sees partial.
        pickle_data(data, out_path)
        print(
            f"  saved -> {out_path.name}   "
            f"(claimed #{n}; this job: {n_failures}; on disk: {_count_failures_on_disk(failure_dir)}/{args.num_failures})"
        )

    final_on_disk = _count_failures_on_disk(failure_dir)
    print(
        f"\nDone. This job contributed {n_failures} failures in {n_rollouts} rollouts; "
        f"shared total on disk: {final_on_disk}/{args.num_failures}."
    )

    # IsaacGym destructors segfault during normal Python shutdown.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
