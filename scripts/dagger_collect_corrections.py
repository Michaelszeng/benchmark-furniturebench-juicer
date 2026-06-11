"""
DAgger Step 3: restore each labeled failure to its gate state and collect a
scripted-expert correction.

For each failure pkl under `dagger_iter{N}/{randomness}/failure/` that has a
gate appended (written as an additional xz stream by dagger_label_gates.py):
  1. env.reset()
  2. take one priming step so OSC controllers are instantiated (the snapshot
     captured at the gate has populated osc_ctrls and restore() expects them
     to exist on the destination env).
  3. restore(snapshots[gate_idx]) so the env state matches the policy's state
     at the gate timestep.
  4. zero the step counters so the expert gets a fresh max_env_steps budget.
  5. roll out the scripted expert until success or timeout, recording obs /
     actions / rewards / skills in the exact format produced by DataCollector.save()
     so the resulting pickle is consumed unmodified by process_pickles.py.
  6. on failure, retry up to --max-retries times with a different seed.

Output (only successful corrections) goes to:
    $DATA_DIR_RAW/raw/sim/{furniture}/dagger_iter{N}/{randomness}/correction/success/

i.e., a sibling of the iter's `failure/` dir, so every artifact from iter N
lives under `dagger_iterN/`. To aggregate corrections across iterations for
training, pass multiple sources (or a glob) to process_pickles:

    python -m src.data_processing.process_pickles \\
        --env sim --furniture one_leg --randomness low --demo-outcome success \\
        --source 'dagger_iter*/correction' --output-name dagger_combined

Usage:
    python scripts/dagger_collect_corrections.py --furniture one_leg --iter 0 --headless
"""

from __future__ import annotations

import argparse
import datetime
import lzma
import os
import pickle
import re
import sys
import uuid
from pathlib import Path

# Matches the "NN__" numeric prefix produced by dagger_collect_failures.py's
# _claim_failure_number; re-used here so a correction pkl carries the same
# prefix as the failure pkl it came from (for at-a-glance association).
_FAILURE_NUMBER_PREFIX_RE = re.compile(r"^(\d+)__")

if "DATA_DIR_RAW" not in os.environ:
    os.environ["DATA_DIR_RAW"] = "dataset"

import furniture_bench  # noqa: F401  must come before isaacgym / torch
from isaacgym import gymtorch  # noqa: F401

import torch  # noqa: F401  # isort: skip

import gym
import numpy as np
from furniture_bench.sim_config import sim_config
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action

from src.common.files import trajectory_save_dir
from src.common.geometry import np_action_6d_to_quat
from src.data_processing.utils import resize, resize_crop
from src.visualization.render_mp4 import pickle_data

# ── Restore ───────────────────────────────────────────────────────────────────


def _to_device_deep(v, device):
    """Recursively move CPU tensors back to `device`, preserving nested structure.

    Inverse of `_to_cpu_deep` in dagger_collect_failures.py.
    """
    if isinstance(v, torch.Tensor):
        return v.to(device)
    if isinstance(v, dict):
        return {k: _to_device_deep(vv, device) for k, vv in v.items()}
    if isinstance(v, list):
        return [_to_device_deep(vv, device) for vv in v]
    if isinstance(v, tuple):
        return tuple(_to_device_deep(vv, device) for vv in v)
    return v


def _restore_part(part, snap: dict, device) -> None:
    """Reflective inverse of `_snap_part`: write every snapshotted field back.

    For tensors of matching shape, copies in-place to preserve object identity
    (some env code may hold references to the original tensor); otherwise
    setattr's the new value.
    """
    for k, v in snap.items():
        moved = _to_device_deep(v, device)
        existing = getattr(part, k, None)
        if isinstance(existing, torch.Tensor) and isinstance(moved, torch.Tensor) and existing.shape == moved.shape:
            existing.copy_(moved)
        else:
            setattr(part, k, moved)


def _restore_osc(ctrl, s):
    ctrl.ee_pos_desired.data.copy_(s["ee_pos_desired"])
    ctrl.ee_quat_desired.data.copy_(s["ee_quat_desired"])
    ctrl.goal_pos.copy_(s["goal_pos"])
    ctrl.prev_goal_pos.copy_(s["prev_goal_pos"])
    ctrl.step_num_pos = s["step_num_pos"]
    ctrl.goal_ori.copy_(s["goal_ori"])
    ctrl.prev_goal_ori.copy_(s["prev_goal_ori"])
    ctrl.step_num_ori = s["step_num_ori"]
    ctrl.prev_interp_pos.copy_(s["prev_interp_pos"])
    ctrl.prev_interp_ori.copy_(s["prev_interp_ori"])
    ctrl.fraction = s["fraction"]
    ctrl.repeated_torques_counter = s["repeated_torques_counter"]
    ctrl.prev_torques.copy_(s["prev_torques"])


def restore(raw_env, phys, parts, zero_velocities: bool = False, gripper_open_offset: float = 0.0):
    """Push a snapshot back into the running simulation.

    See scripts/test_state_rollback_v2.py for the original commentary on
    zero_velocities / gripper_open_offset (used during contact-cache flush
    passes there; the DAgger pipeline never sets either flag).
    """
    raw_env.dof_states.copy_(phys["dof_states"])
    if zero_velocities:
        raw_env.dof_states[:, 1] = 0.0
    if gripper_open_offset:
        raw_env.dof_states[7, 0] += gripper_open_offset
        raw_env.dof_states[8, 0] += gripper_open_offset
    raw_env.isaac_gym.set_dof_state_tensor(raw_env.sim, gymtorch.unwrap_tensor(raw_env.dof_states))

    raw_env.root_tensor.copy_(phys["root_tensor"])
    rt = raw_env.root_tensor.view(raw_env.num_envs, -1, 13)
    for name, rb_idxs in raw_env.part_idxs.items():
        if name.startswith("obstacle"):
            continue
        actor_idx = raw_env.parts_handles.get(name)
        if actor_idx is None:
            continue
        for env_idx, rb_idx in enumerate(rb_idxs):
            rt[env_idx, actor_idx] = phys["rb_states"][rb_idx]
            if zero_velocities:
                rt[env_idx, actor_idx, 7:] = 0.0
    raw_env.isaac_gym.set_actor_root_state_tensor_indexed(
        raw_env.sim,
        gymtorch.unwrap_tensor(raw_env.root_tensor),
        gymtorch.unwrap_tensor(raw_env.part_actor_idxs_all_t),
        len(raw_env.part_actor_idxs_all_t),
    )

    raw_env.rb_states.copy_(phys["rb_states"])
    raw_env.jacobian.copy_(phys["jacobian"])
    raw_env.mm.copy_(phys["mm"])
    raw_env.last_grasp.copy_(phys["last_grasp"])

    raw_env.env_steps.copy_(phys["env_steps"])
    raw_env.scripted_timeout[:] = phys["scripted_timeout"]
    for fi, furn in enumerate(raw_env.furnitures):
        furn.assembled_set = set(phys["assembled_sets"][fi])

    if phys["ctrl_started"]:
        if phys["last_torque_action"] is not None:
            raw_env.last_torque_action = phys["last_torque_action"].clone()
            raw_env.isaac_gym.set_dof_actuation_force_tensor(
                raw_env.sim, gymtorch.unwrap_tensor(raw_env.last_torque_action)
            )
        for ctrl, s in zip(raw_env.osc_ctrls, phys["osc_ctrls"]):
            _restore_osc(ctrl, s)
    else:
        raw_env.ctrl_started = False
        raw_env.osc_ctrls.clear()
        raw_env.diffik_ctrls.clear()
        raw_env.last_torque_action = None
        raw_env.isaac_gym.set_dof_actuation_force_tensor(
            raw_env.sim, gymtorch.unwrap_tensor(torch.zeros_like(raw_env.dof_pos))
        )

    # Reflective per-part restore — handles all NM state (counter cycles,
    # latent offsets, sticky countdowns, `_NM_LATENT_PLAN`, etc.) plus the
    # originally-captured FSM fields. New fields auto-flow through.
    for fi, fsnap in enumerate(parts):
        for pi, ps in enumerate(fsnap):
            _restore_part(raw_env.furnitures[fi].parts[pi], ps, raw_env.device)

    # Env-level NM state: any phys key starting with `_nm_` was captured
    # reflectively in snapshot() and is restored here by the same convention.
    # Future env-level NM fields auto-flow through; no per-field maintenance.
    for k, v in phys.items():
        if k.startswith("_nm_") and hasattr(raw_env, k):
            setattr(raw_env, k, _to_device_deep(v, raw_env.device))


def _obs_to_numpy(obs, env_idx: int = 0) -> dict:
    """Match DataCollector._obs_to_numpy: image resize/crop + per-env slice."""
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


def _preview_for_pkl(pkl_path: Path) -> Path:
    name = pkl_path.name
    if name.endswith(".pkl.xz"):
        stem = name[: -len(".pkl.xz")]
    elif name.endswith(".pkl"):
        stem = name[: -len(".pkl")]
    else:
        stem = pkl_path.stem
    return pkl_path.parent / f"{stem}.preview.mp4"


def _has_gate_mtime(pkl_path: Path) -> bool:
    """Cheap pre-filter: pkl mtime > preview mtime ⇒ gate has been appended.

    Falls back to True (so we attempt the full load) if the preview is missing,
    which happens if the user deleted previews after labeling.
    """
    preview = _preview_for_pkl(pkl_path)
    if not preview.exists():
        return True
    return pkl_path.stat().st_mtime > preview.stat().st_mtime


def _load_pkl_with_gate(pkl_path: Path):
    """Load a failure pkl that may have appended gate xz streams.

    Returns (data: dict, gate_idx: int | None, overrides: dict). If multiple
    gate streams have been appended (e.g. via --relabel), the most recent one
    wins. lzma.open transparently handles multi-stream xz, so we just
    pickle.load until EOF.
    """
    objects = []
    with lzma.open(pkl_path, "rb") as f:
        while True:
            try:
                objects.append(pickle.load(f))
            except EOFError:
                break
    if not objects:
        raise ValueError(f"no pickle objects in {pkl_path}")
    data = objects[0]
    gate_idx = None
    overrides: dict = {}
    for obj in objects[1:]:
        if isinstance(obj, dict) and "gate_idx" in obj:
            gate_idx = int(obj["gate_idx"])
            overrides = obj.get("overrides") or {}
    return data, gate_idx, overrides


def _apply_overrides(raw_env, overrides: dict) -> None:
    """Apply per-gate overrides written by dagger_label_gates.py.

    Currently supports:
      - force_pre_assemble_done: list of part indices → set part.pre_assemble_done=True.
      - force_state: {part_idx: state_name} → set part._last_state.

    By design (per user) we do NOT clear NM cycle counters, align_offset, or
    any other part-internal cache: the overridden state is presumed not-yet-
    reached, so its state-specific counters should already be at initial
    defaults from the snapshot. If that assumption fails (e.g. user overrides
    to a state that *was* visited earlier), choose a different override state
    or extend this function.
    """
    if not overrides:
        return
    parts = raw_env.furnitures[0].parts
    for pi in overrides.get("force_pre_assemble_done", []):
        parts[pi].pre_assemble_done = True
        print(f"    override: parts[{pi}].pre_assemble_done = True")
    for pi, new_state in (overrides.get("force_state") or {}).items():
        old = getattr(parts[pi], "_last_state", None)
        parts[pi]._last_state = new_state
        print(f"    override: parts[{pi}]._last_state = {new_state!r} (was {old!r})")


def _snapshot_to_device(phys_cpu, parts, device):
    """Move stored CPU tensors back onto `device` for restore()."""
    phys = {}
    for k, v in phys_cpu.items():
        if isinstance(v, torch.Tensor):
            phys[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], dict):
            phys[k] = [{kk: (vv.to(device) if isinstance(vv, torch.Tensor) else vv) for kk, vv in d.items()} for d in v]
        else:
            phys[k] = v
    return phys, parts


def _reset_step_counters(raw_env):
    """After restore, give the expert a fresh max_env_steps budget."""
    raw_env.env_steps.zero_()
    if isinstance(raw_env.scripted_timeout, list):
        for i in range(len(raw_env.scripted_timeout)):
            raw_env.scripted_timeout[i] = 0
    else:
        try:
            raw_env.scripted_timeout.zero_()
        except Exception:
            pass


def _scale_for_env(action, raw_env):
    pos_bounds_m = 0.02 if raw_env.ctrl_mode == "diffik" else 0.025
    ori_bounds_deg = 15 if raw_env.ctrl_mode == "diffik" else 20
    return scale_scripted_action(
        action.detach().cpu().clone(),
        pos_bounds_m=pos_bounds_m,
        ori_bounds_deg=ori_bounds_deg,
        device=raw_env.device,
    )


def run_expert_from_state(env, raw_env, phys_cpu, parts_cpu, max_steps: int, retry_seed: int, overrides: dict | None = None):
    """Restore to the snapshot, then roll out the scripted expert to terminal.

    Returns (success: bool, episode_data: dict). episode_data is ALWAYS returned
    (even on failure / exception) so callers can save failed attempts for
    debugging. The dict's "success" / "error" fields reflect the actual outcome.
    Length invariant holds in all paths: len(observations) == len(actions) + 1.
    """
    np.random.seed(retry_seed)

    # Full reset, then a priming step so the env lazily instantiates osc_ctrls.
    # snapshot()'s ctrl-state restore relies on `for ctrl, s in zip(raw_env.osc_ctrls, ...)`,
    # which is a no-op when osc_ctrls is empty — so without this step the gate-time
    # controller state would be silently dropped.
    env.reset()
    try:
        _, prime_clean, _ = env.get_assembly_action()
        env.step(_scale_for_env(prime_clean, raw_env))
    except Exception as e:
        # Not fatal — restore() will overwrite the state regardless.
        print(f"    priming step warning: {e}")

    phys, parts = _snapshot_to_device(phys_cpu, parts_cpu, raw_env.device)
    restore(raw_env, phys, parts)
    # Overrides must be applied AFTER restore() (which overwrites _last_state /
    # pre_assemble_done with the snapshot values) and BEFORE the first FSM tick.
    _apply_overrides(raw_env, overrides or {})
    _reset_step_counters(raw_env)
    raw_env.refresh()
    obs = raw_env._get_observation()

    buf_obs: list = []
    buf_acts: list = []
    buf_rews: list = []
    buf_skills: list = []
    step = 0
    done = False
    error_msg = ""

    while not done and step < max_steps:
        try:
            noisy_action, clean_action, skill_complete = env.get_assembly_action()
        except Exception as e:
            error_msg = f"get_assembly_action raised error at step {step}: {e}"
            print(f"    {error_msg}")
            break

        action_to_apply = _scale_for_env(noisy_action, raw_env)
        record_action = _scale_for_env(clean_action, raw_env)

        try:
            next_obs, rew, done_t, info = env.step(action_to_apply)
        except RuntimeError as e:
            error_msg = f"env.step RuntimeError at step {step}: {e}"
            print(f"    {error_msg}")
            break

        if not info.get("action_success", True):
            obs = next_obs
            done = bool(done_t.any())
            continue

        # Store (obs_t, clean_action_t, reward_{t+1}, skill_complete_t) — matches
        # the per-step layout in DataCollector._store_step.
        obs_np = _obs_to_numpy(obs)
        buf_obs.append({k: obs_np[k] for k in ["color_image1", "color_image2", "robot_state", "parts_poses"]})

        ra = record_action[0]
        if isinstance(ra, torch.Tensor):
            ra = ra.detach().cpu().numpy()
        else:
            ra = np.asarray(ra)
        buf_acts.append(ra)
        buf_rews.append(float(rew[0].squeeze().cpu()))
        sk = skill_complete[0] if isinstance(skill_complete, (list, tuple)) else skill_complete
        buf_skills.append(int(sk))

        obs = next_obs
        done = bool(done_t.any())
        step += 1

    # Terminal obs — always appended so the length invariant holds even on
    # early exit (exception path) or premature timeout. DataCollector does the
    # same after its loop.
    obs_np = _obs_to_numpy(obs)
    buf_obs.append({k: obs_np[k] for k in ["color_image1", "color_image2", "robot_state", "parts_poses"]})

    success = bool(raw_env.furnitures[0].all_assembled())
    return success, {
        "observations": buf_obs,
        "actions": buf_acts,
        "rewards": buf_rews,
        "skills": buf_skills,
        "success": success,
        "error": bool(error_msg),
        "error_description": error_msg,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--furniture", "-f", required=True)
    parser.add_argument("--randomness", "-r", default="low")
    parser.add_argument("--iter", type=int, required=True, help="DAgger iteration whose failures we correct")
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--non-markovian",
        action="store_true",
        help="Build the env with non_markovian=True and apply the NM latent plan per episode "
        "(must match the setting used during failure collection — same flag passed to "
        "dagger_collect_failures.py).",
    )
    parser.add_argument(
        "--instance-idx",
        type=int,
        default=0,
        help="0-indexed instance number for parallel runs. Combined with --total-instances, "
        "selects this instance's slice of labeled failures via candidates[idx::total].",
    )
    parser.add_argument("--total-instances", type=int, default=1)
    parser.add_argument(
        "--action-horizon",
        type=int,
        required=True,
        help="Action horizon used during failure collection (used only to locate the iter dir).",
    )
    parser.add_argument(
        "--dart-amount",
        type=float,
        default=0.0,
        help="DART action-noise scale passed to the env. 0.0 = no noise.",
    )
    parser.add_argument(
        "--debug-full-rollout",
        action="store_true",
        help="DEBUG: prepend the entire policy rollout to the corrections so that the recorded .pkl contains the "
        "entire episode (not just the correction). Used only for debugging; saves to `correction_debug/success/`.",
    )
    parser.add_argument(
        "--save-failed-attempts",
        action="store_true",
        help="DEBUG: also save every failed expert attempt to `correction[_debug]/failure/` with a "
        "`_retry{N}_` segment in the filename. Useful for visualizing what the expert tried before "
        "eventually succeeding (or giving up after --max-retries).",
    )
    args = parser.parse_args()

    process_seed = args.seed if args.seed is not None else (uuid.uuid4().int & 0x7FFFFFFF)
    print(f"[seed] process_seed={process_seed}")

    # demo_source MUST match what dagger_collect_failures.py wrote: same (iter, ah, nm) tuple.
    nm_tag = "nm" if args.non_markovian else "m"
    demo_source = f"dagger_iter{args.iter}_ah{args.action_horizon}_{nm_tag}"
    print(f"demo_source={demo_source}")

    # Discover labeled failures.
    failure_dir = (
        trajectory_save_dir(
            environment="sim",
            task=args.furniture,
            demo_source=demo_source,
            randomness=args.randomness,
            create=False,
        )
        / "failure"
    )
    print(f"Failure dir: {failure_dir}")
    if not failure_dir.exists():
        print("  (does not exist)")
        return

    all_pkls = sorted(failure_dir.glob("*.pkl.xz")) + sorted(failure_dir.glob("*.pkl"))
    candidates = [fp for fp in all_pkls if _has_gate_mtime(fp)]
    print(f"Found {len(all_pkls)} failure pkls; {len(candidates)} appear labeled (will verify on load).")
    if args.total_instances > 1:
        my_slice = candidates[args.instance_idx :: args.total_instances]
        print(
            f"Parallel mode: instance {args.instance_idx}/{args.total_instances} "
            f"handles {len(my_slice)} of {len(candidates)} labeled pkls (stride {args.total_instances})."
        )
        candidates = my_slice
    if not candidates:
        return

    # Corrections live as a sibling of the failure dir: dagger_iterN/{rand}/correction/success/.
    iter_dir = trajectory_save_dir(
        environment="sim",
        task=args.furniture,
        demo_source=demo_source,
        randomness=args.randomness,
    )
    correction_subdir = "correction_debug" if args.debug_full_rollout else "correction"
    success_dir = iter_dir / correction_subdir / "success"
    success_dir.mkdir(parents=True, exist_ok=True)
    print(f"Correction dir: {success_dir}  (mode={'B-debug' if args.debug_full_rollout else 'B-prime'})")

    if args.non_markovian:
        rollout_max_steps = sim_config["nm_scripted_timeout"].get(
            args.furniture, sim_config["scripted_timeout"].get(args.furniture, 3000)
        )
    else:
        rollout_max_steps = sim_config["scripted_timeout"].get(args.furniture, 3000)
    print(f"max_steps per expert rollout: {rollout_max_steps}, non_markovian={args.non_markovian}")

    # Same env construction as DataCollector (is_sim=True, scripted=True): no
    # act_rot_repr override means env.step accepts the 8-D quat actions that
    # scale_scripted_action emits and that process_pickles.py expects.
    env = gym.make(
        "FurnitureSimFull-v0",
        furniture=args.furniture,
        max_env_steps=rollout_max_steps,
        headless=args.headless,
        num_envs=1,
        manual_done=False,
        resize_img=False,
        np_step_out=False,
        channel_first=False,
        randomness=args.randomness,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        ctrl_mode="osc",
        non_markovian=args.non_markovian,
        dart_amount=args.dart_amount,
    )
    raw_env = env.unwrapped

    n_saved = 0
    n_unrecoverable = 0

    for pi, fp in enumerate(candidates):
        print(f"\n[{pi + 1}/{len(candidates)}] {fp.name}")
        fail_data, gate_idx, overrides = _load_pkl_with_gate(fp)
        if gate_idx is None:
            print("  no gate stream found in pkl; skipping (mtime heuristic was wrong)")
            continue
        print(f"  gate_idx={gate_idx}" + (f"  overrides={overrides}" if overrides else ""))

        snaps = fail_data["snapshots"]
        if gate_idx >= len(snaps):
            print(f"  gate_idx {gate_idx} >= n_snapshots {len(snaps)}; skipping")
            n_unrecoverable += 1
            continue
        phys_cpu, parts_cpu = snaps[gate_idx]

        # Mirror the source failure pkl's "NN__" prefix on outputs so each
        # correction (or failed attempt) is visually associated with its origin.
        m = _FAILURE_NUMBER_PREFIX_RE.match(fp.name)
        prefix = f"{m.group(1)}__" if m else ""

        saved = False
        for retry in range(args.max_retries):
            retry_seed = (process_seed + pi * 1000 + retry) % (2**31)
            print(f"  retry {retry + 1}/{args.max_retries} (seed={retry_seed})")
            ok, ep = run_expert_from_state(
                env,
                raw_env,
                phys_cpu,
                parts_cpu,
                rollout_max_steps,
                retry_seed,
                overrides=overrides,
            )

            # ── Prepend pre-gate policy data so the gate window has the right history ──
            # Applied to BOTH success and failure paths so failed attempts (saved when
            # --save-failed-attempts is on) have the same format/visualization as
            # successful corrections.
            assert gate_idx >= 1, (
                f"gate_idx={gate_idx} must be >= 1; dagger_label_gates.py should have rejected this at label time."
            )
            if args.debug_full_rollout:
                # Pre-pend the entire policy rollout to the corrections.
                # obs[0..gate_idx] from policy, obs[gate_idx+1..] from expert.
                # actions[0..gate_idx-1] from policy, actions[gate_idx..] from expert.
                pre_obs_list = list(fail_data["observations"][: gate_idx + 1])  # o_0..o_n
                pre_acts_10d = np.asarray(fail_data["actions"][:gate_idx], dtype=np.float32)
                pre_acts_8d = list(np_action_6d_to_quat(pre_acts_10d))
                pre_rewards = [float(r) for r in fail_data["rewards"][:gate_idx]]
                ep["observations"] = pre_obs_list + ep["observations"][1:]
                ep["actions"] = pre_acts_8d + ep["actions"]
                ep["rewards"] = pre_rewards + ep["rewards"]
                ep["skills"] = [0] * gate_idx + ep["skills"]
                ep["dagger_pre_gate_obs_included"] = gate_idx + 1
                ep["dagger_debug_full_rollout"] = True
            else:
                # Pre-pend single (o_{n-1}, o_n) so gate window's history matches policy view.
                pre_obs = fail_data["observations"][gate_idx - 1]
                gate_obs = fail_data["observations"][gate_idx]
                pre_act_10d = np.asarray(fail_data["actions"][gate_idx - 1], dtype=np.float32)
                pre_act_8d = np_action_6d_to_quat(pre_act_10d[np.newaxis, :])[0]
                pre_reward = float(fail_data["rewards"][gate_idx - 1])
                ep["observations"] = [pre_obs, gate_obs] + ep["observations"][1:]
                ep["actions"] = [pre_act_8d] + ep["actions"]
                ep["rewards"] = [pre_reward] + ep["rewards"]
                ep["skills"] = [0] + ep["skills"]
                ep["dagger_pre_gate_obs_included"] = 2
            ep["furniture"] = args.furniture
            ep["dagger_origin"] = fp.name
            ep["dagger_gate_idx"] = gate_idx
            ep["dagger_iter"] = args.iter
            if overrides:
                ep["dagger_overrides"] = overrides

            ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")

            if ok:
                out_path = success_dir / f"{prefix}{ts}_pid{os.getpid()}.pkl.xz"
                pickle_data(ep, out_path)
                print(f"  saved -> {out_path.name}   (length {len(ep['actions'])})")
                n_saved += 1
                saved = True
                break
            elif args.save_failed_attempts:
                failed_dir = iter_dir / correction_subdir / "failure"
                failed_dir.mkdir(parents=True, exist_ok=True)
                ep["dagger_retry"] = retry
                failed_path = failed_dir / f"{prefix}{ts}_retry{retry}_pid{os.getpid()}.pkl.xz"
                pickle_data(ep, failed_path)
                print(f"    saved failed attempt -> {failed_path.name}   (length {len(ep['actions'])}); retrying")
            else:
                print("    expert did not succeed from gate; retrying")

        if not saved:
            print(f"  unrecoverable after {args.max_retries} retries")
            n_unrecoverable += 1

    print(f"\nDone. Saved {n_saved} corrections; {n_unrecoverable} unrecoverable.")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
