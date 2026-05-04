"""
DART-rollback data collection for action-chunking policies.

At each step T of a noisy (DART) trajectory:
  1. Record obs_T from the current state S_T.
  2. Snapshot S_T (physics + FSM).
  3. Run CHUNK_SIZE clean steps from S_T → action chunk label.
  4. Restore to S_T.
  5. Apply one noisy DART step → S_{T+1}, obs_{T+1}.

Output: pickle per episode with noisy-trajectory observations paired with
coherent clean action chunks.

Usage:
    conda run -n imitation-juicer python scripts/test_state_rollback_v2.py \\
        --furniture one_leg --headless
"""

import argparse
import datetime
import lzma
import os
import pickle
import uuid

import numpy as np

if "DATA_DIR_RAW" not in os.environ:
    os.environ["DATA_DIR_RAW"] = "dataset"

import furniture_bench  # noqa: F401
from isaacgym import gymtorch

import torch  # noqa: F401  # isort: skip

from furniture_bench.utils.scripted_demo_mod import scale_scripted_action

from src.common.files import trajectory_save_dir
from src.data_collection.data_collector import DataCollector
from src.data_processing.utils import resize, resize_crop

# ── Contact-cache flush parameters ───────────────────────────────────────────

# Number of flush-restore + refresh passes before the exact restore.
# Each pass drives a simulate() call with zero part velocities + a slightly
# open gripper so PhysX computes static-equilibrium contact forces instead
# of the stale high-velocity insertion impulses from the clean rollout.
_N_FLUSH = 2
# Metres to nudge each gripper finger outward during flush steps.
# Small enough that the finger stays in contact with the part; large enough
# that the contact compression force is reduced.
_GRIPPER_OPEN_OFFSET = 0.00025

# ── Snapshot / restore ────────────────────────────────────────────────────────


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


def snapshot(raw_env):
    """Clone physics + FSM state into a plain Python dict."""
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
        # env_steps / scripted_timeout are incremented by every env.step() call,
        # including clean-lookahead steps.  Restore them so the timeout counter
        # only counts noisy steps.
        "env_steps": raw_env.env_steps.clone(),
        "scripted_timeout": list(raw_env.scripted_timeout),
        # assembled_set is updated by compute_assemble() inside every env.step().
        # If the clean lookahead completes assembly, all_assembled() would return
        # True on the next noisy step, terminating the episode prematurely.
        "assembled_sets": [set(furn.assembled_set) for furn in raw_env.furnitures],
    }
    parts = [
        [
            {
                "_last_state": p._last_state,
                "_current_speed": dict(p._current_speed),
                "gripper_action": p.gripper_action,
                "pre_assemble_done": p.pre_assemble_done,
                "prev_cnt": p.prev_cnt,
                "curr_cnt": p.curr_cnt,
            }
            for p in furn.parts
        ]
        for furn in raw_env.furnitures
    ]
    return phys, parts


def restore(raw_env, phys, parts, zero_part_velocities: bool = False, gripper_open_offset: float = 0.0):
    """Push a snapshot back into the running simulation.

    zero_part_velocities: zero furniture-part lin/ang velocities in the root
      tensor before the PhysX set call.  Causes simulate() to compute
      static-equilibrium contact forces rather than stale insertion impulses.
    gripper_open_offset: metres to add to each Franka finger DOF position
      (indices 7 and 8) before the set call.  Reduces contact compression
      during flush steps without fully releasing the part.
    Both flags are used together during flush passes; neither is set on the
    final exact restore.
    """
    # DOF state (robot joints + gripper).
    raw_env.dof_states.copy_(phys["dof_states"])
    if gripper_open_offset:
        # Franka finger DOFs: indices 7 (left) and 8 (right) within each env.
        # num_envs=1 here, so these are simply rows 7 and 8 of dof_states.
        raw_env.dof_states[7, 0] += gripper_open_offset
        raw_env.dof_states[8, 0] += gripper_open_offset
    raw_env.isaac_gym.set_dof_state_tensor(raw_env.sim, gymtorch.unwrap_tensor(raw_env.dof_states))

    # Furniture-part root states: root_tensor is stale (only refreshed at init);
    # rb_states is authoritative.  Overwrite root_tensor entries from rb_states
    # before calling the indexed set function.
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
            if zero_part_velocities:
                rt[env_idx, actor_idx, 7:] = 0.0  # zero lin_vel (7:10) and ang_vel (10:13)
    raw_env.isaac_gym.set_actor_root_state_tensor_indexed(
        raw_env.sim,
        gymtorch.unwrap_tensor(raw_env.root_tensor),
        gymtorch.unwrap_tensor(raw_env.part_actor_idxs_all_t),
        len(raw_env.part_actor_idxs_all_t),
    )

    # Derived tensors read by env.step() before the first simulate().
    raw_env.rb_states.copy_(phys["rb_states"])
    raw_env.jacobian.copy_(phys["jacobian"])
    raw_env.mm.copy_(phys["mm"])
    raw_env.last_grasp.copy_(phys["last_grasp"])

    # Step counters — must match the noisy-trajectory count, not the lookahead count.
    raw_env.env_steps.copy_(phys["env_steps"])
    raw_env.scripted_timeout[:] = phys["scripted_timeout"]
    for fi, furn in enumerate(raw_env.furnitures):
        furn.assembled_set = set(phys["assembled_sets"][fi])

    # Controller state.
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

    # FSM state.
    for fi, fsnap in enumerate(parts):
        for pi, ps in enumerate(fsnap):
            p = raw_env.furnitures[fi].parts[pi]
            p._last_state = ps["_last_state"]
            p._current_speed = dict(ps["_current_speed"])
            p.gripper_action = ps["gripper_action"]
            p.pre_assemble_done = ps["pre_assemble_done"]
            p.prev_cnt = ps["prev_cnt"]
            p.curr_cnt = ps["curr_cnt"]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _scale(action, env):
    pos_bounds_m = 0.02 if env.ctrl_mode == "diffik" else 0.025
    ori_bounds_deg = 15 if env.ctrl_mode == "diffik" else 20
    return scale_scripted_action(
        action.detach().cpu().clone(),
        pos_bounds_m=pos_bounds_m,
        ori_bounds_deg=ori_bounds_deg,
        device=env.device,
    )


def _obs_to_numpy(obs, env_idx=0):
    """Extract env_idx slice of a batched GPU obs dict and convert to numpy."""
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


# ── Collection loop ───────────────────────────────────────────────────────────


def collect_episode(env, raw_env, chunk_size: int, verbose: bool = True):
    """
    Run one full episode with DART-rollback action-chunk labelling.

    Returns a dict with:
      observations   : list of per-step obs dicts (from the noisy trajectory)
      action_chunks  : list of (chunk_size, action_dim) arrays — clean labels
      actions        : list of (action_dim,) arrays — recorded clean actions at T
      rewards        : list of floats
      skills         : list of ints
      success        : bool
    """
    obs = env.reset()
    done = torch.zeros(1, dtype=torch.bool)

    observations, action_chunks, actions, rewards, skills = [], [], [], [], []
    step = 0

    while not done.any():
        # ── 1. Record obs_T ────────────────────────────────────────────────
        obs_np = _obs_to_numpy(obs)
        obs_small = {k: obs_np[k] for k in ["color_image1", "color_image2", "robot_state", "parts_poses"]}

        # ── 2. Snapshot S_T ────────────────────────────────────────────────
        phys_snap, part_snap = snapshot(raw_env)

        # ── 3. Clean lookahead: CHUNK_SIZE steps from S_T ─────────────────
        chunk = []
        for _ in range(chunk_size):
            _, ca, _ = env.get_assembly_action()
            env.step(_scale(ca, env))
            chunk.append(ca.detach().cpu().numpy().squeeze())
            if done.any():  # episode ended during lookahead; pad with last action
                while len(chunk) < chunk_size:
                    chunk.append(chunk[-1].copy())
                break

        # ── 4. Restore to S_T with contact-cache flush ────────────────────
        # The clean rollout may drive the leg through insertion contact,
        # leaving large stale impulses in the PhysX warm-start cache.
        # Each flush pass: set correct part positions with zero velocities
        # and the gripper fingers nudged slightly open, then call refresh()
        # so PhysX runs simulate() and computes static-equilibrium contact
        # forces (gripper lightly holding a stationary part) rather than
        # the stale high-velocity insertion impulses.  Zero velocities
        # prevent the part from drifting during the flush simulate(); the
        # small gripper retraction lowers compression without releasing it.
        # After all flush passes, the exact restore puts everything back.
        for _ in range(_N_FLUSH):
            restore(raw_env, phys_snap, part_snap, zero_part_velocities=True, gripper_open_offset=_GRIPPER_OPEN_OFFSET)
            raw_env.refresh()
        restore(raw_env, phys_snap, part_snap)

        # ── 5. Apply noisy DART step from S_T → obs_{T+1} ─────────────────
        noisy_action, clean_action, skill_complete = env.get_assembly_action()
        obs, rew, done, info = env.step(_scale(noisy_action, env))

        # Record clean action at T (what the policy would cleanly do from S_T).
        rec_action = _scale(clean_action, env).detach().cpu().numpy().squeeze()
        rew_val = float(rew[0].squeeze().cpu())
        skill_val = int(skill_complete[0]) if isinstance(skill_complete, (list, tuple)) else int(skill_complete)

        observations.append(obs_small)
        action_chunks.append(np.stack(chunk))
        actions.append(rec_action)
        rewards.append(rew_val)
        skills.append(skill_val)
        step += 1

        if verbose and step % 50 == 0:
            print(f"  step {step:4d}  reward_sum={sum(rewards):.1f}")

    # Store terminal obs.
    obs_np = _obs_to_numpy(obs)
    observations.append({k: obs_np[k] for k in ["color_image1", "color_image2", "robot_state", "parts_poses"]})

    success = raw_env.furnitures[0].all_assembled()
    if verbose:
        print(f"  Episode done: {step} steps, success={success}, reward_sum={sum(rewards):.1f}")

    return {
        "observations": observations,
        "action_chunks": action_chunks,
        "actions": actions,
        "rewards": rewards,
        "skills": skills,
        "success": success,
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def _count_success(data_path) -> int:
    """Count existing success .pkl / .pkl.xz files on disk."""
    d = data_path / "success"
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix in (".pkl", ".xz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--furniture", "-f", type=str, required=True)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--num-demos", "-n", type=int, default=400)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--dart-amount", type=float, default=1.0)
    parser.add_argument("--save-failure", action="store_true")
    args = parser.parse_args()

    data_path = trajectory_save_dir(
        environment="sim",
        task=args.furniture,
        demo_source="scripted_chunk",
        randomness="low",
    )

    process_seed = uuid.uuid4().int & 0x7FFFFFFF
    print(f"[seed] process_seed={process_seed}  (log this to reproduce any episode)")

    collector = DataCollector(
        is_sim=True,
        data_path=data_path,
        furniture=args.furniture,
        device_interface=None,
        headless=args.headless,
        manual_label=False,
        scripted=True,
        draw_marker=True,
        randomness="low",
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        resize_sim_img=False,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        ctrl_mode="osc",
        compress_pickles=True,
        no_noise=False,
        dart_amount=args.dart_amount,
        num_envs=1,
        seed=process_seed,
    )

    env = collector.env
    raw_env = collector.env.unwrapped

    target = args.num_demos
    episode_idx = 0
    n_fail = 0

    while _count_success(data_path) < target:
        on_disk = _count_success(data_path)
        print(
            f"\n[episode={episode_idx}  on-disk={on_disk}/{target}] chunk_size={args.chunk_size}  dart={args.dart_amount}"
        )

        episode_seed = (process_seed + episode_idx) % (2**31)
        np.random.seed(episode_seed)

        data = collect_episode(env, raw_env, chunk_size=args.chunk_size)
        data["furniture"] = args.furniture
        data["chunk_size"] = args.chunk_size
        data["process_seed"] = process_seed
        data["episode_idx"] = episode_idx
        episode_idx += 1

        if data["success"]:
            # Re-check before writing; another process may have filled the quota.
            if _count_success(data_path) >= target:
                print("  Target reached by another process — discarding episode.")
                break
        else:
            n_fail += 1
            if not args.save_failure:
                print(f"  Failure — skipping save (on-disk={_count_success(data_path)}/{target})")
                continue

        subdir = "success" if data["success"] else "failure"
        out_dir = data_path / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        out_path = out_dir / f"{ts}.pkl.xz"
        with lzma.open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved → {out_path}")
        print(f"  Progress: on-disk={_count_success(data_path)}/{target}, this-process failures={n_fail}")

    print(
        f"\nDone. process_seed={process_seed}, episodes_run={episode_idx}, on-disk={_count_success(data_path)}/{target}."
    )


if __name__ == "__main__":
    main()
