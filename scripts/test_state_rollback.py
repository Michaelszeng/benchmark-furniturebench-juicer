"""
Test physics-state save/restore for IsaacGym simulation rollback.

At each checkpoint step, saves full physics state (DOF positions/velocities +
actor root positions/orientations/velocities), runs `--lookahead` clean steps,
restores the snapshot, re-runs the same clean steps, and checks that the two
trajectories match.  A matching result confirms that cloning dof_states +
root_tensor and pushing them back via set_dof_state_tensor /
set_actor_root_state_tensor_indexed faithfully rolls back the simulation — including
mid-grasp configurations where reset_env_to() would fail.

Non-Markovian stubs (snapshot_nm_state / restore_nm_state) capture all
per-env lists and per-part FSM attributes needed for full NM support; they
are included here for reference and are exercised when --non-markovian is
passed, but the physics snapshot/restore is identical in both modes.

Usage (Markovian, headless — recommended; omitting --headless renders the viewer
and camera images at ~30fps, making steps much slower):
    conda run -n imitation-juicer python scripts/test_state_rollback.py \\
        --furniture one_leg --headless

Custom checkpoints / lookahead:
    conda run -n imitation-juicer python scripts/test_state_rollback.py \\
        --furniture one_leg --headless --checkpoints 10 40 80 --lookahead 8

Non-Markovian:
    conda run -n imitation-juicer python scripts/test_state_rollback.py \\
        --furniture one_leg --headless --non-markovian

Speed note: this test runs faster than scripted.py for two reasons:
  1. --headless skips GUI rendering; scripted.py without --headless is
     throttled to ~30fps by the viewer.
  2. run_clean_steps uses noise-free clean actions, so the robot reaches its
     targets in fewer steps and FSM transitions happen sooner than under DART
     noise.  Each checkpoint also burns 2*lookahead extra clean steps.
"""

import argparse
import copy
import os
import sys
import time

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

# ── Physics + control snapshot ────────────────────────────────────────────────


def snapshot_osc_state(ctrl):
    """Snapshot all mutable state of one OSCController instance."""
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


def restore_osc_state(ctrl, snap):
    """Restore all mutable state of one OSCController instance."""
    ctrl.ee_pos_desired.data.copy_(snap["ee_pos_desired"])
    ctrl.ee_quat_desired.data.copy_(snap["ee_quat_desired"])
    ctrl.goal_pos.copy_(snap["goal_pos"])
    ctrl.prev_goal_pos.copy_(snap["prev_goal_pos"])
    ctrl.step_num_pos = snap["step_num_pos"]
    ctrl.goal_ori.copy_(snap["goal_ori"])
    ctrl.prev_goal_ori.copy_(snap["prev_goal_ori"])
    ctrl.step_num_ori = snap["step_num_ori"]
    ctrl.prev_interp_pos.copy_(snap["prev_interp_pos"])
    ctrl.prev_interp_ori.copy_(snap["prev_interp_ori"])
    ctrl.fraction = snap["fraction"]
    ctrl.repeated_torques_counter = snap["repeated_torques_counter"]
    ctrl.prev_torques.copy_(snap["prev_torques"])


def snapshot_physics(raw_env):
    """
    Clone the full simulation state needed for faithful rollback.

    Physics tensors (primary + derived):
      dof_states, root_tensor  — pushed to the physics engine via set_*_tensor.
      rb_states, jacobian, mm  — read by env.step() BEFORE the first simulate()
        to compute EE pose and OSC torques; must be correct at restore time.

    Control state:
      last_grasp        — gripper hysteresis; determines whether a gripper
                          command triggers a torque change or is a no-op.
      last_torque_action — actuation forces queued for the next simulate();
                          applied in refresh()'s first substep before new
                          torques are computed from the restored state.
      osc_ctrls         — OSC goal/interpolation/cache state; in particular,
                          repeated_torques_counter determines whether the
                          controller returns cached or freshly computed torques.
    """
    return {
        "dof_states": raw_env.dof_states.clone(),
        "root_tensor": raw_env.root_tensor.clone(),
        "rb_states": raw_env.rb_states.clone(),
        "jacobian": raw_env.jacobian.clone(),
        "mm": raw_env.mm.clone(),
        "last_grasp": raw_env.last_grasp.clone(),
        "ctrl_started": raw_env.ctrl_started,
        "last_torque_action": raw_env.last_torque_action.clone() if raw_env.last_torque_action is not None else None,
        "osc_ctrls": [snapshot_osc_state(c) for c in raw_env.osc_ctrls],
    }


def restore_physics(raw_env, snap, gripper_open_offset=0.0):
    """Restore the full simulation state from a snapshot.

    gripper_open_offset: if >0, nudge each finger DOF position open by this
    many metres before pushing state to PhysX.  Breaks gripper-object contact
    so the stale PhysX warm-start cache does not generate explosive corrective
    impulses when restoring into a mid-grasp state.  The gripper torque
    re-closes the fingers within the first step.
    """
    # Primary physics state — queued for the next simulate().
    raw_env.dof_states.copy_(snap["dof_states"])

    if gripper_open_offset > 0.0:
        # Finger DOFs are at per-env indices 7 and 8 within the 9-DOF Franka.
        max_finger = raw_env.max_gripper_width / 2
        finger_rows = torch.tensor(
            [i * 9 + f for i in range(raw_env.num_envs) for f in (7, 8)],
            device=raw_env.dof_states.device,
        )
        raw_env.dof_states[finger_rows, 0] = torch.clamp(
            raw_env.dof_states[finger_rows, 0] + gripper_open_offset,
            max=max_finger,
        )
        raw_env.dof_states[finger_rows, 1] = 0.0  # zero finger velocities

    raw_env.isaac_gym.set_dof_state_tensor(raw_env.sim, gymtorch.unwrap_tensor(raw_env.dof_states))

    # root_tensor is only refreshed via refresh_actor_root_state_tensor, which is
    # called once at init and never again during normal simulation.  It therefore
    # holds the last value written by a set_actor_root_state_tensor_indexed call
    # (e.g. an assembly snap-to-position), NOT the current PhysX state.
    # rb_states IS refreshed every step, so snap["rb_states"] is the authoritative
    # source for furniture part positions.  Overwrite the furniture part entries in
    # root_tensor from rb_states before calling the indexed set function.
    raw_env.root_tensor.copy_(snap["root_tensor"])
    rt_view = raw_env.root_tensor.view(raw_env.num_envs, -1, 13)
    for part_name, rb_idxs in raw_env.part_idxs.items():
        if part_name.startswith("obstacle"):
            continue
        actor_idx = raw_env.parts_handles.get(part_name)
        if actor_idx is None:
            continue
        for env_idx, rb_idx in enumerate(rb_idxs):
            # rb_states layout == root_tensor layout: [pos(3), quat(4), lin_vel(3), ang_vel(3)]
            rt_view[env_idx, actor_idx] = snap["rb_states"][rb_idx]

    raw_env.isaac_gym.set_actor_root_state_tensor_indexed(
        raw_env.sim,
        gymtorch.unwrap_tensor(raw_env.root_tensor),
        gymtorch.unwrap_tensor(raw_env.part_actor_idxs_all_t),
        len(raw_env.part_actor_idxs_all_t),
    )
    # Derived tensors — read by env.step() before the first simulate().
    raw_env.rb_states.copy_(snap["rb_states"])
    raw_env.jacobian.copy_(snap["jacobian"])
    raw_env.mm.copy_(snap["mm"])
    # Gripper hysteresis.
    raw_env.last_grasp.copy_(snap["last_grasp"])

    if snap["ctrl_started"]:
        # Controller was already initialised at snapshot time: restore full state.
        if snap["last_torque_action"] is not None:
            raw_env.last_torque_action = snap["last_torque_action"].clone()
            raw_env.isaac_gym.set_dof_actuation_force_tensor(
                raw_env.sim, gymtorch.unwrap_tensor(raw_env.last_torque_action)
            )
        for ctrl, ctrl_snap in zip(raw_env.osc_ctrls, snap["osc_ctrls"]):
            restore_osc_state(ctrl, ctrl_snap)
    else:
        # Controller was NOT yet initialised at snapshot time (e.g. step 0).
        # Clear the lists so env.step() calls init_ctrl() fresh from the
        # restored EE pose, exactly as it did in the original run.
        raw_env.ctrl_started = False
        raw_env.osc_ctrls.clear()
        raw_env.diffik_ctrls.clear()
        raw_env.last_torque_action = None
        # Explicitly zero actuation forces; without this, stale forces from
        # the end of the previous lookahead remain queued and are applied in
        # the normalization refresh(), diverging from the original run.
        zero = torch.zeros_like(raw_env.dof_pos)
        raw_env.isaac_gym.set_dof_actuation_force_tensor(raw_env.sim, gymtorch.unwrap_tensor(zero))


# ── Per-part FSM state snapshot (Markovian + NM) ─────────────────────────────


def snapshot_part_states(raw_env):
    """
    Capture per-part FSM state for all furniture environments.

    Fields captured beyond _last_state / _current_speed:
      gripper_action  — the gripper command (-1 open / +1 close) that the FSM
                        last set.  Returned directly in the action tuple by
                        pre_assemble(); if stale from the lookahead's release
                        phase (-1), the first post-restore call outputs act_grip=-1,
                        which sign-flips against last_grasp=+1 and physically opens
                        the gripper.
      pre_assemble_done — gating flag checked in get_assembly_action() to decide
                        whether to call pre_assemble() at all.  If the lookahead
                        completed the pre-assembly (set to True) but the restore
                        target is mid-pre-assembly (should be False), the policy
                        would skip the entire pre-assembly phase.
      prev_cnt / curr_cnt — FSM step counters; elapsed = curr_cnt - prev_cnt
                        determines when satisfy() fires.  Without restoring them
                        the FSM transitions too early or too late.
    """
    return [
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


def restore_part_states(raw_env, snap):
    """Restore per-part FSM state for all furniture environments."""
    for fi, furn_snap in enumerate(snap):
        for pi, ps in enumerate(furn_snap):
            p = raw_env.furnitures[fi].parts[pi]
            p._last_state = ps["_last_state"]
            p._current_speed = dict(ps["_current_speed"])
            p.gripper_action = ps["gripper_action"]
            p.pre_assemble_done = ps["pre_assemble_done"]
            p.prev_cnt = ps["prev_cnt"]
            p.curr_cnt = ps["curr_cnt"]


# ── Non-Markovian state snapshot ──────────────────────────────────────────────


def snapshot_nm_state(raw_env):
    """
    Capture all non-Markovian policy state that lives outside the physics sim.
    Must be saved alongside snapshot_physics() when using --non-markovian.

    Attributes accessed via raw_env (unwrapped FurnitureSimEnv) to bypass the
    gym Wrapper.__getattr__ block on private names.
    """

    def _cp(x):
        return x.copy() if hasattr(x, "copy") else copy.deepcopy(x)

    return {
        # Per-env pause injection state
        "_nm_pause_remaining": list(raw_env._nm_pause_remaining),
        "_nm_pause_gripper": list(raw_env._nm_pause_gripper),
        # Per-env virtual-target walk state
        "_nm_vt_pos": [_cp(x) if x is not None else None for x in raw_env._nm_vt_pos],
        "_nm_vt_vel": [_cp(x) for x in raw_env._nm_vt_vel],
        "_nm_vt_ori": [_cp(x) for x in raw_env._nm_vt_ori],
        "_nm_vt_ang_vel": [_cp(x) for x in raw_env._nm_vt_ang_vel],
        "_nm_vt_falloff_dist": list(raw_env._nm_vt_falloff_dist),
        "_nm_vt_falloff_angle": list(raw_env._nm_vt_falloff_angle),
        # Per-step temporally-correlated noise state
        "_corr_noise_state": {
            k: {"pos": v["pos"].clone(), "aa": v["aa"].copy()} for k, v in raw_env._corr_noise_state.items()
        },
        # Per-furniture part NM-specific FSM state (_last_state handled by snapshot_part_states)
        "furnitures": [
            {
                "parts": [
                    {
                        "_nm_sn_post_pause_pending": p._nm_sn_post_pause_pending,
                        "_nm_sn_post_pause_steps": p._nm_sn_post_pause_steps,
                        "_nm_sticky_countdowns": dict(p._nm_sticky_countdowns),
                        "_nm_transition_pause_requested": p._nm_transition_pause_requested,
                        "_nm_pending_next_state": p._nm_pending_next_state,
                    }
                    for p in furniture.parts
                ]
            }
            for furniture in raw_env.furnitures
        ],
    }


def restore_nm_state(raw_env, snap):
    """Restore all non-Markovian policy state from a snapshot."""
    raw_env._nm_pause_remaining[:] = snap["_nm_pause_remaining"]
    raw_env._nm_pause_gripper[:] = snap["_nm_pause_gripper"]
    for i, x in enumerate(snap["_nm_vt_pos"]):
        raw_env._nm_vt_pos[i] = x.copy() if x is not None else None
    for i, x in enumerate(snap["_nm_vt_vel"]):
        raw_env._nm_vt_vel[i] = x.copy()
    for i, x in enumerate(snap["_nm_vt_ori"]):
        raw_env._nm_vt_ori[i] = x.copy()
    for i, x in enumerate(snap["_nm_vt_ang_vel"]):
        raw_env._nm_vt_ang_vel[i] = x.copy()
    raw_env._nm_vt_falloff_dist[:] = snap["_nm_vt_falloff_dist"]
    raw_env._nm_vt_falloff_angle[:] = snap["_nm_vt_falloff_angle"]
    raw_env._corr_noise_state.clear()
    for k, v in snap["_corr_noise_state"].items():
        raw_env._corr_noise_state[k] = {"pos": v["pos"].clone(), "aa": v["aa"].copy()}
    for fi, fs in enumerate(snap["furnitures"]):
        for pi, ps in enumerate(fs["parts"]):
            p = raw_env.furnitures[fi].parts[pi]
            p._nm_sn_post_pause_pending = ps["_nm_sn_post_pause_pending"]
            p._nm_sn_post_pause_steps = ps["_nm_sn_post_pause_steps"]
            p._nm_sticky_countdowns = dict(ps["_nm_sticky_countdowns"])
            p._nm_transition_pause_requested = ps["_nm_transition_pause_requested"]
            p._nm_pending_next_state = ps["_nm_pending_next_state"]


# ── Test helpers ──────────────────────────────────────────────────────────────


def _extract_record(obs, clean_action, raw_env):
    """Pull comparable state from one step's obs + clean_action into a dict."""
    rs = obs.get("robot_state", {})
    if isinstance(rs, dict):
        raw_jp = rs.get("joint_positions", [])
    else:
        raw_jp = rs
    if isinstance(raw_jp, torch.Tensor):
        raw_jp = raw_jp.cpu()
    joint_pos = np.asarray(raw_jp).flatten()
    if not isinstance(rs, dict):
        joint_pos = joint_pos[:7]

    pp = obs["parts_poses"]
    if isinstance(pp, torch.Tensor):
        pp = pp.cpu()

    # Read gripper DOF positions directly from physics (finger separation)
    gripper_dof = raw_env.dof_pos[0, 7:9].cpu().numpy()

    return {
        "joint_pos": joint_pos,
        "parts_poses": np.asarray(pp).flatten(),
        "action": clean_action.cpu().numpy().copy().squeeze(),
        "gripper_dof": gripper_dof,
    }


def _print_ctrl_state(raw_env, label):
    """Print key control-layer state for debugging."""
    lg = raw_env.last_grasp[0].item()
    rtc = raw_env.osc_ctrls[0].repeated_torques_counter if raw_env.osc_ctrls else "N/A"
    pt = raw_env.osc_ctrls[0].prev_torques.cpu().numpy() if raw_env.osc_ctrls else []
    gp = raw_env.dof_pos[0, 7:9].cpu().numpy()
    lta = raw_env.last_torque_action[0, :7].cpu().numpy() if raw_env.last_torque_action is not None else []
    print(f"  [{label}]")
    print(f"    last_grasp={lg:+.3f}  gripper_dof={np.round(gp, 4)}  osc_counter={rtc}")
    print(f"    prev_torques   = {np.round(pt, 3)}")
    print(f"    last_torque_action[:7] = {np.round(lta, 3)}")


def _debug_part_poses(raw_env, label):
    """Print root_tensor buffer values, rb_states positions, and FSM state for furniture parts."""
    rt = raw_env.root_tensor.view(raw_env.num_envs, -1, 13)[0]  # (num_actors, 13)
    print(f"  [DEBUG {label}] part_actor_idxs = {raw_env.part_actor_idxs_all_t.cpu().numpy()}")
    for furn in raw_env.furnitures:
        for p in furn.parts:
            part_name = p.name
            rb_idxs = raw_env.part_idxs.get(part_name)
            if rb_idxs is None or part_name.startswith("obstacle"):
                continue
            actor_idx = raw_env.parts_handles.get(part_name)
            rt_pos = rt[actor_idx, :3].cpu().numpy() if actor_idx is not None else "N/A"
            rb_pos = raw_env.rb_states[rb_idxs[0], :3].cpu().numpy()
            print(
                f"  [DEBUG {label}] {part_name}: root_tensor={np.round(rt_pos, 4)}"
                f"  rb_states={np.round(rb_pos, 4)}"
                f"  fsm={p._last_state!r}  grip_act={p.gripper_action}  pre_done={p.pre_assemble_done}"
                f"  cnt=({p.prev_cnt},{p.curr_cnt})"
            )


def run_clean_steps(env, raw_env, n_steps, label="", n_norm_steps=3, debug=False):
    """
    Execute up to n_steps using clean (no-noise) actions.

    Begins with n_norm_steps refresh() calls ("normalisation") to flush the
    PhysX contact warm-start cache.  The cache is NOT restorable via any
    IsaacGym API; a stale cache from the end of the previous lookahead causes
    premature FSM transitions (act_grip sign flip) and a constant ~8e-4 offset
    in parts_poses.

    With n_norm_steps=1 the contact pairs are recomputed once from the restored
    geometry but the warm-start biases still influence the solver.  With
    n_norm_steps≥3, both original and restored trajectories converge to the
    same settled contact state regardless of prior cache history, reducing the
    residual parts_poses offset.

    Both original and restored call this block from the same snapshot state T
    with the same queued forces, so both reach the same T+n_norm_steps and the
    comparison remains valid.
    """
    # Ensure identical forces for the normalisation steps.
    # For ctrl_started=False (e.g. step 0) no forces have ever been set;
    # explicitly zero them so "never-set" and "explicitly-zero" behave the same.
    if raw_env.last_torque_action is None:
        zero = torch.zeros_like(raw_env.dof_pos)
        raw_env.isaac_gym.set_dof_actuation_force_tensor(raw_env.sim, gymtorch.unwrap_tensor(zero))
    print(f"    {label}[norm] {n_norm_steps}x refresh() to settle contact cache...")
    for i in range(n_norm_steps):
        raw_env.refresh()
        if debug and i == 0:
            # After first simulate(): refresh actor root tensor to see what PhysX
            # actually placed the actors at.  root_tensor is not normally refreshed
            # in refresh(), so this explicit call is the only way to read it back.
            raw_env.isaac_gym.refresh_actor_root_state_tensor(raw_env.sim)
            print(
                f"  [DEBUG {label}] actor root states post-norm-step-0 (from PhysX via refresh_actor_root_state_tensor):"
            )
            _debug_part_poses(raw_env, f"{label} post-norm")

    records = []
    for i in range(n_steps):
        _, clean_action, _ = env.get_assembly_action()

        pos_bounds_m = 0.02 if env.ctrl_mode == "diffik" else 0.025
        ori_bounds_deg = 15 if env.ctrl_mode == "diffik" else 20
        action = scale_scripted_action(
            clean_action.detach().cpu().clone(),
            pos_bounds_m=pos_bounds_m,
            ori_bounds_deg=ori_bounds_deg,
            device=env.device,
        )

        obs, _, done, _ = env.step(action)
        rec = _extract_record(obs, clean_action, raw_env)
        records.append(rec)

        act = rec["action"]
        print(
            f"    {label}[{i}] jp={np.round(rec['joint_pos'], 3)}  "
            f"grip_dof={np.round(rec['gripper_dof'], 4)}  "
            f"act_grip={act[-1]:+.3f}  "
            f"act_pos={np.round(act[:3], 4)}  "
            f"parts0xyz={np.round(rec['parts_poses'][:3], 4)}"
        )

        if done.any():
            print(f"    {label} episode ended at lookahead step {i}.")
            return records, True
    return records, False


def compare_trajectories(orig, rest, label, tol=5e-3):
    n = min(len(orig), len(rest))
    if n == 0:
        print(f"  [SKIP] {label} — no steps to compare")
        return True

    print(f"\n  --- Per-step diff ({label}) ---")
    print(
        f"  {'step':>4}  {'joint_max_Δ':>11}  {'parts_max_Δ':>11}  {'act_max_Δ':>10}  {'act_grip_o':>10}  {'act_grip_r':>10}  {'worst_act_idx':>13}"
    )
    any_fail = False
    for t in range(n):
        jd = np.max(np.abs(orig[t]["joint_pos"] - rest[t]["joint_pos"]))
        pd = np.max(np.abs(orig[t]["parts_poses"] - rest[t]["parts_poses"]))
        # Quaternion sign-invariant action comparison:
        # q and -q represent the same rotation; compare both and take the smaller.
        # Action layout: [dx, dy, dz, qx, qy, qz, qw, gripper] — indices 3:7 are quat.
        ao, ar = orig[t]["action"], rest[t]["action"]
        ad_raw = np.abs(ao - ar)
        ar_flipped = ar.copy()
        ar_flipped[3:7] *= -1
        ad_flipped = np.abs(ao - ar_flipped)
        # Use flipped comparison for quat indices if it gives a smaller max.
        ad = ad_raw.copy()
        if np.max(ad_flipped[3:7]) < np.max(ad_raw[3:7]):
            ad[3:7] = ad_flipped[3:7]
        worst_i = int(np.argmax(ad))
        step_ok = jd < tol and pd < tol and float(np.max(ad)) < tol
        if not step_ok:
            any_fail = True
        tag = "" if step_ok else " ←"
        print(
            f"  {t:>4}  {jd:>11.3e}  {pd:>11.3e}  {np.max(ad):>10.3e}"
            f"  {orig[t]['action'][-1]:>+10.4f}  {rest[t]['action'][-1]:>+10.4f}"
            f"  [{worst_i}]{tag}"
        )

    max_joint = max(np.max(np.abs(orig[t]["joint_pos"] - rest[t]["joint_pos"])) for t in range(n))
    max_parts = max(np.max(np.abs(orig[t]["parts_poses"] - rest[t]["parts_poses"])) for t in range(n))
    max_action = max(np.max(np.abs(orig[t]["action"] - rest[t]["action"])) for t in range(n))
    ok = not any_fail
    tag = "PASS" if ok else "FAIL"
    print(f"\n  [{tag}] {label}")
    print(f"         joint Δ={max_joint:.2e}  parts Δ={max_parts:.2e}  action Δ={max_action:.2e}")
    return ok


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--furniture", "-f", type=str, required=True)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--non-markovian", action="store_true", help="Use NM scripted policy and exercise NM state snapshot/restore."
    )
    parser.add_argument(
        "--dart-amount", type=float, default=1.0, help="DART noise scale for noisy steps between checkpoints."
    )
    parser.add_argument("--seed", type=int, default=75)
    parser.add_argument(
        "--checkpoints",
        type=int,
        nargs="+",
        default=[200],
        help="Episode step indices at which to test rollback.",
    )
    parser.add_argument(
        "--lookahead",
        type=int,
        default=25,
        help="Clean steps to run before and after each rollback for comparison (default: 65).",
    )
    parser.add_argument(
        "--norm-steps",
        type=int,
        default=1,
        help="refresh() calls before each comparison window to flush the PhysX contact cache (default: 1). "
        "Values >1 cause the robot arm to drift under gravity and make things worse.",
    )
    parser.add_argument(
        "--gripper-open-offset",
        type=float,
        default=0.000,
        help="Metres to nudge each finger DOF open at restore time to break gripper-object contact "
        "before PhysX sees the state. Set to 0 to disable.",
    )
    args = parser.parse_args()

    data_path = trajectory_save_dir(
        environment="sim",
        task=args.furniture,
        demo_source="scripted_rollback_test",
        randomness="low",
    )

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
        save_failure=False,
        num_demos=1,
        resize_sim_img=False,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        ctrl_mode="osc",
        compress_pickles=False,
        non_markovian=args.non_markovian,
        n_video_trials=0,
        record_failures=False,
        no_noise=False,
        dart_amount=args.dart_amount,
        num_envs=1,
        seed=args.seed,
    )

    # collector.env is a gym Wrapper; non-private attrs delegate to FurnitureSimEnv
    # via __getattr__.  Private attrs (_nm_*, etc.) need .unwrapped.
    env = collector.env
    raw_env = collector.env.unwrapped
    env.reset()

    checkpoints = sorted(args.checkpoints)
    print(f"\nFurniture        : {args.furniture}")
    print(f"Mode             : {'non-Markovian' if args.non_markovian else 'Markovian'}")
    print(f"Checkpoints      : {checkpoints}")
    print(f"Lookahead        : {args.lookahead} clean steps per rollback")
    print(f"DART amount      : {args.dart_amount}")
    print(f"Gripper offset   : {args.gripper_open_offset} m per finger\n")

    step = 0
    done = torch.zeros(1, dtype=torch.bool)
    results = []

    for cp in checkpoints:
        # ── Advance to checkpoint with noisy (DART) actions ──────────────────
        while step < cp:
            print(f"CURRENT STEP: {step}")
            # --- Compute actions for all envs ---
            _, clean_action, _ = env.get_assembly_action()

            pos_bounds_m = 0.02 if env.ctrl_mode == "diffik" else 0.025
            ori_bounds_deg = 15 if env.ctrl_mode == "diffik" else 20
            action = scale_scripted_action(
                clean_action.detach().cpu().clone(),
                pos_bounds_m=pos_bounds_m,
                ori_bounds_deg=ori_bounds_deg,
                device=env.device,
            )

            obs, _, done, _ = env.step(action)
            step += 1
            if done.any():
                break

        time.sleep(3.0)

        if done.any():
            print(f"Episode ended at step {step} before checkpoint {cp}. Stopping.")
            break

        # ── Checkpoint state dump ─────────────────────────────────────────────
        jp = raw_env.dof_pos[0, :7].cpu().numpy()
        pp = raw_env.root_tensor.view(1, -1, 13)[0, :, :3].cpu().numpy()  # actor positions
        ee_pos, ee_quat = raw_env.get_ee_pose()
        fsm_states = [p._last_state for p in raw_env.furnitures[0].parts]
        print(f"\n=== Checkpoint step {step} ===")
        print(f"  joint_pos   = {np.round(jp, 4)}")
        print(f"  gripper_dof = {np.round(raw_env.dof_pos[0, 7:9].cpu().numpy(), 4)}")
        print(f"  ee_pos      = {np.round(ee_pos[0].cpu().numpy(), 4)}")
        print(f"  ee_quat     = {np.round(ee_quat[0].cpu().numpy(), 4)}")
        print(f"  actor_xyz   = {np.round(pp, 4)}")
        print(f"  fsm_states  = {fsm_states}")
        _print_ctrl_state(raw_env, "snapshot")

        # ── Save state ───────────────────────────────────────────────────────
        phys_snap = snapshot_physics(raw_env)
        part_snap = snapshot_part_states(raw_env)
        nm_snap = snapshot_nm_state(raw_env) if args.non_markovian else None

        # ── Original: clean lookahead from saved state ───────────────────────
        print(f"\n  → Original: {args.lookahead} clean steps")
        traj_orig, _ = run_clean_steps(env, raw_env, args.lookahead, label="orig", n_norm_steps=args.norm_steps)

        # ── Restore ──────────────────────────────────────────────────────────
        restore_physics(raw_env, phys_snap, gripper_open_offset=args.gripper_open_offset)
        restore_part_states(raw_env, part_snap)
        if args.non_markovian:
            restore_nm_state(raw_env, nm_snap)

        jp_after = raw_env.dof_pos[0, :7].cpu().numpy()
        ee_pos2, ee_quat2 = raw_env.get_ee_pose()
        fsm_after = [p._last_state for p in raw_env.furnitures[0].parts]
        print("\n  After restore:")
        print(f"  joint_pos   = {np.round(jp_after, 4)}")
        print(f"  gripper_dof = {np.round(raw_env.dof_pos[0, 7:9].cpu().numpy(), 4)}")
        print(f"  ee_pos      = {np.round(ee_pos2[0].cpu().numpy(), 4)}")
        print(f"  fsm_states  = {fsm_after}")
        _print_ctrl_state(raw_env, "restored")
        # Debug: show what root_tensor buffer and rb_states have immediately after
        # restore (before any simulate() is called).  root_tensor buffer = snapshot
        # values; rb_states = still from the end of the original lookahead.
        _debug_part_poses(raw_env, "restore (pre-norm)")

        # ── Restored: same clean steps ────────────────────────────────────────
        print(f"\n  → Restored: {args.lookahead} clean steps")
        traj_rest, _ = run_clean_steps(
            env, raw_env, args.lookahead, label="rest", n_norm_steps=args.norm_steps, debug=True
        )

        ok = compare_trajectories(traj_orig, traj_rest, f"rollback @ step {cp}")
        results.append((cp, ok))

        # Continue from end of re-run lookahead.
        step += len(traj_rest)
        done = torch.zeros(1, dtype=torch.bool)
        print()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=== Summary ===")
    all_pass = True
    for cp, ok in results:
        print(f"  step {cp:>4d}: {'PASS' if ok else 'FAIL'}")
        all_pass = all_pass and ok
    if not results:
        print("  No checkpoints reached (episode ended before first checkpoint).")

    overall = "ALL PASS" if (all_pass and results) else "SOME FAILURES"
    print(f"\nOverall: {overall}")
    sys.exit(0 if (all_pass and results) else 1)


if __name__ == "__main__":
    main()
