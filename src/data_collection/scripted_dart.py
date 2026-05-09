"""
Same as scripted.py but uses DART-rollbackto collect data with DART.

At each step T of a noisy (DART) trajectory:
  1. Record obs_T from the current state S_T.
  2. Snapshot S_T (physics + FSM).
  3. Run CHUNK_SIZE clean steps from S_T → action chunk labels.
  4. Restore to S_T (with warm-start flush if insertion contact detected).
  5. Apply one noisy DART step → S_{T+1}, obs_{T+1}.

Output: pickle per episode with noisy-trajectory observations paired with
coherent clean action chunks.

NOTE: only compatible with --n-envs = 1

Usage:
    conda run -n imitation-juicer python -m src.data_collection.scripted_dart \\
        --furniture one_leg --headless
"""

import argparse
import copy
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

import furniture_bench.controllers.control_utils as C
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action

from src.common.files import trajectory_save_dir
from src.data_collection.data_collector import DataCollector
from src.data_processing.utils import resize, resize_crop

# Gripper opening added to each finger DOF during the zero-velocity flush step.
# Shrinks the friction cone so PGS projects large insertion-phase impulses toward zero.
_GRIPPER_OPEN_OFFSET = 0.001
_LEG_TIP_OFFSET = 0.05625  # Leg._LEG_TIP_OFFSET: distance from mesh origin to screw tip
_INSERTION_Z_THRESHOLD = 0.01675  # leg.py STUCK_Z_HIGH: screw tip is inside the hole

# Z threshold (sim world frame) at which we consider the leg to have been lifted off the table by the gripper.
_PICK_Z_SIM = 0.45


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
        "env_steps": raw_env.env_steps.clone(),
        "scripted_timeout": list(raw_env.scripted_timeout),
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
                # All nm_* / _nm_* instance attributes (NM policy state + prev_leg_tip_z_rel).
                # Using vars() captures only instance attrs, not class methods.
                "nm_attrs": {
                    k: copy.deepcopy(v)
                    for k, v in vars(p).items()
                    if k.startswith("nm_") or k.startswith("_nm_") or k in ("prev_leg_tip_z_rel", "prev_leg_z_vel_robot")
                },
            }
            for p in furn.parts
        ]
        for furn in raw_env.furnitures
    ]
    # NM virtual-target state lives on raw_env (one entry per env).
    phys["nm_vt"] = {
        "_nm_vt_pos": copy.deepcopy(raw_env._nm_vt_pos),
        "_nm_vt_vel": copy.deepcopy(raw_env._nm_vt_vel),
        "_nm_vt_ori": copy.deepcopy(raw_env._nm_vt_ori),
        "_nm_vt_ang_vel": copy.deepcopy(raw_env._nm_vt_ang_vel),
        "_nm_vt_falloff_dist": list(raw_env._nm_vt_falloff_dist),
        "_nm_vt_falloff_angle": list(raw_env._nm_vt_falloff_angle),
    }
    return phys, parts


def restore(raw_env, phys, parts, zero_velocities: bool = False, gripper_open_offset: float = 0.0):
    """Push a snapshot back into the running simulation.

    zero_velocities: zero all DOF and furniture-part velocities before the PhysX
      set call so the contact solver sees a fully static scene during the flush
      simulate(), preventing dynamic-friction contamination of the warm-start.
    gripper_open_offset: metres added to each finger DOF position to reduce
      contact compression during flush steps.
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

    for fi, fsnap in enumerate(parts):
        for pi, ps in enumerate(fsnap):
            p = raw_env.furnitures[fi].parts[pi]
            p._last_state = ps["_last_state"]
            p._current_speed = dict(ps["_current_speed"])
            p.gripper_action = ps["gripper_action"]
            p.pre_assemble_done = ps["pre_assemble_done"]
            p.prev_cnt = ps["prev_cnt"]
            p.curr_cnt = ps["curr_cnt"]
            for k, v in ps["nm_attrs"].items():
                setattr(p, k, copy.deepcopy(v))

    # NM virtual-target state.
    nm = phys["nm_vt"]
    raw_env._nm_vt_pos = copy.deepcopy(nm["_nm_vt_pos"])
    raw_env._nm_vt_vel = copy.deepcopy(nm["_nm_vt_vel"])
    raw_env._nm_vt_ori = copy.deepcopy(nm["_nm_vt_ori"])
    raw_env._nm_vt_ang_vel = copy.deepcopy(nm["_nm_vt_ang_vel"])
    raw_env._nm_vt_falloff_dist = list(nm["_nm_vt_falloff_dist"])
    raw_env._nm_vt_falloff_angle = list(nm["_nm_vt_falloff_angle"])


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


def collect_episode(env, raw_env, chunk_size: int):
    obs = env.reset()
    done = torch.zeros(1, dtype=torch.bool)

    observations, action_chunks, actions, rewards, skills = [], [], [], [], []

    # Active leg: use skill_attach_part_idx rather than the first "leg" in
    # part_idxs — other legs may be pre-assembled and fixed at a constant z.
    furn = raw_env.furnitures[0]
    leg_name = furn.parts[furn.skill_attach_part_idx].name
    top_name = furn.parts[0].name

    def _leg_tip_z_rel():
        """Screw-tip height above the table surface in robot frame (mirrors leg.py)."""
        sim_to_april = raw_env.sim_to_april_mat
        april_to_robot = raw_env.april_to_robot_mat
        rb = raw_env.rb_states
        leg_pose = C.to_homogeneous(
            rb[raw_env.part_idxs[leg_name]][0][:3],
            C.quat2mat(rb[raw_env.part_idxs[leg_name]][0][3:7]),
        )
        leg_pose_robot = april_to_robot @ sim_to_april @ leg_pose
        top_pose = C.to_homogeneous(
            rb[raw_env.part_idxs[top_name]][0][:3],
            C.quat2mat(rb[raw_env.part_idxs[top_name]][0][3:7]),
        )
        top_pose_robot = april_to_robot @ sim_to_april @ top_pose
        leg_z_rel = leg_pose_robot[2, 3] - top_pose_robot[2, 3]
        return (leg_z_rel - leg_pose_robot[2, 1] * _LEG_TIP_OFFSET).item()

    leg_has_been_picked = False

    while not done.any():
        obs_np = _obs_to_numpy(obs)
        obs_small = {k: obs_np[k] for k in ["color_image1", "color_image2", "robot_state", "parts_poses"]}

        # Latch True once the leg rises clearly above the table; never resets
        # within an episode.  Guards against spurious insertion-contact flushes
        # during the reach/grasp phases when the leg is still on the table.
        if not leg_has_been_picked:
            leg_z = raw_env.rb_states[raw_env.part_idxs[leg_name]][0][2].item()
            if leg_z > _PICK_Z_SIM:
                leg_has_been_picked = True

        phys_snap, part_snap = snapshot(raw_env)

        chunk = []
        for _ in range(chunk_size):
            _, ca, _ = env.get_assembly_action()
            _, _, lookahead_done, _ = env.step(_scale(ca, env))
            chunk.append(ca.detach().cpu().numpy().squeeze())
            if lookahead_done.any():
                while len(chunk) < chunk_size:
                    chunk.append(chunk[-1].copy())
                break

        # Flush the PhysX warm-start cache only when the lookahead drove the
        # screw tip into the hole — otherwise the centripetal forces in the
        # warm-start are correct and should be preserved.
        # The leg_has_been_picked gate prevents spurious flushes before the leg is airborne.
        if leg_has_been_picked and _leg_tip_z_rel() < _INSERTION_Z_THRESHOLD:
            restore(raw_env, phys_snap, part_snap, zero_velocities=True, gripper_open_offset=_GRIPPER_OPEN_OFFSET)
            raw_env.refresh()
            restore(raw_env, phys_snap, part_snap)
            raw_env.refresh()
            restore(raw_env, phys_snap, part_snap)
        else:
            restore(raw_env, phys_snap, part_snap)

        noisy_action, clean_action, skill_complete = env.get_assembly_action()
        scaled_noisy = _scale(noisy_action, env)
        obs, rew, done, info = env.step(scaled_noisy)

        rew_val = float(rew[0].squeeze().cpu())
        skill_val = int(skill_complete[0]) if isinstance(skill_complete, (list, tuple)) else int(skill_complete)

        observations.append(obs_small)
        action_chunks.append(np.stack(chunk))
        actions.append(scaled_noisy.detach().cpu().numpy().squeeze())
        rewards.append(rew_val)
        skills.append(skill_val)

    return {
        "observations": observations,
        "action_chunks": action_chunks,
        "actions": actions,
        "rewards": rewards,
        "skills": skills,
        "success": raw_env.furnitures[0].all_assembled(),
    }


# ── Entry point ───────────────────────────────────────────────────────────────


def _count_success(data_path) -> int:
    d = data_path / "success"
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.suffix in (".pkl", ".xz"))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--furniture", "-f", type=str, required=True)
    parser.add_argument("--randomness", "-r", type=str, default="low")
    parser.add_argument("--num-demos", "-n", type=int, default=100)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--num-envs", "-e", type=int, default=1)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--dart-amount", type=float, default=1.0)
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--save-failure", action="store_true")
    parser.add_argument("--non-markovian", action="store_true")
    parser.add_argument("--output-dir-suffix", type=str, default=None)
    parser.add_argument("--n-video-trials", type=int, default=20)
    parser.add_argument("--record-failures", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None, help="Fixed process seed (default: random uuid)")
    args = parser.parse_args()
    assert args.num_envs == 1, "scripted_dart.py only supports --num-envs 1 (DART rollback is single-env only)"

    suffix_parts = []
    if args.output_dir_suffix:
        suffix_parts.append(args.output_dir_suffix)
    demo_source = "scripted" + (f"_{'_'.join(suffix_parts)}" if suffix_parts else "")

    data_path = trajectory_save_dir(
        environment="sim",
        task=args.furniture,
        demo_source=demo_source,
        randomness=args.randomness,
    )

    process_seed = args.seed if args.seed is not None else (uuid.uuid4().int & 0x7FFFFFFF)
    print(f"[seed] process_seed={process_seed}  demo_source={demo_source}")

    collector = DataCollector(
        is_sim=True,
        data_path=data_path,
        furniture=args.furniture,
        device_interface=None,
        headless=args.headless,
        manual_label=False,
        scripted=True,
        draw_marker=True,
        randomness=args.randomness,
        save_failure=args.save_failure,
        num_demos=args.num_demos,
        resize_sim_img=False,
        compute_device_id=args.gpu_id,
        graphics_device_id=args.gpu_id,
        ctrl_mode="osc",
        compress_pickles=True,
        non_markovian=args.non_markovian,
        n_video_trials=args.n_video_trials,
        record_failures=args.record_failures,
        no_noise=args.no_noise,
        dart_amount=args.dart_amount,
        num_envs=args.num_envs,
        seed=process_seed,
    )

    env = collector.env
    raw_env = collector.env.unwrapped

    target = args.num_demos
    episode_idx = 0
    n_fail = 0

    while _count_success(data_path) < target:
        print(
            f"\n[episode={episode_idx}  on-disk={_count_success(data_path)}/{target}]"
            f"  chunk_size={args.chunk_size}  dart={args.dart_amount}"
        )

        np.random.seed((process_seed + episode_idx) % (2**31))

        data = collect_episode(env, raw_env, chunk_size=args.chunk_size)
        data["furniture"] = args.furniture
        data["chunk_size"] = args.chunk_size
        data["process_seed"] = process_seed
        data["episode_idx"] = episode_idx
        episode_idx += 1

        if data["success"]:
            if _count_success(data_path) >= target:
                print("  Target reached by another process — discarding.")
                break
        else:
            n_fail += 1
            if not args.save_failure:
                print(f"  Failure — skipping save  (failures={n_fail})")
                continue

        subdir = "success" if data["success"] else "failure"
        out_dir = data_path / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        out_path = out_dir / f"{ts}_pid{os.getpid()}.pkl.xz"
        with lzma.open(out_path, "wb") as f:
            pickle.dump(data, f)
        print(f"  Saved → {out_path}  (on-disk={_count_success(data_path)}/{target}, failures={n_fail})")

    print(f"\nDone. episodes_run={episode_idx}, on-disk={_count_success(data_path)}/{target}.")


if __name__ == "__main__":
    main()
