"""Define data collection class that rollout the environment, get action from the interface
(e.g., teleoperation, automatic scripts), and save data."""

from datetime import datetime
from pathlib import Path

import gym
import numpy as np
import torch
from furniture_bench.data.collect_enum import CollectEnum
from furniture_bench.device.device_interface import DeviceInterface
from furniture_bench.envs.initialization_mode import Randomness
from furniture_bench.sim_config import sim_config
from furniture_bench.utils.scripted_demo_mod import scale_scripted_action

from src.data_processing.utils import resize, resize_crop
from src.visualization.render_mp4 import create_mp4, data_to_video, pickle_data


class DataCollector:
    """Demonstration collection class.

    Only used for collecting scripted demos in the simulator

    For teleoperation demos, use `DataCollectorSM` instead.

    For now, only stores pickle files, either compressed or uncompressed.
    """

    def __init__(
        self,
        is_sim: bool,
        data_path: str,
        device_interface: DeviceInterface,
        furniture: str,
        headless: bool,
        draw_marker: bool,
        manual_label: bool,
        scripted: bool,
        randomness: Randomness.LOW,
        compute_device_id: int,
        graphics_device_id: int,
        save_failure: bool = False,
        num_demos: int = 100,
        resize_sim_img: bool = False,
        ctrl_mode: str = "osc",
        compress_pickles: bool = False,
        verbose: bool = False,
        non_markovian: bool = False,
        n_video_trials: int = 0,
        record_failures: bool = False,
        no_noise: bool = False,
        dart_amount: float = 1.0,
        num_envs: int = 1,
    ):
        """
        Args:
            is_sim (bool): Whether to use simulator or real world environment.
            data_path (str): Path to save data.
            device_interface (DeviceInterface): Keyboard and/or Oculus interface.
            furniture (str): Name of the furniture.
            headless (bool): Whether to use headless mode.
            draw_marker (bool): Whether to draw AprilTag marker.
            manual_label (bool): Whether to manually label the reward.
            scripted (bool): Whether to use scripted function for getting action.
            randomness (str): Initialization randomness level.
            compute_device_id (int): GPU device ID used for simulation.
            graphics_device_id (int): GPU device ID used for rendering.
            save_failure (bool): Whether to save failure trajectories.
            num_demos (int): The maximum number of demonstrations to collect in this run. Internal loop will be terminated when this number is reached.
            resize_sim_img (bool): Read resized image.
            ctrl_mode (str): 'osc' (joint torque, with operation space control) or 'diffik' (joint impedance, with differential inverse kinematics control).
            compress_pickles (bool): Whether to compress the pickle files with gzip.
            n_video_trials (int): Save videos for the first N trials. Set to -1 to save all. Default 0 (no video).
            record_failures (bool): If True, also save videos of all failed trials beyond n_video_trials.
            num_envs (int): Number of parallel Isaac Gym environments. Only supported for scripted sim collection.
        """
        np.random.seed(2043961395)
        if is_sim:
            self.env = gym.make(
                "FurnitureSimFull-v0",
                furniture=furniture,
                max_env_steps=sim_config["scripted_timeout"][furniture] if scripted else 3000,
                headless=headless,
                num_envs=num_envs,
                manual_done=False if scripted else True,
                resize_img=resize_sim_img,
                np_step_out=False,  # Always output Tensor in this setting. Will change to numpy in this code.
                channel_first=False,
                randomness=randomness,
                compute_device_id=compute_device_id,
                graphics_device_id=graphics_device_id,
                ctrl_mode=ctrl_mode,
                no_noise=no_noise,
                dart_amount=dart_amount,
            )
        else:
            if num_envs > 1:
                raise ValueError("num_envs > 1 is only supported for sim (is_sim=True)")
            if randomness == "med":
                randomness = Randomness.MEDIUM_COLLECT
            elif randomness == "high":
                randomness = Randomness.HIGH_COLLECT

            self.env = gym.make(
                "FurnitureBench-v0",
                furniture=furniture,
                resize_img=False,
                manual_done=True,
                with_display=not headless,
                draw_marker=draw_marker,
                randomness=randomness,
            )

        self.is_sim = is_sim
        self.num_envs = num_envs
        self.data_path = Path(data_path)
        self.device_interface = device_interface
        self.headless = headless
        self.manual_label = manual_label
        self.furniture = furniture
        self.num_demos = num_demos
        self.scripted = scripted

        self.traj_counter = 0
        self.num_success = 0
        self.num_fail = 0

        self.save_failure = save_failure
        self.resize_sim_img = resize_sim_img
        self.compress_pickles = compress_pickles
        self.verbose = verbose
        self.non_markovian = non_markovian
        self.n_video_trials = n_video_trials
        self.record_failures = record_failures
        self.n_videos_saved = 0

        self._reset_all_buffers()

    def _verbose_print(self, msg):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Buffer helpers
    # ------------------------------------------------------------------

    def _make_empty_buffers(self):
        return dict(obs=[], org_obs=[], acts=[], rews=[], skills=[], step_counter=0, last_reward_idx=-1, skill_set=[])

    def _reset_all_buffers(self):
        self._bufs = [self._make_empty_buffers() for _ in range(self.num_envs)]

    def _reset_env_buffer(self, env_idx: int):
        self._bufs[env_idx] = self._make_empty_buffers()

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _obs_to_numpy(self, obs: dict, env_idx: int) -> dict:
        """Extract env_idx's slice from a batched obs dict and convert to numpy."""
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

    def _store_step(self, env_idx: int, obs_np: dict, record_action, rew_val: float, skill_val):
        """Append one (obs, action, reward, skill) tuple to env_idx's buffer."""
        buf = self._bufs[env_idx]
        buf["org_obs"].append(obs_np.copy())
        ob = {k: obs_np[k] for k in ["color_image1", "color_image2", "robot_state", "parts_poses"]}
        buf["obs"].append(ob)
        if isinstance(record_action, torch.Tensor):
            record_action = record_action.detach().cpu().numpy()
        buf["acts"].append(record_action)
        buf["rews"].append(rew_val)
        if rew_val == 1:
            buf["last_reward_idx"] = len(buf["acts"]) - 1
        buf["skills"].append(int(skill_val))
        buf["step_counter"] += 1

    def _store_terminal_obs(self, env_idx: int, obs: dict):
        """Append the terminal (post-done) obs to env_idx's buffer."""
        obs_np = self._obs_to_numpy(obs, env_idx)
        buf = self._bufs[env_idx]
        buf["org_obs"].append(obs_np)
        ob = {k: obs_np[k] for k in ["color_image1", "color_image2", "robot_state", "parts_poses"]}
        buf["obs"].append(ob)

    # ------------------------------------------------------------------
    # Per-env reset (without resetting all envs)
    # ------------------------------------------------------------------

    def _reset_single_env(self, env_idx: int):
        """Reset env_idx in-place inside the running simulation."""
        sim = self.env.unwrapped
        sim.reset_env(env_idx)
        if sim.ctrl_mode == "osc":
            from isaacgym import gymtorch

            torque_action = torch.zeros_like(sim.dof_pos)
            sim.isaac_gym.set_dof_actuation_force_tensor(sim.sim, gymtorch.unwrap_tensor(torque_action))
        sim.refresh()

    def _configure_episode(self, env_idx: int = 0):
        """Propagate per-episode non-Markovian config to part FSMs for env_idx."""
        if not self.non_markovian:
            return
        for part in self.env.furnitures[env_idx].parts:
            if hasattr(part, "apply_non_markovian_config"):
                part.apply_non_markovian_config()

    # ------------------------------------------------------------------
    # Main collection loop
    # ------------------------------------------------------------------

    def collect(self):
        print("[data collection] Start collecting the data!")

        # Full initial reset of all envs.
        obs = self.env.reset()
        self._reset_all_buffers()
        for env_idx in range(self.num_envs):
            self._configure_episode(env_idx)

        done = torch.zeros(self.num_envs, dtype=torch.bool)

        def num_saved():
            return self.num_success + (self.num_fail if self.save_failure else 0)

        while num_saved() < self.num_demos:
            # --- Handle envs that finished in the previous step ---
            if done.any():
                for env_idx in range(self.num_envs):
                    if not done[env_idx]:
                        continue
                    if num_saved() >= self.num_demos:
                        continue

                    # obs[env_idx] is the terminal obs from the previous env.step().
                    self._store_terminal_obs(env_idx, obs)

                    success = self.env.furnitures[env_idx].all_assembled()
                    collect_enum = CollectEnum.SUCCESS if success else CollectEnum.FAIL

                    if success:
                        self.save(env_idx, collect_enum, {})
                        self.num_success += 1
                        print(f"[env {env_idx}] SUCCESS — Success: {self.num_success}, Fail: {self.num_fail}")
                    else:
                        if self.save_failure:
                            print(f"[env {env_idx}] Saving failure trajectory.")
                            self.save(env_idx, collect_enum, {})
                        else:
                            print(f"[env {env_idx}] Failed to assemble — saving video only, not pickle.")
                            self.save(env_idx, collect_enum, {}, save_pickle=False)
                        self.num_fail += 1
                        print(f"[env {env_idx}] FAIL   — Success: {self.num_success}, Fail: {self.num_fail}")

                    self.traj_counter += 1
                    print(f"[env {env_idx}] Saved {self.traj_counter} trajectories in this run.")

                    self._reset_single_env(env_idx)
                    self._configure_episode(env_idx)
                    self._reset_env_buffer(env_idx)

                # Refresh physics and get updated obs (including post-reset obs for reset envs).
                obs = self.env.unwrapped._get_observation()
                done = torch.zeros(self.num_envs, dtype=torch.bool)

                if num_saved() >= self.num_demos:
                    break

            # --- Compute actions for all envs ---
            noisy_action, clean_action, skill_complete = self.env.get_assembly_action()
            pos_bounds_m = 0.02 if self.env.ctrl_mode == "diffik" else 0.025
            ori_bounds_deg = 15 if self.env.ctrl_mode == "diffik" else 20
            action = scale_scripted_action(
                noisy_action.detach().cpu().clone(),
                pos_bounds_m=pos_bounds_m,
                ori_bounds_deg=ori_bounds_deg,
                device=self.env.device,
            )
            record_action = scale_scripted_action(
                clean_action.detach().cpu().clone(),
                pos_bounds_m=pos_bounds_m,
                ori_bounds_deg=ori_bounds_deg,
                device=self.env.device,
            )

            # --- Step all envs ---
            next_obs, rew, done_new, info = self.env.step(action)

            # --- Store transitions for all active envs ---
            if info["action_success"]:
                for env_idx in range(self.num_envs):
                    obs_np = self._obs_to_numpy(obs, env_idx)
                    rew_val = float(rew[env_idx].squeeze().cpu())
                    skill_val = skill_complete[env_idx] if isinstance(skill_complete, (list, tuple)) else skill_complete
                    rec_act_i = record_action[env_idx]
                    self._store_step(env_idx, obs_np, rec_act_i, rew_val, skill_val)
                    self._verbose_print(f"[env {env_idx}] step={self._bufs[env_idx]['step_counter']}")

            obs = next_obs
            done = done_new.squeeze(-1)

        kind = "total" if self.save_failure else "successful"
        print(
            f"Collected {num_saved()} / {self.num_demos} {kind} trajectories"
            f" (success={self.num_success}, fail={self.num_fail})!"
        )

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------

    def save(self, env_idx: int, collect_enum: CollectEnum, info, save_pickle: bool = True):
        """Save pickle file (and optionally video) of an episode."""
        buf = self._bufs[env_idx]
        print(f"[env {env_idx}] Length of trajectory: {len(buf['obs'])}")

        data = {}
        data["observations"] = buf["obs"]
        data["actions"] = buf["acts"]
        data["rewards"] = buf["rews"]
        data["skills"] = buf["skills"]
        data["success"] = collect_enum == CollectEnum.SUCCESS
        data["furniture"] = self.furniture

        if "error" in info:
            data["error_description"] = info["error"].value
            data["error"] = True
        else:
            data["error"] = False
            data["error_description"] = ""

        demo_path = self.data_path / ("success" if data["success"] else "failure")
        demo_path.mkdir(parents=True, exist_ok=True)

        # Use microsecond resolution to avoid filename collisions across parallel envs.
        timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
        pkl_path = demo_path / f"{timestamp}.pkl"
        if self.compress_pickles:
            pkl_path = pkl_path.with_suffix(".pkl.xz")

        if save_pickle:
            pickle_data(data, pkl_path)
            print(f"[env {env_idx}] Data saved at {pkl_path}")

        video_budget = self.n_video_trials if self.n_video_trials >= 0 else float("inf")
        should_record = self.n_videos_saved < video_budget or (
            self.record_failures and not data["success"]
        )
        if should_record:
            frames = data_to_video(data)
            video_path = (demo_path / timestamp).with_suffix(".mp4")
            create_mp4(frames, video_path)
            self.n_videos_saved += 1
            print(f"[env {env_idx}] Video saved at {video_path}")

    def __del__(self):
        del self.env

        if self.device_interface is not None:
            self.device_interface.close()
