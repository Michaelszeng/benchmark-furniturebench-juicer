import argparse
from typing import List, Union

import cv2
import numpy as np
import torch
import zarr
from ipdb import set_trace as bp
from tqdm import trange

from src.common.control import ControlMode
from src.common.tasks import furniture2idx
from src.common.vision import (
    FrontCameraTransform,
    WristCameraTransform,
)
from src.dataset.normalizer import Normalizer
from src.dataset.zarr import combine_zarr_datasets


def create_sample_indices(
    episode_ends: np.ndarray,
    sequence_length: int,
    pad_before: int = 0,
    pad_after: int = 0,
):
    indices = list()
    for i in range(len(episode_ends)):
        start_idx = 0
        if i > 0:
            start_idx = episode_ends[i - 1]
        end_idx = episode_ends[i]
        episode_length = end_idx - start_idx

        min_start = -pad_before
        max_start = episode_length - sequence_length + pad_after

        # range stops one idx before end
        for idx in range(min_start, max_start + 1):
            buffer_start_idx = max(idx, 0) + start_idx
            buffer_end_idx = min(idx + sequence_length, episode_length) + start_idx
            start_offset = buffer_start_idx - (idx + start_idx)
            end_offset = (idx + sequence_length + start_idx) - buffer_end_idx
            sample_start_idx = 0 + start_offset
            sample_end_idx = sequence_length - end_offset
            indices.append([buffer_start_idx, buffer_end_idx, sample_start_idx, sample_end_idx, i])
    indices = np.array(indices)
    return indices


def sample_sequence(
    train_data,
    sequence_length,
    buffer_start_idx,
    buffer_end_idx,
    sample_start_idx,
    sample_end_idx,
):
    result = dict()
    for key, input_arr in train_data.items():
        sample = input_arr[buffer_start_idx:buffer_end_idx]
        data = sample
        if (sample_start_idx > 0) or (sample_end_idx < sequence_length):
            data = np.zeros(shape=(sequence_length,) + input_arr.shape[1:], dtype=input_arr.dtype)
            if sample_start_idx > 0:
                data[:sample_start_idx] = sample[0]
            if sample_end_idx < sequence_length:
                data[sample_end_idx:] = sample[-1]
            data[sample_start_idx:sample_end_idx] = sample
        result[key] = data
    return result


class FurnitureImageDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        dataset_paths: Union[List[str], str],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        normalizer: Normalizer,
        augment_image: bool = False,
        data_subset: int = None,
        first_action_idx: int = 0,
        control_mode: ControlMode = ControlMode.delta,
        pad_after: bool = True,
        max_episode_count: Union[dict, None] = None,
    ):
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.control_mode = control_mode

        normalizer = normalizer.cpu()

        # Read from zarr dataset
        combined_data, metadata = combine_zarr_datasets(
            dataset_paths,
            [
                "color_image1",
                "color_image2",
                "robot_state",
                f"action/{control_mode}",
                "skill",
            ],
            max_episodes=data_subset,
            max_ep_cnt=max_episode_count,
        )

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends = combined_data["episode_ends"]
        self.metadata = metadata
        print(f"Loading dataset of {len(self.episode_ends)} episodes:")
        for path, data in metadata.items():
            print(f"  {path}: {data['n_episodes_used']} episodes, {data['n_frames_used']}")

        self.train_data = {
            "color_image1": combined_data["color_image1"],
            "color_image2": combined_data["color_image2"],
            "robot_state": combined_data["robot_state"],
            "action": combined_data[f"action/{control_mode}"],
        }

        # Normalize data to [-1,1]
        for key in normalizer.keys():
            self.train_data[key] = normalizer(self.train_data[key], key, forward=True)

        # compute start and end of each state-action sequence
        # also handles padding
        self.indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1 if pad_after else 0,
        )

        # Add image augmentation
        self.augment_image = augment_image
        self.image1_transform = WristCameraTransform(mode="train" if augment_image else "eval")
        self.image2_transform = FrontCameraTransform(mode="train" if augment_image else "eval")

        self.task_idxs = np.array([furniture2idx[f] for f in combined_data["furniture"]])
        self.successes = combined_data["success"].astype(np.uint8)
        self.skills = combined_data["skill"].astype(np.uint8)
        self.failure_idx = combined_data["failure_idx"]

        # Add action and observation dimensions to the dataset
        self.action_dim = self.train_data["action"].shape[-1]
        self.robot_state_dim = self.train_data["robot_state"].shape[-1]

        # Take into account possibility of predicting an action that doesn't align with the first observation
        # TODO: Verify this works with the BC_RNN baseline
        self.first_action_idx = first_action_idx
        if first_action_idx < 0:
            self.first_action_idx = self.obs_horizon + first_action_idx

        self.final_action_idx = self.first_action_idx + self.pred_horizon

        if self.augment_image:
            self.train()
        else:
            self.eval()

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            demo_idx,
        ) = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # Discard unused observations
        nsample["color_image1"] = nsample["color_image1"][: self.obs_horizon, :]
        nsample["color_image2"] = nsample["color_image2"][: self.obs_horizon, :]
        nsample["robot_state"] = torch.from_numpy(nsample["robot_state"][: self.obs_horizon, :])

        # Discard unused actions
        nsample["action"] = torch.from_numpy(nsample["action"][self.first_action_idx : self.final_action_idx, :])

        # Apply the image augmentation
        nsample["color_image1"] = torch.stack(
            [self.image1_transform(img) for img in torch.from_numpy(nsample["color_image1"]).permute(0, 3, 1, 2)]
        ).permute(0, 2, 3, 1)
        nsample["color_image2"] = torch.stack(
            [self.image2_transform(img) for img in torch.from_numpy(nsample["color_image2"]).permute(0, 3, 1, 2)]
        ).permute(0, 2, 3, 1)

        # Add the task index and success flag to the sample
        nsample["task_idx"] = torch.LongTensor([self.task_idxs[demo_idx]])
        nsample["success"] = torch.IntTensor([self.successes[demo_idx]])

        return nsample

    def train(self):
        if self.augment_image:
            self.image1_transform.train()
            self.image2_transform.train()
        else:
            self.eval()

    def eval(self):
        self.image1_transform.eval()
        self.image2_transform.eval()


class FurnitureFeatureDataset(torch.utils.data.Dataset):
    """
    This is the dataset used for precomputed image features.
    """

    def __init__(
        self,
        dataset_paths: Union[List[str], str],
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        normalizer: Normalizer,
        encoder_name: str,
        data_subset: int = None,
        first_action_idx: int = 0,
        control_mode: ControlMode = ControlMode.delta,
    ):
        # Read from zarr dataset
        combined_data = combine_zarr_datasets(
            dataset_paths,
            [
                f"feature/{encoder_name}/feature1",
                f"feature/{encoder_name}/feature2",
                "robot_state",
                f"action/{control_mode}",
            ],
            max_episodes=data_subset,
        )

        # (N, D)
        # Get only the first data_subset episodes
        self.episode_ends = combined_data["episode_ends"]
        print(f"Loading dataset of {len(self.episode_ends)} episodes")

        train_data = {
            "feature1": combined_data[f"feature/{encoder_name}/feature1"],
            "feature2": combined_data[f"feature/{encoder_name}/feature2"],
            "robot_state": combined_data["robot_state"],
            "action": combined_data[f"action/{control_mode}"],
        }

        # compute start and end of each state-action sequence
        # also handles padding
        indices = create_sample_indices(
            episode_ends=self.episode_ends,
            sequence_length=pred_horizon,
            pad_before=obs_horizon - 1,
            pad_after=action_horizon - 1,
        )

        self.task_idxs = np.array([furniture2idx[f] for f in combined_data["furniture"]])
        self.successes = combined_data["success"].astype(np.uint8)

        normalizer: Normalizer = normalizer.cpu()

        # Normalize data to [-1,1]
        for key in normalizer.keys():
            train_data[key] = normalizer(train_data[key], key, forward=True)

        self.indices = indices
        self.train_data = train_data
        self.pred_horizon = pred_horizon
        self.action_horizon = action_horizon
        self.obs_horizon = obs_horizon
        self.feature_dim = train_data["feature1"].shape[-1]

        # Add action and observation dimensions to the dataset
        self.action_dim = train_data["action"].shape[-1]
        self.robot_state_dim = train_data["robot_state"].shape[-1]

        # Take into account possibility of predicting an action that doesn't align with the first observation
        self.first_action_idx = first_action_idx
        if first_action_idx < 0:
            self.first_action_idx = self.obs_horizon + first_action_idx

        self.final_action_idx = self.first_action_idx + self.pred_horizon

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        # get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            demo_idx,
        ) = self.indices[idx]

        # get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        # discard unused observations
        nsample["feature1"] = nsample["feature1"][: self.obs_horizon, :]
        nsample["feature2"] = nsample["feature2"][: self.obs_horizon, :]
        nsample["robot_state"] = nsample["robot_state"][: self.obs_horizon, :]

        # Discard unused actions
        nsample["action"] = nsample["action"][self.first_action_idx : self.final_action_idx, :]

        # Add the task index and success flag to the sample
        nsample["task_idx"] = self.task_idxs[demo_idx]
        nsample["success"] = self.successes[demo_idx]

        # for diffusion policy version (self.first_action_offset = 0)
        # |0|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5| idx
        # |o|o|                             observations:       2
        # | |a|a|a|a|a|a|a|a|               actions executed:   8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16

        # for RNN version (self.first_action_offset = -1) -- meaning the first action aligns with the last observation
        # |0|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5| idx
        # |o|o|o|o|o|o|o|o|o|o|             observations:       2
        # |                 |p|             actions predicted:  1
        # |                 |a|             actions executed:   1

        return nsample

    def train(self):
        pass

    def eval(self):
        pass


class OfflineRLFeatureDataset(FurnitureFeatureDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Also add in rewards and terminal states to the dataset
        self.normalized_train_data["reward"] = self.dataset["reward"][: self.episode_ends[-1]]
        self.normalized_train_data["terminal"] = self.dataset["terminal"][: self.episode_ends[-1]]

    def __getitem__(self, idx):
        # Get the start/end indices for this datapoint
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
        ) = self.indices[idx]

        # Get normalized data using these indices
        nsample = sample_sequence(
            train_data=self.normalized_train_data,
            sequence_length=self.pred_horizon,
            buffer_start_idx=buffer_start_idx,
            buffer_end_idx=buffer_end_idx,
            sample_start_idx=sample_start_idx,
            sample_end_idx=sample_end_idx,
        )

        output = dict(
            action=nsample["action"],
            terminal=int(nsample["terminal"].sum() > 0),
        )

        # Add the current observation to the input
        output["curr_obs"] = dict(
            feature1=nsample["feature1"][: self.obs_horizon, :],
            feature2=nsample["feature2"][: self.obs_horizon, :],
            robot_state=nsample["robot_state"][: self.obs_horizon, :],
        )

        # Add the next obs to the input
        # |0|1|2|3|4|5|6|7|8|9|0|1|2|3|4|5| idx
        # |o|o|                             observations:       2
        # | |a|a|a|a|a|a|a|a|               actions executed:   8
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        # |p|p|p|p|p|p|p|p|p|p|p|p|p|p|p|p| actions predicted: 16
        # | | |r|r|r|r|r|r|r|r|             rewards:            8
        # This is the observation that happens after the self.action_horizon actions have executed
        # Will start at `obs_horizon - 1 + action_horizon - (obs_horizon - 1)`
        # (which simplifies to `action_horizon`)
        # and end at `start + obs_horizon`
        start_idx = self.action_horizon
        end_idx = start_idx + self.obs_horizon
        output["next_obs"] = dict(
            feature1=nsample["feature1"][start_idx:end_idx, :],
            feature2=nsample["feature2"][start_idx:end_idx, :],
            robot_state=nsample["robot_state"][start_idx:end_idx, :],
        )

        # Add the reward to the input
        # What rewards should be counted? The rewards that happen after the first action is executed, up to the last action
        # We sum these into a single reward for the entire sequence
        output["reward"] = nsample["reward"][start_idx:end_idx].sum()

        return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactively view episodes from a processed zarr dataset.")
    parser.add_argument("zarr_path", help="Path to the .zarr dataset file")
    parser.add_argument(
        "--episode",
        "-e",
        type=int,
        default=0,
        help="Episode index to start at (default: 0)",
    )
    parser.add_argument("--fps", type=int, default=10, help="Playback speed in frames per second (default: 10)")
    args = parser.parse_args()

    # src/dataset/zarr.py shadows the installed zarr package when this file is
    # run as a script (Python inserts the script's directory first on sys.path).
    # We must evict the cached local module and re-import with sys.path pruned.
    import importlib
    import sys
    from pathlib import Path as _Path

    _script_dir = str(_Path(__file__).parent.resolve())
    _saved_path = sys.path[:]
    sys.path = [p for p in sys.path if _Path(p).resolve() != _Path(_script_dir).resolve()]
    del sys.modules["zarr"]
    zarr_lib = importlib.import_module("zarr")
    sys.path = _saved_path

    dataset = zarr_lib.open(args.zarr_path, mode="r")
    episode_ends = dataset["episode_ends"][:]
    n_episodes = len(episode_ends)

    print(f"Dataset: {args.zarr_path}")
    print(f"Episodes: {n_episodes}  |  Total frames: {episode_ends[-1]}")
    print()
    print("Controls:")
    print("  k / l     step 1 / 10 frames forward")
    print("  j / h     step 1 / 10 frames backward")
    print("  n / p     next / previous episode")
    print("  Space     toggle play/pause")
    print("  q         quit")
    print()

    ep_idx = max(0, min(args.episode, n_episodes - 1))
    frame_delay_ms = max(1, 1000 // args.fps)
    playing = False

    cv2.namedWindow("Dataset Viewer", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Dataset Viewer", 1280, 480)

    def load_episode_frames(ep):
        start = 0 if ep == 0 else episode_ends[ep - 1]
        end = episode_ends[ep]
        imgs1 = dataset["color_image1"][start:end]  # (T, H, W, 3)
        imgs2 = dataset["color_image2"][start:end]
        success = bool(dataset["success"][ep])
        return np.concatenate([imgs1, imgs2], axis=2), success

    frames, success = load_episode_frames(ep_idx)
    frame_idx = 0

    while True:
        frame = frames[frame_idx]
        # OpenCV expects BGR; zarr images are stored as RGB
        display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        label = (
            f"Ep {ep_idx}/{n_episodes - 1}  |  "
            f"Frame {frame_idx}/{len(frames) - 1}  |  "
            f"{'SUCCESS' if success else 'FAILURE'}"
        )
        cv2.putText(display, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Dataset Viewer", display)

        wait_ms = frame_delay_ms if playing else 0
        key = cv2.waitKey(wait_ms) & 0xFF

        if key == ord("q"):
            break
        elif key == ord(" "):
            playing = not playing
        elif key == ord("k") or (playing and key == 255):
            # advance one frame (also used during auto-play)
            if frame_idx < len(frames) - 1:
                frame_idx += 1
            else:
                playing = False
        elif key == ord("l"):
            frame_idx = min(frame_idx + 10, len(frames) - 1)
        elif key == ord("j"):
            frame_idx = max(frame_idx - 1, 0)
        elif key == ord("h"):
            frame_idx = max(frame_idx - 10, 0)
        elif key == ord("n"):
            ep_idx = min(ep_idx + 1, n_episodes - 1)
            frames, success = load_episode_frames(ep_idx)
            frame_idx = 0
            playing = False
        elif key == ord("p"):
            ep_idx = max(ep_idx - 1, 0)
            frames, success = load_episode_frames(ep_idx)
            frame_idx = 0
            playing = False

    cv2.destroyAllWindows()
