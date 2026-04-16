import argparse
import os

if "DATA_DIR_RAW" not in os.environ:
    os.environ["DATA_DIR_RAW"] = "dataset"

import furniture_bench  # noqa: F401

from src.common.files import trajectory_save_dir
from src.data_collection.data_collector import DataCollector

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--randomness", "-r", type=str, default="low")
    parser.add_argument("--num-demos", "-n", type=int, default=100)
    parser.add_argument("--gpu-id", "-g", type=int, default=0)
    parser.add_argument("--num-envs", "-e", type=int, default=1, help="Number of parallel Isaac Gym environments.")
    # parser.add_argument("--resize-sim-img", action="store_true")
    parser.add_argument("--furniture", "-f", type=str, required=True)
    parser.add_argument("--save-failure", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--non-markovian", action="store_true", help="Activate non-Markovian expert policy")
    parser.add_argument(
        "--output-dir-suffix",
        type=str,
        default=None,
        help="Suffix appended to 'scripted' in the output path (e.g. 'v2' → scripted_v2).",
    )
    parser.add_argument(
        "--n-video-trials",
        type=int,
        default=0,
        help="Save videos for the first N trials (default: 0). Set to -1 to save all.",
    )
    parser.add_argument(
        "--record-failures",
        action="store_true",
        default=False,
        help="If set, also save videos of all failed trials beyond --n-video-trials.",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable all target and action noise in the scripted policy.",
    )
    parser.add_argument(
        "--dart-amount",
        type=float,
        default=1.0,
        help="Scale factor for all target and action noise (1.0 = default noise, 0.0 = no noise, 2.0 = double noise).",
    )

    args = parser.parse_args()

    # TODO: Consider what we do with images of full size and if that's needed
    # For now, we assume that images are stored in 224x224 and we know that as`image`
    # # Add the suffix _highres if we are not resizing images in or after simulation
    # if not args.resize_img_after_sim and not args.small_sim_img_size:
    #     obs_type = obs_type + "_highres"
    resize_sim_img = False

    data_path = trajectory_save_dir(
        environment="sim",
        task=args.furniture,
        demo_source="scripted" if not args.output_dir_suffix else f"scripted_{args.output_dir_suffix}",
        randomness=args.randomness,
    )

    print(f"Saving data to directory: {data_path}")

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
        resize_sim_img=resize_sim_img,
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
    )

    collector.collect()
