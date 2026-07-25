# FurnitureBench: Furniture Assembly Benchmark

This repo is a fork of the [FurnitureBench](https://github.com/clvrai/furniture-bench) (specifically, a fork of it: [JUICER](https://github.com/ankile/imitation-juicer)).

This fork contains unique features:
 - Scripted Markovian expert for automated data collection on the `one_leg` task
 - Scripted Non-Markovian expert for automated data collection on the `one_leg` task
 - Diffusion Policy evaluation pipeline for policies trained using [diffusion-policy-experiments](https://github.com/Michaelszeng/diffusion-policy-experiments)
 - Semi-automated Human-Gated-DAgger pipeline (including failure collection, manual intervention timestep annotation, and correction collectino) for iterative training of policies


## Installation


### Install Conda

First, install Conda by following the instructions on the [Conda website](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) or the [Miniconda website](https://docs.conda.io/en/latest/miniconda.html) (here using Miniconda).

```bash
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

```bash
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

To activate the changes, restart your shell or run:

```bash
source ~/.bashrc
source ~/.zshrc
```

### Create a Conda Environment

Create a new Conda environment by running:

```bash
conda create -n imitation-juicer python=3.8 -y
```

Activate the environment by running:

```bash
conda activate imitation-juicer
```

Once installed and activated, make some compatibility changes to the environment by running:

```bash
pip install setuptools==65.5.0
# IMPORTANT: Keep pip < 24.1 because `gym==0.21.0` (required by `furniture-bench`)
# has metadata that newer pip versions reject.
pip install "pip==24.0" wheel==0.38.4
pip install termcolor
```


### Install IsaacGym

Download the IsaacGym installer from the [IsaacGym website](https://developer.nvidia.com/isaac-gym) and follow the instructions to download the package by running (also refer to the [FurnitureBench installlation instructions](https://clvrai.github.io/furniture-bench/docs/getting_started/installing_furniture_sim.html#download-isaac-gym)):

- Click "Join now" and log into your NVIDIA account.
- Click "Member area".
- Read and check the box for the license agreement.
- Download and unzip `Isaac Gym - Ubuntu Linux 18.04 / 20.04 Preview 4 release`.

Once the zipped file is downloaded, move it to the desired location and unzip it by running:

```bash
tar -xzf IsaacGym_Preview_4_Package.tar.gz
```


Now, you can install the IsaacGym package by navigating to the `isaacgym` directory and running:

```bash
pip install -e python --no-cache-dir --force-reinstall
```

_Note: The `--no-cache-dir` and `--force-reinstall` flags are used to avoid potential issues with the installation we encountered._

_Note: Please ignore Pip's notice that `[notice] To update, run: pip install --upgrade pip`. This codebase depends on `furniture-bench`, which pins `gym==0.21.0`, and that package requires `pip<24.1` due to legacy metadata._

_Tip: The documentation for IsaacGym  is located inside the `docs` directory in the unzipped folder and is not available online. You can open the `index.html` file in your browser to access the documentation._

You can now safely delete the downloaded zipped file and navigate back to the root directory for your project. 


### Install FurnitureBench

To allow for data collection with the SpaceMouse, etc. we used a [custom fork](https://github.com/Michaelszeng/furniture-bench/tree/iros-2024-release-v1) of the [FurnitureBench code](https://github.com/clvrai/furniture-bench). The fork is included in this codebase as a submodule. To install the FurnitureBench package, first run:

```bash
git clone --recursive git@github.com:ankile/imitation-juicer.git
```

_Note: If you forgot to clone the submodule, you can run `git submodule update --init --recursive` to fetch the submodule._

Then, install the FurnitureBench package by running:

```bash
cd imitation-juicer/furniture-bench
pip install -e .
```

To test the installation of FurnitureBench, run:

```bash
python -m furniture_bench.scripts.run_sim_env --furniture one_leg --scripted
```

This should open a window with the simulated environment and the robot in it.

If you encounter the error `ImportError: libpython3.8.so.1.0: cannot open shared object file: No such file or directory`, this might be remedied by adding the conda environment's library path to the `LD_LIBRARY_PATH` environment variable. This can be done by, e.g., running:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
```

To make this persistent (recommended), add a conda activation hook:

```bash
mkdir -p "$CONDA_PREFIX/etc/conda/activate.d" "$CONDA_PREFIX/etc/conda/deactivate.d"

cat > "$CONDA_PREFIX/etc/conda/activate.d/isaacgym.sh" <<'EOF'
export _JUICER_OLD_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
EOF

cat > "$CONDA_PREFIX/etc/conda/deactivate.d/isaacgym.sh" <<'EOF'
export LD_LIBRARY_PATH="${_JUICER_OLD_LD_LIBRARY_PATH:-}"
unset _JUICER_OLD_LD_LIBRARY_PATH
EOF
```

### Install the ImitationJuicer Package

Finally, install the ImitationJuicer package by running:

```bash
cd ..
pip install -e .
```

### Data Collection: Install the SpaceMouse Driver

To make data collection with the SpaceMouse possible, you need to install the SpaceMouse driver: 

```bash
sudo apt install libspnav-dev spacenavd
```

Then, the SpaceMouse driver needs to run in the background. To start the driver, run:

```bash
sudo systemctl start spacenavd
```





## Teleop Data Collection

Note: this repo uses the [Diffusion Policy](https://arxiv.org/abs/2303.04137v4) zarr format for datasets. 

### Published Datasets

The [JUICER](https://imitation-juicer.github.io/) paper provides a dataset of 50 demonstrations. Download data, place into a directory called `./dataset`: https://drive.google.com/drive/folders/13UqtMLXY1_8JCQOZf3j-YbZyMRTsgZ2K

Truncate each episode so they do not contain data after success is achieved:
```bash
python src/data_processing/truncate_at_success.py dataset/imitation-juicer-data-processed-001/processed/sim/one_leg/teleop/low/success.zarr --output dataset/imitation-juicer-data-processed-001/processed/sim/one_leg/teleop/low/success_truncated.zarr
```

Convert to Training `zarr` Format:
```bash
python src/data_processing/process_zarr.py dataset/imitation-juicer-data-processed-001/processed/sim/one_leg/teleop/low/success_truncated.zarr --output dataset/imitation-juicer-data-processed-001/processed/sim/one_leg/teleop/low/success_truncated_translated.zarr
```

### Manual Collection

This requires possession of a Spacemouse, i.e. the 3Dconnexion SpaceMouse Wireless 3DX-700043.

```bash
python src/data_collection/teleop.py --furniture one_leg --num-demos 200 --randomness low
```

To collect data, control the robot with the SpaceMouse. To discard an episode and reset the environment, press `n`. To \"undo\" actions, press `b`. To toggle recording on and off, use `c` and `p`, respectively. Note that the environment automatically resets upon success.

Optionally add the flag `--save-failure` to also store failed trajectories, and add `--no-ee-laser` to remove the red laser from the end-effector from the viewer (it's not rendered in the camera views either way).

Demonstrations are saved as `.pkl` files at:
```bash
./dataset/raw/sim/one_leg/teleop/low/success/
```

To post-process the `.pkl` files into a `.zarr` dataset:

```bash
python src/data_processing/process_pickles.py --env sim --furniture one_leg --source teleop --randomness low --demo-outcome success
```

Then, convert to the training `zarr` format (matching the standard diffusion policy zarr format):

```bash
python src/data_processing/process_zarr.py dataset/processed/sim/one_leg/teleop/low/success.zarr --output dataset/processed/sim/one_leg/teleop/low/success_translated.zarr
```

For debugging, view the dataset using:

```bash
python src/visualization/visualize_dataset.py dataset/processed/sim/one_leg/teleop/low/success_translated.zarr
```

### Puppeteering (For Debugging)

This related helper script also exists; this is not used for data collection, but can be used to puppeteer/set absolute poses of the robot and of parts in the scene:

```bash
python src/data_collection/puppeteer.py -f "one_leg"
```





## Automated Data Collection using Scripted Experts

We provide automated data collection with two kinds of experts: 
1. Markovian (technically 2-Markovian) expert: implemented as a finite-state machine (FSM) that determines its state using the current and previous environment states.
2. Non-Markovian Expert: constructed by injecting a variety of non-Markovian behaviors into the deterministic 2-Markovian expert, including a hidden latent plan (consisting of episodically pre-determined waypoint offsets), sticky FSM state transitions (non-deterministically transitioning to the next FSM state), and latent-count alignment maneuvers (noising the trajectory for a fixed amount of time/iterations before picking or inserting the leg). These injections are designed to mimic human behaviors (e.g. pauses, delay in mentally registering subtask completion, sub-optimal or cyclic alignment motions). Other non-Markovian injections are also available but disabled by default.

To run automated data collection, use `src/data_collection/collect_scripted.sh`, which accepts the following positional arguments (all optional):

```bash
./src/data_collection/collect_scripted.sh [dart_amount] [suffix] [non_markovian] [n_demos]
```

- `dart_amount` (default `0.0`): scale factor for target/action noise injected by the expert (0.0 = no noise, 1.0 = default noise, 2.0 = double noise). Used to implement noise injection augmentation similar to (https://arxiv.org/abs/2507.09061).
- `suffix` (default `""`): appended to the `scripted` output directory name (e.g. `v2` → `scripted_v2`), useful for keeping multiple collection runs separate.
- `non_markovian` (default `False`): if `True`, uses the non-Markovian expert (see above) instead of the Markovian expert.
- `n_demos` (default `200`): number of successful demos to collect.

After collection finishes, the script automatically post-processes the resulting pickles into zarr format (and applies translation augmentation), guarded by a lock so only one parallel job performs post-processing.






## Evaluating a Diffusion Policy

This repo contains a policy evaluation pipeline designed for policies trained using [diffusion-policy-experiments](https://github.com/Michaelszeng/diffusion-policy-experiments), but can be easily adapted for other training pipelines.

Note that this repo contains its own imitation-learning training pipeline in `./src/training`, though this is untested and un-integrated with this evaluation pipeline.

The following one-time setup is required for compatibility with [diffusion-policy-experiments](https://github.com/Michaelszeng/diffusion-policy-experiments):

```bash
pip install dill==0.3.5.1
echo "/path/to/diffusion-policy-experiments" \
    > "$CONDA_PREFIX/lib/python3.8/site-packages/diffusion_policy.pth"
pip install robomimic --no-deps
pip install einops==0.4.1
pip install pandas
pip install accelerate==0.13.2
```

After setup, verify with:
```bash
python -c "import diffusion_policy; print('OK')"
```

To quickly test a checkpoint:
```bash
python -m src.eval.evaluate_model_custom \
    --checkpoint /path/to/checkpoint.ckpt \
    --furniture "one_leg" \
    --n-rollouts 10 \
    --n-envs 1 \
```

Use `src/eval/evaluate_checkpoints.sh` for a more thorough evaluation pipeline. This script is a wrapper around `src/eval/evaluate_model_custom.py` that runs rollouts for a single action horizon, for every checkpoint found under a given path (a single `.ckpt` file, or a directory containing multiple `.ckpt` files):

```bash
./src/eval/evaluate_checkpoints.sh <checkpoint_or_dir> [furniture] <n_action_steps> [n_envs] [n_video_trials] [n_rollouts] [--debug] [--resume] [--task-timeout N]
```

- `checkpoint_or_dir` (required): path to a single `.ckpt` file, or a directory containing one or more `.ckpt` files (all files except `latest.ckpt` are evaluated).
- `furniture` (default `one_leg`): furniture task to evaluate on.
- `n_action_steps` (required): action horizon (number of actions executed per policy inference) to evaluate with.
- `n_envs` (default `1`): number of parallel simulation environments to run rollouts in.
- `n_video_trials` (default `0`): the first `n_video_trials` trials will be recorded as MP4 files (set to `-1` to save all).
- `n_rollouts` (default `500`): total number of rollouts to run.
- `--debug` (optional flag): overrides the above to `n_envs=1 n_rollouts=1 n_video_trials=0` and disables headless mode, for fast iteration with a visible sim window.
- `--resume` (optional flag): skips checkpoints that already have results in their output directory, so an interrupted run can be continued.
- `--task-timeout N` (optional): overrides the max number of steps per rollout (default comes from the sim config).


Results for each checkpoint are written to `outputs/<experiment_name>/T_a_<n_action_steps>/<checkpoint_stem>/`, where `<experiment_name>` is derived from the directory name that precedes `checkpoints/` in the given checkpoint path.

Note: there are various scripts in `./plotting_scripts` that may be helpful for processing the evaluation results.






## Semi-Automated HG-DAgger Pipeline

HG-DAgger (Human-Gated DAgger) is an interactive imitation learning method where a human supervisor monitors a trained policy's rollouts, intervenes right before it starts to fail, and provides corrective demonstrations from that point onward. 

This section implements a semi-automated version of that loop: the HG-DAgger pipeline is a three-step process for collecting failures, annotating them, and gathering corrections:

1. **Collect Failures:**
   Launch parallel jobs to collect policy failures. The final job automatically renders MP4 previews of the failures.
   ```bash
   bash scripts/launch_dagger_collect_failures.sh <num_parallel_processes> <dagger_iter> <action_horizon> <non_markovian>
   ```

2. **Label Gates:**
   Watch the generated MP4 previews and label the "gate" frame for each failure. The gate frame should be the first timestep where the policy deviates from a nominal trajectory in a way that ultimately leads to task failure.
   ```bash
   python src/dagger/dagger_label_gates.py <path_to_failure_dir>
   ```
   The labeler appends each gate as an extra xz stream onto the `.pkl.xz` file, and uses the heuristic `pkl.mtime > preview.mtime` to detect which failures are still unlabeled. To relabel a failure (e.g. after making a mistake), strip its appended gate stream back off with:
   ```bash
   python src/dagger/dagger_unlabel_gates.py <path_to_failure_dir> [--dry-run]
   ```
   This truncates the `.pkl.xz` back to just the original pickle and bumps the matching `.preview.mp4`'s mtime so `dagger_label_gates.py` re-classifies it as unlabeled on the next run. Use `--dry-run` to preview what would be stripped without modifying any files.

3. **Collect Corrections:**
   Launch parallel jobs to collect expert corrections starting from the labeled gates. The final job automatically processes the new data into a `.zarr` dataset.
   ```bash
   bash scripts/launch_dagger_collect_corrections.sh <num_parallel_processes> <dagger_iter> <action_horizon> <non_markovian>
   ```

The outputted corrections `.zarr` should then be included in the next training run to fine-tune the policy on the HG-DAgger corrections. Repeat this process for multiple iterations to steadily improve the policy.

Note that this pipeline is currently designed for use on a SLURM cluster, but may be adapted for other use.





## Citation

If you find the paper or the code useful, please consider citing:

```tex      
TODO
```

```tex      
@misc{ankile2024juicer,
      title={JUICER: Data-Efficient Imitation Learning for Robotic Assembly}, 
      author={Lars Ankile and Anthony Simeonov and Idan Shenfeld and Pulkit Agrawal},
      year={2024},
      eprint={2404.03729},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```
