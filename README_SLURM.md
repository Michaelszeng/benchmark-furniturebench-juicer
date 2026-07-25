This repo supports usage on a SLURM cluster for efficient automated data collection and large-scale evaluation.

## Installation on SLURM Clusters

Follow the same installation steps as in `README.md` except, for the `Install IsaacGym` step, perform this on a device with browser access and `scp`/`rsync` the resulting directory to the cluster.


## Usage on SLURM Clusters

Each pipeline below has a `.sbatch` counterpart to the corresponding shell/Python entrypoint described above. All of the `.sbatch` files share a common structure: they `cd` into `$SLURM_SUBMIT_DIR`, activate the `imitation-juicer` conda environment, set up IsaacGym-related environment variables (`PYTHONPATH`, `LD_LIBRARY_PATH`, `VK_ICD_FILENAMES`, etc.), and then invoke the underlying script. Before use, edit the placeholder `#SBATCH --account=<>`, `--partition=<>`, `--exclude=<>`, and `--qos=<>` fields to match your cluster.

The conda/`HOME`/`LD_LIBRARY_PATH` setup in each `.sbatch` file assumes your miniconda and mamba installs live under a single root directory, `$CLUSTER_USER_HOME` (defaulting to the submitting shell's `$HOME`). If your cluster's compute nodes don't share your login `$HOME` (e.g. it's on a different, non-mounted filesystem), override it when submitting instead of editing the scripts, e.g.:

```bash
CLUSTER_USER_HOME=/data/locomotion/michzeng sbatch src/data_collection/submit_collect_scripted.sbatch
```

### Automated Data Collection using Scripted Experts

`src/data_collection/submit_collect_scripted.sbatch` is a SLURM wrapper around `collect_scripted.sh` and takes the same positional arguments:

```bash
sbatch src/data_collection/submit_collect_scripted.sbatch [dart_amount] [suffix] [non_markovian] [n_demos]
```

Submit multiple jobs with the same arguments to collect data in parallel — the post-processing step in `collect_scripted.sh` is lock-guarded so only the first job to finish converts the collected pickles to zarr.

To run several data collection runs at once over several `dart_amount` values (each submitted as its own job, with `non_markovian` fixed and the `suffix` auto-derived), use:

```bash
./src/data_collection/submit_dart_sweep.sh
```

Edit the `DART_SUFFIXES` associative array in that script to change which `dart_amount` values are swept.


### Evaluating a Diffusion Policy

`src/eval/submit_evaluate_checkpoints.sbatch` is a SLURM wrapper around `evaluate_checkpoints.sh`. Unlike the other `.sbatch` files, most of its parameters (`CHECKPOINT_PATH`, `FURNITURE`, `N_ENVS`, `N_ROLLOUTS`, `N_VIDEO_TRIALS`) are hardcoded constants at the top of the file rather than CLI arguments — edit these directly before submitting. Only the action horizon is passed on the command line:

```bash
sbatch src/eval/submit_evaluate_checkpoints.sbatch <action_horizon>
```

`TASK_TIMEOUT` and `OUTPUT_PREFIX` are auto-derived from `CHECKPOINT_PATH` (e.g. paths containing `teleop` or `non_markovian` get a longer timeout; paths under `furniture_bench_context_ablation` get an `outputs/context_ablation` prefix), and `--resume` is always passed so an interrupted evaluation can be safely resubmitted.

To evaluate the same checkpoint(s) across a sweep of action horizons, submit one job per horizon with:

```bash
./src/eval/batch_submit.sh
```

Edit the `T_a` list in that script to change which action horizons are evaluated.

### Semi-Automated HG-DAgger Pipeline

As explained in `README.md`, the HG-DAgger instructions in `README.md` already apply to SLURM clusters.