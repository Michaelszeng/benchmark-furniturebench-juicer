# JUICER

Data-efficient imitation learning for robotic furniture assembly using IsaacGym simulation
and the FurnitureBench environment. Trains behavior cloning policies from human teleoperation
demonstrations, with trajectory augmentation at bottleneck states.

## Tech Stack

- Python 3.8, PyTorch 2.2.1
- Hydra 1.3.2 + OmegaConf for config management
- WandB for experiment tracking and model checkpointing
- IsaacGym (physics simulation) + FurnitureBench (custom fork, git submodule at furniture-bench/)
- Zarr for processed dataset storage; Pickle for raw trajectory data
- Diffusers library for diffusion policy noise schedulers

## Key Directories

```
src/
├── behavior/       # Policy classes: Actor base + MLPActor, RNNActor, DiffusionPolicy
├── common/         # Shared utilities: files.py, geometry.py, types.py, vision.py, tasks.py
├── config/         # Hydra YAML configs: base.yaml + actor/, vision_encoder/, experiment/ groups
├── data_collection/ # Teleoperation, scripted collection, backward augmentation
├── data_processing/ # Raw pkl -> processed zarr pipeline (process, encode, combine, augment)
├── dataset/        # FurnitureImageDataset, normalizers, FixedStepsDataloader
├── eval/           # evaluate_model.py (WandB run evaluation), rollout.py
├── gym/            # get_env() factory wrapping FurnitureSimEnv
├── models/         # Neural network modules: transformer, unet, vision encoders
└── train/          # bc.py (main training entry point), continue_bc.py, train_value.py
furniture-bench/    # Git submodule: FurnitureBench environment
sweeps/             # WandB hyperparameter sweep configs
slurm/              # SLURM job submission files
scripts/            # Shell scripts for batch evaluation
```

## Environment Variables

- `DATA_DIR_RAW` — root for raw pickle trajectories: `raw/{env}/{task}/{source}/{randomness}/`
- `DATA_DIR_PROCESSED` — root for processed zarr datasets: `processed/{env}/{task}/{source}/{randomness}/{outcome}.zarr`

## Essential Commands

**Install:**
```
pip install -e .
# Also install IsaacGym and initialize the furniture-bench submodule
```

**Train a policy:**
```
python -m src.train.bc +experiment=image_baseline furniture=one_leg
```

**Evaluate a trained run:**
```
python -m src.eval.evaluate_model --run-id <wandb-run-id> --furniture one_leg \
    --n-envs 10 --n-rollouts 10
```

**Collect teleoperation demonstrations:**
```
python -m src.data_collection.teleop --furniture one_leg --num-demos 10
```

**Run backward augmentation on collected data:**
```
python -m src.data_collection.backward_augment --furniture one_leg --randomness low
```

**Process raw pickles to zarr:**
```
python -m src.data_processing.process_pickles
```

**Makefile shortcuts:**
```
make requirements   # Install dependencies
make lint           # Run flake8 on src/
```

## Furniture Tasks

Valid task names: one_leg, lamp, round_table, desk, square_table, cabinet, chair, stool.
Defined as a Literal type at src/common/types.py:43.

## Config System

Hydra composes configs from groups. Base config is src/config/base.yaml. Override groups
are actor/ (mlp, rnn, diffusion), vision_encoder/ (resnet, dino, mae, vip, etc.), and
experiment/ (pre-built combinations). The furniture= argument is always required at the
command line (marked ??? in base.yaml).

Dry-run mode for fast iteration: append `dryrun=true` to any training command.

## Additional Documentation

- `.claude/docs/architectural_patterns.md` — factory functions, Actor class hierarchy,
  dual-camera pipeline, normalizer design, Hydra config composition, data path convention,
  rotation representation, and the raw pkl→zarr data pipeline
