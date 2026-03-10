# Architectural Patterns

## 1. Factory Functions

The codebase uses factory functions to select implementations from config strings, keeping
training scripts free of conditional logic.

**get_actor** — `src/behavior/__init__.py:7`
- Takes `(config, normalizer, device)`, reads `config.actor.name` to dispatch to
  `MLPActor`, `RNNActor`, `DiffusionPolicy`, `MultiTaskDiffusionPolicy`, or
  `SuccessGuidedDiffusionPolicy`.
- All returned objects are `Actor` subclasses.

**get_env** — `src/gym/__init__.py`
- Wraps `FurnitureSimEnv` with the appropriate task/randomness config.

Usage pattern in `src/train/bc.py`: construct normalizer → call `get_actor(config, normalizer, device)`.

---

## 2. Actor Class Hierarchy with PostInitCaller Metaclass

**Definition:** `src/behavior/base.py:19-43`

```python
class PostInitCaller(type(torch.nn.Module)):
    def __call__(cls, *args, **kwargs):
        obj = super().__call__(*args, **kwargs)
        obj.__post_init__(*args, **kwargs)
        return obj

class Actor(torch.nn.Module, metaclass=PostInitCaller):
    ...
```

The metaclass ensures `__post_init__` runs after `__init__` completes. Subclasses
(`MLPActor`, `RNNActor`, `DiffusionPolicy`) use `__post_init__` to run setup that
requires all `__init__` state to be set (e.g., loading pretrained encoder weights,
setting up noise schedulers).

**`_normalized_obs`** method on `Actor` (base.py) applies the normalizer to the
observation dict before passing it to the policy network.

---

## 3. Dual-Camera Vision Pipeline

Every `Actor` instance holds two camera transforms and two encoder pairs:

```python
# src/behavior/base.py:41-48
camera1_transform = WristCameraTransform(mode="eval")   # wrist cam
camera2_transform = FrontCameraTransform(mode="eval")   # front cam
encoder1: nn.Module        # encodes wrist image
encoder1_proj: nn.Module   # projects to encoding_dim
encoder2: nn.Module        # encodes front image
encoder2_proj: nn.Module   # projects to encoding_dim
```

Both encoded features are concatenated and fed to the policy head.
Calling `actor.train()` / `actor.eval()` propagates through to the transforms
(see `WristCameraTransform.train` at `src/common/vision.py:71`).

---

## 4. Camera-Specific Transforms

**`FrontCameraTransform`** — `src/common/vision.py:11`
- Input: `(240, 320)` frames from the front-facing camera.
- Train: ColorJitter + GaussianBlur + CenterCrop to strip 20px side margins → RandomCrop to `(224, 224)`.
- Eval: CenterCrop to `(224, 224)`.

**`WristCameraTransform`** — `src/common/vision.py:49`
- Input: variable-size frames from the wrist-mounted camera.
- Train: ColorJitter + GaussianBlur + Resize to `(224, 224)`.
- Eval: Resize to `(224, 224)`.

Both classes override `.train(mode)` and `.eval()` to sync `self.mode` with the
PyTorch train/eval flag, so the forward pass automatically applies the right pipeline.

---

## 5. Decoupled Normalizer Design

**Base class:** `src/dataset/normalizer.py:10`
- Inherits `nn.Module`; stores stats as `nn.ParameterDict` (non-trainable).
- `forward(x, key, forward=True)` handles numpy↔tensor conversion transparently.
- `_normalize` / `_denormalize` are abstract — subclasses implement these.

**`LinearNormalizer`** — `src/dataset/normalizer.py:119`
- Scales to `[-1, 1]` using min/max stats from `get_data_stats()`.
- `_normalize`: `2 * (x - min) / (max - min) - 1`
- `_denormalize`: inverse of above.

**`GaussianNormalizer`** — `src/dataset/normalizer.py:146`
- Z-score normalization using mean/std stats.

The action key is remapped: whichever of `action/pos` or `action/delta` matches
`control_mode` is stored as `"action"` in the stats dict.

Normalizers are constructed in `src/train/bc.py` and passed into `get_actor()`.

---

## 6. Data Path Convention

All path construction goes through `src/common/files.py`.

**`get_processed_path`** — `src/common/files.py:27`
```
$DATA_DIR_PROCESSED/processed/{env}/{task}/{source}/{randomness}/{outcome}.zarr
```
Accepts lists for any dimension; lists are joined with `-` and sorted alphabetically
(via `add_subdir` at files.py:16). This allows mixing e.g. multiple randomness levels.

**`get_processed_paths`** — `src/common/files.py:57`
Glob-based version that returns multiple matching paths (for wildcard-style loading).

**Raw data** (not processed): `$DATA_DIR_RAW/raw/{env}/{task}/{source}/{randomness}/`
(functions in the same file, ~line 118 and 161).

Never construct data paths by hand — always use these functions to ensure consistency.

---

## 7. Hydra Config Composition

Entry point: `src/train/bc.py` uses `@hydra.main(config_path="../config", config_name="base")`.

**Base config:** `src/config/base.yaml` — defines top-level structure. `furniture` key
is `???` (required, must be provided on CLI).

**Config groups:**
- `actor/` — mlp.yaml, rnn.yaml, diffusion.yaml
- `vision_encoder/` — resnet.yaml, dino.yaml, mae.yaml, vip.yaml, etc.
- `experiment/` — pre-composed combinations (e.g., `image_baseline`)

**Override pattern:**
```
python -m src.train.bc +experiment=image_baseline furniture=one_leg actor.lr=1e-4
```

`+experiment=` appends the experiment group defaults on top of base config.
All config values are accessible via `config.key.subkey` in code.

**Dry-run:** `dryrun=true` flag in base.yaml short-circuits training for fast config validation.

---

## 8. Raw pkl → Zarr Data Pipeline

Raw trajectories are `Trajectory` TypedDicts (see `src/common/types.py:13`) serialized
as `.pkl` files under `$DATA_DIR_RAW`.

Processing stages (each is a separate script in `src/data_processing/`):

1. **`process_pickles.py`** — Loads raw pkls, filters by success/failure, extracts
   observations + actions, writes intermediate zarr.
2. **`encode_in_bulk.py`** — Runs vision encoder over stored raw images to cache
   image features (avoids re-encoding during training).
3. **`combine_zarr.py`** — Merges multiple zarr datasets (e.g., different randomness
   levels) into one.
4. **`augment_dataset.py`** — Applies backward augmentation trajectories to the zarr.

Final zarr layout:
```
{outcome}.zarr/
├── observations/
│   ├── color_image1    # (N, H, W, 3) uint8
│   ├── color_image2    # (N, H, W, 3) uint8
│   └── robot_state     # (N, robot_state_dim) float32
└── actions             # (N, action_dim) float32
```

`FurnitureImageDataset` (`src/dataset/`) reads from zarr with a sliding window to
produce `(obs_horizon, action_horizon)` samples.

---

## 9. FixedStepsDataloader

**Definition:** `src/dataset/dataloader.py:5`

```python
class FixedStepsDataloader(torch.utils.data.DataLoader):
    def __init__(self, *args, n_batches, **kwargs): ...
    def __iter__(self):
        endless_dataloader = itertools.cycle(super().__iter__())
        for _ in range(self.n_batches):
            yield next(endless_dataloader)
    def __len__(self):
        return self.n_batches
```

Yields exactly `n_batches` batches per "epoch" regardless of dataset size. If the
dataset is smaller than `n_batches`, it cycles (with reshuffled order each cycle).
This decouples "steps per epoch" from dataset size, which is important for small
demonstration datasets that would otherwise give very short epochs.

---

## 10. Rotation Representation Convention

**Storage (raw pkl):** IsaacGym quaternions `(x, y, z, w)`.

**PyTorch3D:** expects `(w, x, y, z)`. Conversion via `isaac_quat_to_pytorch3d_quat`
at `src/common/geometry.py:8`.

**Policy input/output:** 6D rotation representation (first two columns of rotation
matrix). Conversion chain:
```
Isaac quat (x,y,z,w) → PyTorch3D quat (w,x,y,z) → rotation matrix → 6D
```
via `isaac_quat_to_rot_6d` at `src/common/geometry.py:24`.

The `proprioceptive_quat_to_6d_rotation` function (imported in `src/behavior/base.py:8`)
applies this conversion to the robot state inside `_normalized_obs`.

**Why 6D?** The 6D representation is continuous and avoids gimbal lock and quaternion
double-cover issues, improving learning stability. See Zhou et al. 2019.
