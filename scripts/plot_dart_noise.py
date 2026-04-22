"""Plot state trajectories from pkl.xz files to compare DART noise levels."""

import lzma
import pickle

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

FILES = {
    "0.03125": "/home/michzeng/Downloads/0_03125.pkl.xz",
    "0.0625": "/home/michzeng/Downloads/0.0625.pkl.xz",
    "0.125": "/home/michzeng/Downloads/0_125.pkl.xz",
    "0.25": "/home/michzeng/Downloads/0_25.pkl.xz",
}

COLORS = {"0.03125": "tab:blue", "0.0625": "tab:orange", "0.125": "tab:red", "0.25": "tab:green"}

# Toggle which panels to show
PLOT_EE_POS = False
PLOT_EE_QUAT = False
PLOT_JOINTS = True
PLOT_GRIPPER_ACTIONS = False

SAVE_FIG = False
SAVE_PATH = "/home/michzeng/Downloads/dart_noise_comparison.png"


def load(path):
    with lzma.open(path, "rb") as f:
        return pickle.load(f)


def extract_arrays(data):
    obs = data["observations"]
    ee_pos = np.stack([o["robot_state"]["ee_pos"] for o in obs])  # (T, 3)
    ee_quat = np.stack([o["robot_state"]["ee_quat"] for o in obs])  # (T, 4)
    joints = np.stack([o["robot_state"]["joint_positions"] for o in obs])  # (T, 7)
    gripper = np.stack([o["robot_state"]["gripper_width"] for o in obs])  # (T, 1)
    actions = np.array(data["actions"])  # (T-1, 8)
    return ee_pos, ee_quat, joints, gripper, actions


datasets = {label: extract_arrays(load(path)) for label, path in FILES.items()}

T_min = min(d[0].shape[0] for d in datasets.values())
t = np.arange(T_min)

panels = [PLOT_EE_POS, PLOT_EE_QUAT, PLOT_JOINTS, PLOT_GRIPPER_ACTIONS]
n_panels = sum(panels)
assert n_panels > 0, "Enable at least one panel."

fig = plt.figure(figsize=(18, 4 * n_panels))
fig.suptitle("DART Noise Comparison — state trajectories", fontsize=14)
gs = gridspec.GridSpec(n_panels, 1, hspace=0.45)
panel_idx = 0

if PLOT_EE_POS:
    ax_pos = fig.add_subplot(gs[panel_idx])
    panel_idx += 1
    ax_pos.set_title("End-effector position (x, y, z)")
    ax_pos.set_ylabel("meters")
    if panel_idx == n_panels:
        ax_pos.set_xlabel("timestep")
    for label, (ee_pos, *_) in datasets.items():
        for dim, ls in zip(range(3), ["-", "--", ":"]):
            ax_pos.plot(
                t,
                ee_pos[:T_min, dim],
                color=COLORS[label],
                linestyle=ls,
                alpha=0.8,
                linewidth=0.9,
            )
    handles = [Line2D([0], [0], color=COLORS[lbl], label=f"σ={lbl}") for lbl in FILES]
    handles += [Line2D([0], [0], color="k", linestyle=ls, label=ax) for ls, ax in zip(["-", "--", ":"], "xyz")]
    ax_pos.legend(handles=handles, ncol=6, fontsize=7, loc="upper right")

if PLOT_EE_QUAT:
    ax_quat = fig.add_subplot(gs[panel_idx])
    panel_idx += 1
    ax_quat.set_title("End-effector quaternion (w, x, y, z)")
    ax_quat.set_ylabel("value")
    if panel_idx == n_panels:
        ax_quat.set_xlabel("timestep")
    for label, (_, ee_quat, *_) in datasets.items():
        for dim, ls in zip(range(4), ["-", "--", ":", "-."]):
            ax_quat.plot(t, ee_quat[:T_min, dim], color=COLORS[label], linestyle=ls, alpha=0.8, linewidth=0.9)
    handles = [Line2D([0], [0], color=COLORS[lbl], label=f"σ={lbl}") for lbl in FILES]
    handles += [Line2D([0], [0], color="k", linestyle=ls, label=c) for ls, c in zip(["-", "--", ":", "-."], "wxyz")]
    ax_quat.legend(handles=handles, ncol=6, fontsize=7, loc="upper right")

if PLOT_JOINTS:
    ax_joints = fig.add_subplot(gs[panel_idx])
    panel_idx += 1
    ax_joints.set_title("Joint positions (7 DOF)")
    ax_joints.set_ylabel("radians")
    if panel_idx == n_panels:
        ax_joints.set_xlabel("timestep")
    for label, (_, _, joints, *_) in datasets.items():
        for dim in range(7):
            ax_joints.plot(t, joints[:T_min, dim], color=COLORS[label], alpha=0.55, linewidth=0.8)
    handles = [Line2D([0], [0], color=COLORS[lbl], label=f"σ={lbl}") for lbl in FILES]
    ax_joints.legend(handles=handles, ncol=4, fontsize=7, loc="upper right")

if PLOT_GRIPPER_ACTIONS:
    ax_misc = fig.add_subplot(gs[panel_idx])
    panel_idx += 1
    ax_misc.set_title("Gripper width  &  action ℓ2 norm")
    ax_misc.set_xlabel("timestep")
    ax_misc.set_ylabel("value")
    for label, (_, _, _, gripper, actions) in datasets.items():
        ax_misc.plot(
            t,
            gripper[:T_min, 0],
            color=COLORS[label],
            linestyle="-",
            alpha=0.8,
            linewidth=0.9,
            label=f"σ={label} gripper",
        )
        T_act = min(actions.shape[0], T_min)
        norm = np.linalg.norm(actions[:T_act, :3], axis=1)  # pos-delta norm
        ax_misc.plot(
            np.arange(T_act),
            norm,
            color=COLORS[label],
            linestyle="--",
            alpha=0.6,
            linewidth=0.9,
            label=f"σ={label} |Δpos|",
        )
    handles = [Line2D([0], [0], color=COLORS[lbl], label=f"σ={lbl}") for lbl in FILES]
    handles += [
        Line2D([0], [0], color="k", linestyle="-", label="gripper"),
        Line2D([0], [0], color="k", linestyle="--", label="|Δpos|"),
    ]
    ax_misc.legend(handles=handles, ncol=6, fontsize=7, loc="upper right")

plt.tight_layout()
if SAVE_FIG:
    plt.savefig(SAVE_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved → {SAVE_PATH}")
plt.show()
