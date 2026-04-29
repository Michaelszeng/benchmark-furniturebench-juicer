"""Visualise the non-Markovian virtual-target biased random walk (position + rotation).

Run with:
    python scripts/plot_nm_virtual_target.py

Tune the parameters at the top of the file to explore different behaviours.
No Isaac Gym required — pure numpy/scipy/matplotlib.
"""

from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3-d projection)
from scipy.spatial.transform import Rotation

# ── Simulation settings ───────────────────────────────────────────────────────
N_TRAJECTORIES = 10  # overlaid sample trajectories
N_STEPS = 30  # steps per trajectory (sim ≈ 60 Hz; typical state lasts 30-100 steps)
SEED = 0  # set to None for a different result each run

# ── Position virtual-target dynamics ─────────────────────────────────────────
ALPHA_POS = 0.5  # velocity momentum (0 = memoryless, →1 = heavy momentum)
K_POS = 0.12  # spring constant toward goal (fraction of remaining distance / step)
SIGMA_POS = 0.006  # per-step position noise std (m) at full distance

# Fraction of total start→goal distance below which noise and momentum ramp down to zero.
NOISE_FALLOFF_FRAC_POS = 0.8

# ── Rotation virtual-target dynamics ─────────────────────────────────────────
ALPHA_ORI = 0.90  # angular-velocity momentum
K_ORI = 0.1  # rotational spring constant
SIGMA_ORI = np.radians(3.0)  # per-step orientation noise std (rad) at full angle error

# Same idea for orientation: fraction of initial angle error.
NOISE_FALLOFF_FRAC_ORI = 0.6

# ── Example start / goal (robot-frame coordinates, one_leg task geometry) ────
# Start: staging position used by match_leg_ori / lift_up
START_POS = np.array([0.45, 0.15, 0.14])
# Goal: above table hole (reach_table_top_xy target)
GOAL_POS = np.array([0.65, 0.14, 0.14])

# Start / goal orientations  (insert_ori = Rx(π) @ Ry(-36°))
START_ORI = Rotation.from_euler("xyz", [np.pi, np.radians(-36), 0]).as_matrix()
# Set GOAL_ORI = START_ORI to test position-only; add a rotation offset to test orientation noise.
GOAL_ORI = Rotation.from_euler("xyz", [np.pi, np.radians(-36), np.radians(90)]).as_matrix()
# ─────────────────────────────────────────────────────────────────────────────


def _rotvec_to_goal(R_from: np.ndarray, R_to: np.ndarray) -> np.ndarray:
    """Rotation vector (axis-angle) that brings R_from to R_to."""
    return Rotation.from_matrix(R_to @ R_from.T).as_rotvec()


def step_pos(
    vt_pos: np.ndarray,
    vt_vel: np.ndarray,
    goal_pos: np.ndarray,
    alpha: float,
    k: float,
    sigma: float,
    falloff_dist: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """One OU step for the virtual position target.

    Both noise and momentum are scaled by a ramp that goes to zero as the
    virtual target nears the goal, preventing the hairball oscillation that
    would otherwise occur when σ·ε dominates the small spring force k·Δx.
    """
    to_goal = goal_pos - vt_pos
    dist = float(np.linalg.norm(to_goal))

    # Scale only noise by distance — momentum (alpha) stays constant so the
    # virtual target remains smooth near the goal rather than becoming jittery.
    noise_scale = np.clip(dist / falloff_dist, 0.0, 1.0)

    vt_vel = alpha * vt_vel + k * to_goal + sigma * noise_scale * np.random.randn(3)

    # Project out any velocity component pointing away from the goal so the
    # virtual target is always converging (never regresses).
    dist_sq = dist**2
    if dist_sq > 1e-10:
        backward = float(np.dot(vt_vel, to_goal))
        if backward < 0:
            vt_vel -= (backward / dist_sq) * to_goal

    return vt_pos + vt_vel, vt_vel


def step_ori(
    vt_ori: np.ndarray,
    vt_ang_vel: np.ndarray,
    goal_ori: np.ndarray,
    alpha: float,
    k: float,
    sigma: float,
    falloff_angle: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """One OU step for the virtual orientation target (axis-angle space)."""
    r_to_goal = _rotvec_to_goal(vt_ori, goal_ori)
    angle = float(np.linalg.norm(r_to_goal))

    # Scale both noise and momentum for orientation — unlike position, rotation
    # builds up angular momentum that can overshoot the (small) goal angle if
    # alpha is left at full strength near the goal.
    scale = np.clip(angle / falloff_angle, 0.0, 1.0)

    vt_ang_vel = alpha * scale * vt_ang_vel + k * r_to_goal + sigma * scale * np.random.randn(3)

    # Same backward-projection in rotation-vector space.
    angle_sq = angle**2
    if angle_sq > 1e-10:
        backward = float(np.dot(vt_ang_vel, r_to_goal))
        if backward < 0:
            vt_ang_vel -= (backward / angle_sq) * r_to_goal

    # Integrate: apply angular-velocity increment to virtual orientation.
    vt_ori = Rotation.from_rotvec(vt_ang_vel).as_matrix() @ vt_ori
    return vt_ori, vt_ang_vel


def simulate(
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    start_ori: np.ndarray,
    goal_ori: np.ndarray,
    n_steps: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run one trajectory; returns (positions [T+1,3], ori_errors_deg [T+1], euler_deg [T+1,3])."""
    vt_pos = start_pos.copy()
    vt_vel = np.zeros(3)
    vt_ori = start_ori.copy()
    vt_ang_vel = np.zeros(3)

    # Convert fractional falloff parameters to absolute values for this trajectory.
    initial_dist = float(np.linalg.norm(goal_pos - start_pos))
    initial_angle = float(np.linalg.norm(_rotvec_to_goal(start_ori, goal_ori)))
    falloff_dist = NOISE_FALLOFF_FRAC_POS * initial_dist
    falloff_angle = NOISE_FALLOFF_FRAC_ORI * initial_angle if initial_angle > 1e-8 else 1e-8

    positions = [vt_pos.copy()]
    ori_errors = [np.degrees(initial_angle)]
    euler_angles = [Rotation.from_matrix(vt_ori).as_euler("xyz", degrees=True)]

    for _ in range(n_steps):
        vt_pos, vt_vel = step_pos(vt_pos, vt_vel, goal_pos, ALPHA_POS, K_POS, SIGMA_POS, falloff_dist)
        vt_ori, vt_ang_vel = step_ori(vt_ori, vt_ang_vel, goal_ori, ALPHA_ORI, K_ORI, SIGMA_ORI, falloff_angle)
        positions.append(vt_pos.copy())
        ori_errors.append(np.degrees(np.linalg.norm(_rotvec_to_goal(vt_ori, goal_ori))))
        euler_angles.append(Rotation.from_matrix(vt_ori).as_euler("xyz", degrees=True))

    return np.array(positions), np.array(ori_errors), np.array(euler_angles)


def main():
    if SEED is not None:
        np.random.seed(SEED)

    trajs, ori_errs, euler_trajs = [], [], []
    for _ in range(N_TRAJECTORIES):
        traj, ori_err, euler = simulate(START_POS, GOAL_POS, START_ORI, GOAL_ORI, N_STEPS)
        trajs.append(traj)
        ori_errs.append(ori_err)
        euler_trajs.append(euler)

    colors = plt.cm.tab10(np.linspace(0, 1, N_TRAJECTORIES))
    steps = np.arange(N_STEPS + 1)

    fig = plt.figure(figsize=(18, 10))
    fig.suptitle(
        f"NM Virtual-Target Walk   "
        f"α_pos={ALPHA_POS}  k_pos={K_POS}  σ_pos={SIGMA_POS * 100:.2f} cm  falloff={NOISE_FALLOFF_FRAC_POS * 100:.0f}% dist  |  "
        f"α_ori={ALPHA_ORI}  k_ori={K_ORI}  σ_ori={np.degrees(SIGMA_ORI):.1f}°  falloff={NOISE_FALLOFF_FRAC_ORI * 100:.0f}% angle",
        fontsize=10,
    )

    # ── 3-D trajectories ─────────────────────────────────────────────────────
    ax3d = fig.add_subplot(231, projection="3d")
    for i, traj in enumerate(trajs):
        ax3d.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=colors[i], alpha=0.75, linewidth=1.2)
    ax3d.scatter(*START_POS, color="green", s=80, zorder=5, label="Start")
    ax3d.scatter(*GOAL_POS, color="red", s=80, zorder=5, label="Goal")
    ax3d.plot(
        [START_POS[0], GOAL_POS[0]],
        [START_POS[1], GOAL_POS[1]],
        [START_POS[2], GOAL_POS[2]],
        "k--",
        alpha=0.35,
        linewidth=1,
        label="Direct",
    )
    ax3d.set_xlabel("X (m)")
    ax3d.set_ylabel("Y (m)")
    ax3d.set_zlabel("Z (m)")
    ax3d.set_title("3-D trajectories")
    ax3d.legend(fontsize=7)
    ax3d.set_aspect("equal")

    # ── XY top-down ───────────────────────────────────────────────────────────
    ax_xy = fig.add_subplot(232)
    for i, traj in enumerate(trajs):
        ax_xy.plot(traj[:, 0], traj[:, 1], color=colors[i], alpha=0.75, linewidth=1.2)
    ax_xy.scatter(*START_POS[:2], color="green", s=80, zorder=5)
    ax_xy.scatter(*GOAL_POS[:2], color="red", s=80, zorder=5)
    ax_xy.plot([START_POS[0], GOAL_POS[0]], [START_POS[1], GOAL_POS[1]], "k--", alpha=0.35, linewidth=1)
    ax_xy.set_xlabel("X (m)")
    ax_xy.set_ylabel("Y (m)")
    ax_xy.set_title("XY (top-down)")
    ax_xy.set_aspect("equal")

    # ── XZ side ───────────────────────────────────────────────────────────────
    ax_xz = fig.add_subplot(233)
    for i, traj in enumerate(trajs):
        ax_xz.plot(traj[:, 0], traj[:, 2], color=colors[i], alpha=0.75, linewidth=1.2)
    ax_xz.scatter(*START_POS[[0, 2]], color="green", s=80, zorder=5)
    ax_xz.scatter(*GOAL_POS[[0, 2]], color="red", s=80, zorder=5)
    ax_xz.plot([START_POS[0], GOAL_POS[0]], [START_POS[2], GOAL_POS[2]], "k--", alpha=0.35, linewidth=1)
    ax_xz.set_xlabel("X (m)")
    ax_xz.set_ylabel("Z (m)")
    ax_xz.set_title("XZ (side view)")
    ax_xz.set_aspect("equal")

    # ── Position error over time ──────────────────────────────────────────────
    ax_perr = fig.add_subplot(234)
    for i, traj in enumerate(trajs):
        ax_perr.plot(steps, np.linalg.norm(traj - GOAL_POS, axis=1) * 100, color=colors[i], alpha=0.75, linewidth=1.2)
    ax_perr.set_xlabel("Step")
    ax_perr.set_ylabel("Distance to goal (cm)")
    ax_perr.set_title("Position convergence")

    # ── Rotation error over time ──────────────────────────────────────────────
    ax_oerr = fig.add_subplot(235)
    for i, ori_err in enumerate(ori_errs):
        ax_oerr.plot(steps, ori_err, color=colors[i], alpha=0.75, linewidth=1.2)
    ax_oerr.set_xlabel("Step")
    ax_oerr.set_ylabel("Rotation error (deg)")
    ax_oerr.set_title("Rotation convergence")

    # ── Per-axis rotation over time ───────────────────────────────────────────
    ax_axes = fig.add_subplot(236)
    axis_styles = [("Rx", "r", "-"), ("Ry", "g", "--"), ("Rz", "b", ":")]
    goal_euler = Rotation.from_matrix(GOAL_ORI).as_euler("xyz", degrees=True)
    for i, euler in enumerate(euler_trajs):
        for ax_idx, (_, color, ls) in enumerate(axis_styles):
            ax_axes.plot(steps, euler[:, ax_idx], color=color, linestyle=ls, alpha=0.3, linewidth=0.9)
    for ax_idx, (label, color, ls) in enumerate(axis_styles):
        ax_axes.axhline(goal_euler[ax_idx], color=color, linestyle=ls, linewidth=1.5, label=f"Goal {label}")
    ax_axes.set_xlabel("Step")
    ax_axes.set_ylabel("Euler angle (deg)")
    ax_axes.set_title("Per-axis rotation (XYZ Euler)")
    ax_axes.legend(fontsize=7)

    plt.tight_layout()
    out_path = "scripts/nm_virtual_target_viz.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
