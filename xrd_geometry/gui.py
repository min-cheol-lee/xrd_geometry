from __future__ import annotations

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

from .plotting import print_angles
from .simulation import SimulationConfig, run_simulation


def _rotation_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    """Return display rotation matrix for intrinsic X->Y->Z rotations."""
    rx = np.deg2rad(rx_deg)
    ry = np.deg2rad(ry_deg)
    rz = np.deg2rad(rz_deg)
    mx = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(rx), -np.sin(rx)],
            [0.0, np.sin(rx), np.cos(rx)],
        ]
    )
    my = np.array(
        [
            [np.cos(ry), 0.0, -np.sin(ry)],
            [0.0, 1.0, 0.0],
            [np.sin(ry), 0.0, np.cos(ry)],
        ]
    )
    mz = np.array(
        [
            [np.cos(rz), np.sin(rz), 0.0],
            [-np.sin(rz), np.cos(rz), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return mz @ (my @ mx)


def launch_interactive_viewer(default: SimulationConfig | None = None) -> None:
    """Interactive matplotlib viewer with input sliders.

    Mouse controls on the 3D axes provide rotate + zoom.
    """

    cfg = default or SimulationConfig()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0.15, bottom=0.45)

    # slider axes
    ax_alpha = fig.add_axes([0.15, 0.18, 0.75, 0.03])
    ax_dev = fig.add_axes([0.15, 0.13, 0.75, 0.03])
    ax_h = fig.add_axes([0.15, 0.08, 0.23, 0.03])
    ax_k = fig.add_axes([0.43, 0.08, 0.23, 0.03])
    ax_l = fig.add_axes([0.71, 0.08, 0.19, 0.03])
    ax_rx = fig.add_axes([0.15, 0.33, 0.75, 0.03])
    ax_ry = fig.add_axes([0.15, 0.28, 0.75, 0.03])
    ax_rz = fig.add_axes([0.15, 0.23, 0.75, 0.03])

    s_alpha = Slider(ax_alpha, "alpha (deg)", 0.1, 20.0, valinit=cfg.alpha_deg)
    s_dev = Slider(ax_dev, "dev (deg)", -30.0, 30.0, valinit=cfg.dev_angle_deg)
    s_h = Slider(ax_h, "h", 0, 6, valinit=cfg.h, valstep=1)
    s_k = Slider(ax_k, "k", 0, 6, valinit=cfg.k, valstep=1)
    s_l = Slider(ax_l, "l", 0, 6, valinit=cfg.l, valstep=1)
    s_rx = Slider(ax_rx, "Rx (deg)", -180.0, 180.0, valinit=0.0)
    s_ry = Slider(ax_ry, "Ry (deg)", -180.0, 180.0, valinit=0.0)
    s_rz = Slider(ax_rz, "Rz (deg)", -180.0, 180.0, valinit=0.0)

    def draw_scene(local_cfg: SimulationConfig):
        elev = ax.elev
        azim = ax.azim
        ax.cla()
        result = run_simulation(local_cfg)
        v = result.vectors
        m_disp = _rotation_xyz(float(s_rx.val), float(s_ry.val), float(s_rz.val))

        def seg(vec, scale=1.0):
            p0 = np.zeros(3)
            p1 = m_disp @ (np.asarray(vec) * scale)
            return np.vstack([p0, p1])

        for key, color, lw, ls, scale in [
            ("xin_lab", "k", 2, "-", 1.0),
            ("xout_lab", "k", 2, "-", 1.0),
            ("ghkl_lab", (1.0, 0.6, 0.0), 3, "-", 1.0),
            ("surface_lab", (0.0, 0.6, 1.0), 3, "-", 1.0),
            ("optical", (0.8, 0.0, 0.0), 3, "-", 1.0),
            ("sam_b_lab", "b", 2, "--", 0.5),
            ("sam_c_lab", "r", 2, "--", 0.5),
        ]:
            line = seg(v[key], scale)
            ax.plot(line[:, 0], line[:, 1], line[:, 2], color=color, linewidth=lw, linestyle=ls)

        sample_vertices = (m_disp @ result.sample_vertices.T).T
        faces = [[sample_vertices[idx - 1] for idx in face] for face in result.sample_faces]
        patch = Poly3DCollection(faces, facecolor=(0.8, 0.8, 0.8), alpha=0.35, edgecolor="k", linewidth=0.6)
        ax.add_collection3d(patch)

        for axis_vec, c in [
            (np.array([1.0, 0.0, 0.0]), "0.5"),
            (np.array([0.0, 1.0, 0.0]), "0.5"),
            (np.array([0.0, 0.0, 1.0]), "0.5"),
        ]:
            ref_line = seg(axis_vec, scale=0.6)
            ax.plot(ref_line[:, 0], ref_line[:, 1], ref_line[:, 2], color=c, linewidth=1.0, linestyle=":")

        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_title("XRD geometry (interactive + 3D rotation test)")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        if elev == 30 and azim == -60:
            ax.view_init(elev=30, azim=120)
        else:
            ax.view_init(elev=elev, azim=azim)
        print_angles(result.angles_deg)

    def update(_):
        new_cfg = SimulationConfig(
            h=int(s_h.val),
            k=int(s_k.val),
            l=int(s_l.val),
            alpha_deg=float(s_alpha.val),
            dev_angle_deg=float(s_dev.val),
            wavelength=cfg.wavelength,
            surface=cfg.surface,
            ybco_a=cfg.ybco_a,
            ybco_b=cfg.ybco_b,
            ybco_c=cfg.ybco_c,
            n_points=cfg.n_points,
        )
        draw_scene(new_cfg)
        fig.canvas.draw_idle()

    for slider in (s_alpha, s_dev, s_h, s_k, s_l, s_rx, s_ry, s_rz):
        slider.on_changed(update)

    draw_scene(cfg)
    plt.show()
