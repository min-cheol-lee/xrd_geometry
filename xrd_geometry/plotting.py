from __future__ import annotations

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .simulation import SimulationResult


def _segment(vec: np.ndarray, scale: float = 1.0) -> np.ndarray:
    vec = np.asarray(vec, dtype=float)
    return np.vstack([np.zeros(3), vec * scale])


def _draw_patch(ax, vertices: np.ndarray, faces: np.ndarray, alpha: float = 0.35) -> None:
    poly3d = [[vertices[idx - 1] for idx in face] for face in faces]
    patch = Poly3DCollection(poly3d, facecolor=(0.8, 0.8, 0.8), alpha=alpha, edgecolor="k", linewidth=0.6)
    ax.add_collection3d(patch)


def _setup_axes(ax, title: str, limits=(-1, 1)) -> None:
    ax.set_title(title)
    ax.set_xlim(*limits)
    ax.set_ylim(*limits)
    ax.set_zlim(*limits)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")


def plot_lab_frame(result: SimulationResult) -> plt.Figure:
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    v = result.vectors

    for key, color, lw, ls, scale in [
        ("xin_lab", "k", 2, "-", 1.0),
        ("xout_lab", "k", 2, "-", 1.0),
        ("ghkl_lab", (1.0, 0.6, 0.0), 3, "-", 1.0),
        ("surface_lab", (0.0, 0.6, 1.0), 3, "-", 1.0),
        ("optical", (0.8, 0.0, 0.0), 3, "-", 1.0),
        ("sam_b_lab", "b", 2, "--", 0.5),
        ("sam_c_lab", "r", 2, "--", 0.5),
    ]:
        seg = _segment(v[key], scale=scale)
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linestyle=ls, linewidth=lw, color=color)

    for key, color in [("optical_b", "b"), ("optical_c", "r")]:
        center = v["optical"] * 2 / 3
        seg = np.vstack([center - v[key] / 8, center + v[key] / 8])
        ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linewidth=3, color=color)

    _draw_patch(ax, result.sample_vertices, result.sample_faces)
    _setup_axes(ax, "Lab frame geometry")
    ax.view_init(elev=30, azim=120)
    return fig


def plot_sample_views(result: SimulationResult) -> plt.Figure:
    fig = plt.figure(figsize=(10, 8))
    views = [(30, 120, "Perspective"), (90, 0, "Top"), (0, 90, "Side"), (0, 0, "Front")]

    for i, (elev, azim, title) in enumerate(views, start=1):
        ax = fig.add_subplot(2, 2, i, projection="3d")
        v = result.vectors
        for key, color, lw, ls, scale in [
            ("xin_lab", "k", 2, "-", 1.0),
            ("surface_lab", (0.0, 0.6, 1.0), 3, "-", 1.0),
            ("sam_b_lab", "k", 1.5, "--", 0.5),
            ("sam_c_lab", "k", 1.5, "--", 0.5),
        ]:
            seg = _segment(v[key], scale=scale)
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], linestyle=ls, linewidth=lw, color=color)

        _draw_patch(ax, result.sample_vertices, result.sample_faces)
        _setup_axes(ax, title)
        ax.view_init(elev=elev, azim=azim)

    fig.tight_layout()
    return fig


def print_angles(angles_deg: Dict[str, float]) -> None:
    print("Computed alignment / geometry outputs (degrees):")
    for key in [
        "omega_deg",
        "chi_deg",
        "phi_deg",
        "th_deg",
        "tth_deg",
        "optical_alpha_deg",
        "pol_optical_c_axis_deg",
        "angle_optical_b_axis_deg",
        "alpha_real_deg",
    ]:
        print(f"  {key:>24s}: {angles_deg[key]: .6f}")
