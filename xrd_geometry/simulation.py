from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class SimulationConfig:
    """Input parameters for the XRD geometry simulation."""

    h: int = 2
    k: int = 1
    l: int = 1
    alpha_deg: float = 3.0
    dev_angle_deg: float = 10.0
    wavelength: float = 1.2398
    surface: Tuple[float, float, float] = (1.0, 0.0, 0.0)
    ybco_a: float = 7.55186
    ybco_b: float = 3.81058
    ybco_c: float = 11.49652
    n_points: int = 10_000


@dataclass
class SimulationResult:
    vectors: Dict[str, np.ndarray]
    sample_vertices: np.ndarray
    sample_faces: np.ndarray
    angles_deg: Dict[str, float]


def _normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Cannot normalize a zero vector")
    return v / n


def _rot_x(theta: float) -> np.ndarray:
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, np.cos(theta), -np.sin(theta)],
            [0.0, np.sin(theta), np.cos(theta)],
        ]
    )


def _rot_y(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), 0.0, -np.sin(theta)],
            [0.0, 1.0, 0.0],
            [np.sin(theta), 0.0, np.cos(theta)],
        ]
    )


def _rot_z(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), np.sin(theta), 0.0],
            [-np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def _build_sample_vertices(surface: np.ndarray) -> np.ndarray:
    thick, inp1, inp2 = 0.1, 0.5, 0.5
    vert = np.array(
        [
            [-thick, -inp1, -inp2],
            [-thick, inp1, -inp2],
            [0.0, inp1, -inp2],
            [0.0, -inp1, -inp2],
            [-thick, -inp1, inp2],
            [-thick, inp1, inp2],
            [0.0, inp1, inp2],
            [0.0, -inp1, inp2],
        ]
    )

    if np.array_equal(surface, np.array([0.0, 1.0, 0.0])):
        vert[:, [0, 1]] = vert[:, [1, 0]]
    elif np.array_equal(surface, np.array([0.0, 0.0, 1.0])):
        vert[:, [0, 2]] = vert[:, [2, 0]]

    return vert


def run_simulation(config: SimulationConfig) -> SimulationResult:
    h, k, l = config.h, config.k, config.l
    surface = np.array(config.surface, dtype=float)
    surface = _normalize(surface)

    # crystal_setting.m
    a, b, c = config.ybco_a, config.ybco_b, config.ybco_c
    d = 1.0 / np.sqrt(h**2 / a**2 + k**2 / b**2 + l**2 / c**2)
    G = 2.0 * np.pi / d
    th = np.arcsin(config.wavelength / (2.0 * d))

    ghkl = np.array([2.0 * np.pi / a * h, 2.0 * np.pi / b * k, 2.0 * np.pi / c * l])
    ghkl_n = _normalize(ghkl)
    ghkl_th = np.arctan(np.sqrt(ghkl_n[0] ** 2 + ghkl_n[1] ** 2) / ghkl_n[2])
    ghkl_phi = np.arctan(ghkl_n[1] / ghkl_n[0])

    xaxis = np.array([1.0, 0.0, 0.0])
    yaxis = np.array([0.0, 1.0, 0.0])
    zaxis = np.array([0.0, 0.0, 1.0])

    # graphic_setting.m
    dev_angle = np.deg2rad(config.dev_angle_deg)
    alpha = np.deg2rad(config.alpha_deg)

    m_r_th = np.array(
        [
            [np.cos(ghkl_th), 0.0, np.sin(ghkl_th)],
            [0.0, 1.0, 0.0],
            [-np.sin(ghkl_th), 0.0, np.cos(ghkl_th)],
        ]
    )
    m_r_phi = np.array(
        [
            [np.cos(ghkl_phi), -np.sin(ghkl_phi), 0.0],
            [np.sin(ghkl_phi), np.cos(ghkl_phi), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    sample_vertices = _build_sample_vertices(surface)
    sample_faces = np.array(
        [
            [1, 2, 3, 4],
            [2, 6, 7, 3],
            [4, 3, 7, 8],
            [1, 5, 8, 4],
            [1, 2, 6, 5],
            [5, 6, 7, 8],
        ],
        dtype=int,
    )

    # figure_crystal_frame.m (beam search for alpha)
    xin = np.zeros((config.n_points, 3), dtype=float)
    in_ang = np.zeros((config.n_points, 2), dtype=float)

    for i in range(config.n_points):
        phase = 2.0 * np.pi * (i + 1) / config.n_points
        xin_z = np.array([np.cos(phase), np.sin(phase), np.tan(th)])
        xin_z = _normalize(xin_z)
        xin_i = m_r_phi @ (m_r_th @ xin_z)
        xin[i, :] = xin_i
        in_ang[i, 0] = 90.0 - np.rad2deg(np.arccos(np.dot(surface, xin_i)))
        in_ang[i, 1] = in_ang[i, 0] - config.alpha_deg

    idx = int(np.argmin(np.abs(in_ang[:, 1])))
    xin_alpha = xin[idx, :]
    xout_alpha = _normalize(ghkl_n * G - xin_alpha * (2.0 * np.pi / config.wavelength))

    # rotate_crystal_to_lab.m
    xin_th = np.arctan(np.sqrt(xin_alpha[0] ** 2 + xin_alpha[1] ** 2) / xin_alpha[2])
    xin_phi = np.arctan(xin_alpha[1] / xin_alpha[0])

    m_r_th2 = _rot_x(np.pi / 2.0 - xin_th)
    m_r_phi2 = _rot_z(np.pi / 2.0 + xin_phi)

    xout_lab_pre = m_r_th2 @ (m_r_phi2 @ xout_alpha)

    if np.sign(xout_lab_pre[0]) == 1:
        sign_opt_dev = -1.0
        set_rot_angle = 0.0
    else:
        sign_opt_dev = +1.0
        set_rot_angle = np.pi

    rot_angle = np.arctan(xout_lab_pre[2] / xout_lab_pre[0])
    rot_angle_eff = np.pi / 2.0 - rot_angle + set_rot_angle

    m_r_rot = _rot_y(rot_angle_eff)

    xout_lab = m_r_rot @ xout_lab_pre
    m_r_total = m_r_rot @ (m_r_th2 @ m_r_phi2)

    xin_lab = m_r_total @ xin_alpha
    surface_lab = m_r_total @ surface
    ghkl_lab = m_r_total @ ghkl

    sam_b_lab = m_r_total @ yaxis
    sam_c_lab = m_r_total @ zaxis

    m_r_dev = np.array(
        [
            [np.cos(sign_opt_dev * dev_angle), np.sin(sign_opt_dev * dev_angle), 0.0],
            [-np.sin(sign_opt_dev * dev_angle), np.cos(sign_opt_dev * dev_angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    optical = m_r_dev @ xin_lab

    optical_vert = np.cross(np.array([0.0, 0.0, 1.0]), optical)
    optical_vert = _normalize(optical_vert)

    sam_c_hor = float(np.dot(sam_c_lab, np.array([0.0, 0.0, 1.0])))
    sam_c_ver = float(np.dot(sam_c_lab, optical_vert))
    sam_b_hor = float(np.dot(sam_b_lab, np.array([0.0, 0.0, 1.0])))
    sam_b_ver = float(np.dot(sam_b_lab, optical_vert))

    pol_optical_c_axis = np.arctan(sam_c_hor / sam_c_ver)
    angle_optical_b_axis = np.arctan(sam_b_hor / sam_b_ver)

    optical_c = np.sin(pol_optical_c_axis) * np.array([0.0, 0.0, 1.0]) + np.cos(
        pol_optical_c_axis
    ) * optical_vert
    optical_b = np.sin(pol_optical_c_axis - np.pi / 2.0) * np.array([0.0, 0.0, 1.0]) + np.cos(
        pol_optical_c_axis - np.pi / 2.0
    ) * optical_vert

    sample_vertices_lab = (m_r_total @ sample_vertices.T).T

    # rotate_crystal_init_alignment.m
    sam_chi = np.arctan(np.sqrt(surface_lab[0] ** 2 + surface_lab[1] ** 2) / surface_lab[2])
    sam_omega = np.arctan(surface_lab[1] / surface_lab[0])

    m_r_th3 = _rot_x(np.pi / 2.0 - sam_chi)
    m_r_phi3 = _rot_z(np.pi / 2.0 + sam_omega)
    m_r_init = m_r_th3 @ m_r_phi3

    sam_b_init = m_r_init @ sam_b_lab

    sam_phi = np.arctan(sam_b_init[2] / sam_b_init[0])

    optical_alpha_deg = 90.0 - np.rad2deg(np.arccos(np.dot(optical, surface_lab)))

    vectors = {
        "xin_lab": xin_lab,
        "xout_lab": xout_lab,
        "ghkl_lab": ghkl_lab,
        "surface_lab": surface_lab,
        "optical": optical,
        "sam_b_lab": sam_b_lab,
        "sam_c_lab": sam_c_lab,
        "optical_b": optical_b,
        "optical_c": optical_c,
    }

    angles_deg = {
        "omega_deg": float(np.rad2deg(sam_omega)),
        "chi_deg": float(np.rad2deg(sam_chi) + 90.0),
        "phi_deg": float(np.rad2deg(sam_phi)),
        "optical_alpha_deg": float(optical_alpha_deg),
        "pol_optical_c_axis_deg": float(np.rad2deg(pol_optical_c_axis)),
        "angle_optical_b_axis_deg": float(np.rad2deg(angle_optical_b_axis)),
        "th_deg": float(np.rad2deg(th)),
        "tth_deg": float(2.0 * np.rad2deg(th)),
        "alpha_real_deg": float(in_ang[idx, 0]),
    }

    return SimulationResult(
        vectors=vectors,
        sample_vertices=sample_vertices_lab,
        sample_faces=sample_faces,
        angles_deg=angles_deg,
    )
