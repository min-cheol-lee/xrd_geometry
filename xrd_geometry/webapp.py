from __future__ import annotations

from typing import Dict, Tuple

import dash
from dash import Dash, Input, Output, dcc, html
import numpy as np
import plotly.graph_objects as go

from .simulation import SimulationConfig, SimulationResult, run_simulation


def _rotation_xyz(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
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


def _line_trace(
    vec: np.ndarray, name: str, color: str, width: int, dash_style: str, rot: np.ndarray, scale: float = 1.0
) -> go.Scatter3d:
    p0 = np.zeros(3)
    p1 = rot @ (np.asarray(vec, dtype=float) * scale)
    return go.Scatter3d(
        x=[p0[0], p1[0]],
        y=[p0[1], p1[1]],
        z=[p0[2], p1[2]],
        mode="lines",
        name=name,
        line={"color": color, "width": width, "dash": dash_style},
    )


def _mesh_from_faces(vertices: np.ndarray, faces: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    tri_i = []
    tri_j = []
    tri_k = []
    for face in faces:
        a, b, c, d = (int(idx) - 1 for idx in face)
        tri_i.extend([a, a])
        tri_j.extend([b, c])
        tri_k.extend([c, d])
    return np.array(tri_i), np.array(tri_j), np.array(tri_k)


def _build_figure(result: SimulationResult, rx: float, ry: float, rz: float) -> go.Figure:
    v = result.vectors
    rot = _rotation_xyz(rx, ry, rz)

    traces = [
        _line_trace(v["xin_lab"], "xin", "#111111", 8, "solid", rot, 1.0),
        _line_trace(v["xout_lab"], "xout", "#111111", 8, "solid", rot, 1.0),
        _line_trace(v["ghkl_lab"], "ghkl", "#dd6b20", 10, "solid", rot, 1.0),
        _line_trace(v["surface_lab"], "surface", "#0284c7", 10, "solid", rot, 1.0),
        _line_trace(v["optical"], "optical", "#dc2626", 10, "solid", rot, 1.0),
        _line_trace(v["sam_b_lab"], "sam_b", "#2563eb", 5, "dash", rot, 0.5),
        _line_trace(v["sam_c_lab"], "sam_c", "#7c3aed", 5, "dash", rot, 0.5),
    ]

    for axis_vec, label in [
        (np.array([1.0, 0.0, 0.0]), "X"),
        (np.array([0.0, 1.0, 0.0]), "Y"),
        (np.array([0.0, 0.0, 1.0]), "Z"),
    ]:
        traces.append(_line_trace(axis_vec, f"ref_{label}", "#94a3b8", 3, "dot", rot, 0.7))

    vertices = (rot @ result.sample_vertices.T).T
    i_idx, j_idx, k_idx = _mesh_from_faces(vertices, result.sample_faces)
    traces.append(
        go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=i_idx,
            j=j_idx,
            k=k_idx,
            color="#d4d4d8",
            opacity=0.42,
            name="sample",
            showscale=False,
        )
    )

    fig = go.Figure(data=traces)
    fig.update_layout(
        template="plotly_white",
        margin={"l": 10, "r": 10, "t": 40, "b": 10},
        title="XRD Geometry Web GUI",
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.12, "xanchor": "left", "x": 0.0},
        scene={
            "xaxis": {"title": "X", "range": [-1.05, 1.05], "backgroundcolor": "#f8fafc"},
            "yaxis": {"title": "Y", "range": [-1.05, 1.05], "backgroundcolor": "#f8fafc"},
            "zaxis": {"title": "Z", "range": [-1.05, 1.05], "backgroundcolor": "#f8fafc"},
            "aspectmode": "cube",
            "camera": {"eye": {"x": 1.3, "y": 1.2, "z": 0.95}},
        },
        paper_bgcolor="#f1f5f9",
        plot_bgcolor="#f1f5f9",
    )
    return fig


def _slider(label: str, slider_id: str, min_v: float, max_v: float, value: float, step: float = 1.0) -> html.Div:
    marks = {
        float(min_v): str(int(min_v)),
        float(max_v): str(int(max_v)),
    }
    return html.Div(
        [
            html.Label(label, style={"fontWeight": "600", "display": "block", "marginBottom": "0.2rem"}),
            dcc.Slider(
                id=slider_id,
                min=min_v,
                max=max_v,
                step=step,
                value=value,
                marks=marks,
                tooltip={"placement": "bottom", "always_visible": True},
            ),
        ],
        style={"marginBottom": "1rem"},
    )


def _angles_table(angles: Dict[str, float]) -> html.Table:
    rows = []
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
        rows.append(
            html.Tr(
                [
                    html.Td(key, style={"padding": "0.25rem 0.5rem", "fontFamily": "Consolas, monospace"}),
                    html.Td(f"{angles[key]:.5f}", style={"padding": "0.25rem 0.5rem", "textAlign": "right"}),
                ]
            )
        )
    return html.Table(rows, style={"width": "100%", "borderCollapse": "collapse"})


def create_dash_app(default: SimulationConfig | None = None) -> Dash:
    cfg = default or SimulationConfig()
    app = dash.Dash(__name__)
    app.title = "XRD Geometry Web GUI"

    app.layout = html.Div(
        [
            html.H2("XRD Geometry Simulator", style={"marginTop": "0"}),
            html.P("Interactive web GUI with 3D rotation test controls for website deployment."),
            html.Div(
                [
                    html.Div(
                        [
                            _slider("h", "h", 0, 6, cfg.h, 1),
                            _slider("k", "k", 0, 6, cfg.k, 1),
                            _slider("l", "l", 0, 6, cfg.l, 1),
                            _slider("alpha (deg)", "alpha", 0.1, 20.0, cfg.alpha_deg, 0.1),
                            _slider("dev (deg)", "dev", -30.0, 30.0, cfg.dev_angle_deg, 0.1),
                            _slider("Rx (deg)", "rx", -180.0, 180.0, 0.0, 1.0),
                            _slider("Ry (deg)", "ry", -180.0, 180.0, 0.0, 1.0),
                            _slider("Rz (deg)", "rz", -180.0, 180.0, 0.0, 1.0),
                        ],
                        style={
                            "background": "#ffffff",
                            "padding": "1rem",
                            "borderRadius": "12px",
                            "boxShadow": "0 6px 24px rgba(0,0,0,0.08)",
                        },
                    ),
                    html.Div(
                        [html.Div(id="angles-table"), dcc.Graph(id="scene-3d", style={"height": "74vh"})],
                        style={
                            "background": "#ffffff",
                            "padding": "1rem",
                            "borderRadius": "12px",
                            "boxShadow": "0 6px 24px rgba(0,0,0,0.08)",
                        },
                    ),
                ],
                style={
                    "display": "grid",
                    "gridTemplateColumns": "minmax(300px, 360px) 1fr",
                    "gap": "1rem",
                    "alignItems": "start",
                },
            ),
        ],
        style={
            "padding": "1.2rem",
            "fontFamily": "Segoe UI, Arial, sans-serif",
            "background": "linear-gradient(135deg, #e2e8f0 0%, #f8fafc 100%)",
            "minHeight": "100vh",
        },
    )

    @app.callback(
        [Output("scene-3d", "figure"), Output("angles-table", "children")],
        [
            Input("h", "value"),
            Input("k", "value"),
            Input("l", "value"),
            Input("alpha", "value"),
            Input("dev", "value"),
            Input("rx", "value"),
            Input("ry", "value"),
            Input("rz", "value"),
        ],
    )
    def _update(h: float, k: float, l: float, alpha: float, dev: float, rx: float, ry: float, rz: float):
        cfg_local = SimulationConfig(
            h=int(h),
            k=int(k),
            l=int(l),
            alpha_deg=float(alpha),
            dev_angle_deg=float(dev),
            wavelength=cfg.wavelength,
            surface=cfg.surface,
            ybco_a=cfg.ybco_a,
            ybco_b=cfg.ybco_b,
            ybco_c=cfg.ybco_c,
            n_points=cfg.n_points,
        )
        result = run_simulation(cfg_local)
        figure = _build_figure(result, float(rx), float(ry), float(rz))
        table = _angles_table(result.angles_deg)
        return figure, table

    return app

