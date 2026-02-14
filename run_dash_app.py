from __future__ import annotations

import argparse

from xrd_geometry import SimulationConfig, create_dash_app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Dash web GUI for XRD geometry simulation")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--alpha-deg", type=float, default=3.0)
    parser.add_argument("--dev-angle-deg", type=float, default=10.0)
    parser.add_argument("--wavelength", type=float, default=1.2398)
    parser.add_argument("--surface", type=float, nargs=3, default=[1.0, 0.0, 0.0])
    parser.add_argument("--n-points", type=int, default=10_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimulationConfig(
        h=args.h,
        k=args.k,
        l=args.l,
        alpha_deg=args.alpha_deg,
        dev_angle_deg=args.dev_angle_deg,
        wavelength=args.wavelength,
        surface=tuple(args.surface),
        n_points=args.n_points,
    )
    app = create_dash_app(config)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()

