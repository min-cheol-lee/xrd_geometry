from __future__ import annotations

import argparse

import matplotlib.pyplot as plt

from xrd_geometry import (
    SimulationConfig,
    launch_interactive_viewer,
    plot_lab_frame,
    plot_sample_views,
    print_angles,
    run_simulation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Python version of the XRD geometry simulation")
    parser.add_argument("--h", type=int, default=2)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--l", type=int, default=1)
    parser.add_argument("--alpha-deg", type=float, default=3.0)
    parser.add_argument("--dev-angle-deg", type=float, default=10.0)
    parser.add_argument("--wavelength", type=float, default=1.2398)
    parser.add_argument("--surface", type=float, nargs=3, default=[1.0, 0.0, 0.0])
    parser.add_argument("--n-points", type=int, default=10_000)
    parser.add_argument("--interactive", action="store_true", help="launch slider-based viewer")
    parser.add_argument("--save-prefix", default=None, help="save figures instead of showing them")
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

    if args.interactive:
        launch_interactive_viewer(config)
        return

    result = run_simulation(config)
    print_angles(result.angles_deg)

    fig_lab = plot_lab_frame(result)
    fig_views = plot_sample_views(result)

    if args.save_prefix:
        fig_lab.savefig(f"{args.save_prefix}_lab_frame.png", dpi=150, bbox_inches="tight")
        fig_views.savefig(f"{args.save_prefix}_sample_views.png", dpi=150, bbox_inches="tight")
        print(f"Saved: {args.save_prefix}_lab_frame.png")
        print(f"Saved: {args.save_prefix}_sample_views.png")
        plt.close(fig_lab)
        plt.close(fig_views)
    else:
        plt.show()


if __name__ == "__main__":
    main()
