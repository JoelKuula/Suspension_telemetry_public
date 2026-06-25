#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.postprocess_gui_app.backend.analysis_service import build_derived_analog
from scripts.postprocess_gui_app.backend.occupancy_service import (
    compute_occupancy_grid,
    save_occupancy_grid_csv,
    save_occupancy_heatmap_png,
)
from scripts.postprocess_gui_app.backend.session_service import open_export_session


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot a position/velocity heatmap from an exported analog.csv file."
    )
    parser.add_argument("path", type=Path, help="Path to exported analog.csv")
    parser.add_argument("--channel", choices=("front", "rear"), default="front", help="Which analog channel to plot")
    parser.add_argument("--output", type=Path, help="Output image path. Defaults to <csv stem>_<channel>_heatmap.png")
    parser.add_argument("--grid-output", type=Path, help="Optional CSV output for the heatmap grid")
    parser.add_argument("--travel-bins", type=int, default=100, help="Number of travel bins")
    parser.add_argument("--velocity-bins", type=int, default=120, help="Number of velocity bins")
    parser.add_argument(
        "--manual-anchor-count",
        type=float,
        help="Manual raw ADC anchor count for the selected channel",
    )
    parser.add_argument(
        "--velocity-source",
        choices=("filtered", "csv"),
        default="filtered",
        help="'filtered' uses the derived smoothed gradient. 'csv' uses the exported derivative column.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=21,
        help="Odd moving-average window in samples for filtered velocity generation",
    )
    parser.add_argument("--travel-percentile-low", type=float, default=0.0, help="Low travel percentile for clipping")
    parser.add_argument("--travel-percentile-high", type=float, default=100.0, help="High travel percentile for clipping")
    parser.add_argument("--velocity-percentile", type=float, default=99.5, help="Absolute velocity percentile for clipping")
    parser.add_argument("--travel-min", type=float, help="Explicit travel-axis minimum")
    parser.add_argument("--travel-max", type=float, help="Explicit travel-axis maximum")
    parser.add_argument("--velocity-min", type=float, help="Explicit velocity-axis minimum")
    parser.add_argument("--velocity-max", type=float, help="Explicit velocity-axis maximum")
    parser.add_argument("--color-scale", choices=("linear", "sqrt", "log"), default="sqrt", help="Color normalization")
    parser.add_argument("--dpi", type=int, default=180, help="Output image DPI")
    return parser.parse_args()


def build_output_path(csv_path: Path, channel: str) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_{channel}_heatmap.png")


def main() -> int:
    args = parse_args()
    export_dir = args.path.parent
    bundle = open_export_session(export_dir)

    config = copy.deepcopy(bundle.session_config)
    config[args.channel]["travel_reference"] = "manual"
    if args.manual_anchor_count is not None:
        config[args.channel]["manual_reference_count"] = args.manual_anchor_count
    config[args.channel]["velocity_filter_window"] = args.smooth_window
    derived_df, _ = build_derived_analog(
        export_dir=export_dir,
        config=config,
        analog_df=bundle.analog_df,
        force=True,
    )

    occupancy = compute_occupancy_grid(
        derived_df=derived_df,
        channel=args.channel,
        travel_bins=args.travel_bins,
        velocity_bins=args.velocity_bins,
        velocity_source=args.velocity_source,
        color_scale=args.color_scale,
        travel_percentile_low=args.travel_percentile_low,
        travel_percentile_high=args.travel_percentile_high,
        velocity_percentile=args.velocity_percentile,
        travel_min=args.travel_min,
        travel_max=args.travel_max,
        velocity_min=args.velocity_min,
        velocity_max=args.velocity_max,
    )

    output_path = args.output if args.output is not None else build_output_path(args.path, args.channel)
    save_occupancy_heatmap_png(output_path, occupancy, dpi=args.dpi)
    if args.grid_output is not None:
        save_occupancy_grid_csv(args.grid_output, occupancy)

    print(f"Saved heatmap to {output_path}")
    manual_anchor = config[args.channel].get("manual_reference_count")
    print(
        f"  channel={args.channel} manual_anchor={manual_anchor if manual_anchor is not None else 'unset'} "
        f"velocity_source={occupancy['velocity_source']}"
    )
    print(
        f"  samples={occupancy['sample_count']} time_s={occupancy['total_occupancy_s']:.3f} "
        f"travel_range=[{occupancy['travel_range'][0]:.3f}, {occupancy['travel_range'][1]:.3f}] "
        f"velocity_range=[{occupancy['velocity_range'][0]:.3f}, {occupancy['velocity_range'][1]:.3f}]"
    )
    if args.grid_output is not None:
        print(f"  grid_csv={args.grid_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
