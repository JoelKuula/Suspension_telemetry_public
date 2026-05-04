#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.postprocess_gui_app.backend.export_service import export_bin_to_directory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a suspension telemetry BIN log into engineering-friendly CSV files.")
    parser.add_argument("path", type=Path, help="Path to a .BIN session file")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for exported CSV files. Defaults to ./exports/<BIN stem>",
    )
    parser.add_argument("--front-mm-per-count", type=float, help="Front channel scale in mm per ADC count")
    parser.add_argument("--rear-mm-per-count", type=float, help="Rear channel scale in mm per ADC count")
    parser.add_argument("--front-full-scale-mm", type=float, help="Front channel full-scale travel in mm across the ADC range")
    parser.add_argument("--rear-full-scale-mm", type=float, help="Rear channel full-scale travel in mm across the ADC range")
    parser.add_argument("--front-zero-count", type=int, default=0, help="Front zero reference ADC count")
    parser.add_argument("--rear-zero-count", type=int, default=0, help="Rear zero reference ADC count")
    parser.add_argument("--front-invert", action="store_true", help="Invert the sign of the front derived position and velocity")
    parser.add_argument("--rear-invert", action="store_true", help="Invert the sign of the rear derived position and velocity")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary, output_dir = export_bin_to_directory(
        bin_path=args.path,
        output_dir=args.output_dir,
        front_mm_per_count=args.front_mm_per_count,
        rear_mm_per_count=args.rear_mm_per_count,
        front_full_scale_mm=args.front_full_scale_mm,
        rear_full_scale_mm=args.rear_full_scale_mm,
        front_zero_count=args.front_zero_count,
        rear_zero_count=args.rear_zero_count,
        front_invert=args.front_invert,
        rear_invert=args.rear_invert,
    )

    print(f"Exported {args.path} to {output_dir}")
    print(
        f"  records={sum(summary['record_counts'].values())} "
        f"analog={summary['counts']['analog_rows']} "
        f"imu_frames={summary['counts']['imu_frame_rows']} "
        f"wheel={summary['counts']['wheel_rows']} "
        f"stat={summary['counts']['stat_rows']}"
    )
    derived_scales = summary.get("derived_scales", {})
    if derived_scales.get("front_mm_per_count") is not None or derived_scales.get("rear_mm_per_count") is not None:
        print(
            f"  derived scales: front_mm_per_count={derived_scales.get('front_mm_per_count')} "
            f"rear_mm_per_count={derived_scales.get('rear_mm_per_count')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
