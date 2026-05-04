#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.postprocess_gui_app.backend.quicklook_service import run_quicklook


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a focused trackside quicklook from a .BIN log or an existing exports/<session>/ directory."
    )
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to a raw .BIN session file or an existing exports/<session>/ directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Directory for quicklook outputs. Defaults to <export>/analysis/quicklook",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_quicklook(args.input, output_dir=args.output_dir)
    front = result.summary["channels"]["front"]
    rear = result.summary["channels"]["rear"]
    highlights = result.summary["status"]["highlighted_statuses"]

    print(f"Quicklook outputs written to {result.output_dir}")
    print(f"  summary_json={result.summary_json_path}")
    print(f"  summary_text={result.summary_text_path}")
    print(f"  travel_png={result.artifact_paths['travel_overview_png']}")
    print(f"  velocity_png={result.artifact_paths['velocity_histograms_png']}")
    print(
        f"  duration_s={result.summary['session']['duration_s']:.3f} "
        f"analog_rows={result.summary['session']['analog_rows']} "
        f"wheel_rows={result.summary['session']['wheel_rows']} "
        f"imu_frame_rows={result.summary['session']['imu_frame_rows']}"
    )
    print(
        f"  front_used_stroke={front['used_stroke']:.2f} {front['travel_units']} "
        f"rear_used_stroke={rear['used_stroke']:.2f} {rear['travel_units']}"
    )
    if highlights:
        print("  highlighted_statuses=" + ", ".join(f"{row['name']} x{row['count']}" for row in highlights))
    else:
        print("  highlighted_statuses=none")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
