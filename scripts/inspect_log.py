#!/usr/bin/env python3
import argparse
import pathlib
import sys

from log_parser import parse_log


def inspect_file(path: pathlib.Path) -> int:
    try:
        session = parse_log(path)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    print("Header")
    for key, value in session["header"].items():
        print(f"  {key}: {value}")

    print("\nRecord counts")
    for tag, count in sorted(session["record_counts"].items()):
        print(f"  {tag}: {count}")

    print("\nRecord previews")
    for record in session["records"][:12]:
        header = record["header"]
        payload = record["payload"]
        if header["tag"] == "IMU " and "frames_preview" in payload:
            payload = {key: value for key, value in payload.items() if key != "frames"}
        print(
            f"  seq={header['seq']} tag={header['tag']} len={header['payload_len']} "
            f"flags=0x{header['flags']:04X} t_us={header['timestamp_us']}"
        )
        print(f"    {payload}")

    print(f"\nParsed {sum(session['record_counts'].values())} records from {path}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Inspect a suspension telemetry binary log file.")
    parser.add_argument("path", type=pathlib.Path, help="Path to a .BIN session file")
    args = parser.parse_args()
    return inspect_file(args.path)


if __name__ == "__main__":
    raise SystemExit(main())
