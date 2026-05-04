from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.log_parser import parse_log


ANALOG_FIELDS = [
    "record_seq",
    "record_flags",
    "host_timestamp_us",
    "host_time_s",
    "dt_us",
    "sample_index",
    "front_raw",
    "rear_raw",
    "front_position_counts",
    "rear_position_counts",
    "front_norm",
    "rear_norm",
    "front_velocity_counts_per_s",
    "rear_velocity_counts_per_s",
    "front_mm",
    "rear_mm",
    "front_velocity_mm_per_s",
    "rear_velocity_mm_per_s",
]

WHEEL_FIELDS = [
    "record_seq",
    "record_flags",
    "host_timestamp_us",
    "host_time_s",
    "pulse_count",
    "interval_us",
    "period_s",
    "speed_m_s",
    "speed_kph",
    "speed_mph",
]

STAT_FIELDS = [
    "record_seq",
    "record_flags",
    "host_timestamp_us",
    "host_time_s",
    "code",
    "name",
    "flags",
    "value0",
    "value1",
]

IMU_BURST_FIELDS = [
    "record_seq",
    "record_flags",
    "burst_index",
    "burst_host_timestamp_us",
    "burst_host_time_s",
    "frame_count",
    "accel_frames",
    "gyro_frames",
    "timestamp_frames",
    "fifo_level_before",
    "fifo_level_after",
    "burst_flags",
    "first_timestamp_raw",
    "first_timestamp_us",
    "last_timestamp_raw",
    "last_timestamp_us",
]

IMU_FRAME_FIELDS = [
    "record_seq",
    "record_flags",
    "burst_index",
    "burst_host_timestamp_us",
    "burst_host_time_s",
    "frame_index",
    "tag_raw",
    "tag_sensor",
    "tag_name",
    "tag_counter",
    "tag_parity",
    "sample_timestamp_raw",
    "sample_timestamp_us",
    "estimated_host_timestamp_us",
    "estimated_host_time_s",
    "x_raw",
    "y_raw",
    "z_raw",
    "accel_x_g",
    "accel_y_g",
    "accel_z_g",
    "gyro_x_dps",
    "gyro_y_dps",
    "gyro_z_dps",
    "raw_hex",
]

CONFIG_FIELDS = [
    "record_seq",
    "record_flags",
    "host_timestamp_us",
    "host_time_s",
    "analog_rate_hz",
    "analog_resolution_bits",
    "analog_averaging",
    "button_debounce_ms",
    "button_hold_start_ms",
    "button_hold_stop_ms",
    "reed_debounce_us",
    "wheel_circumference_m",
    "imu_accel_range_g",
    "imu_gyro_range_dps",
    "imu_odr_hz",
    "imu_watermark_frames",
    "imu_fifo_mode",
    "imu_timestamp_decimation",
]

TIMELINE_FIELDS = [
    "event_type",
    "record_seq",
    "record_tag",
    "record_flags",
    "host_timestamp_us",
    "host_time_s",
    "sample_index",
    "front_raw",
    "rear_raw",
    "front_position_counts",
    "rear_position_counts",
    "front_norm",
    "rear_norm",
    "front_velocity_counts_per_s",
    "rear_velocity_counts_per_s",
    "front_mm",
    "rear_mm",
    "front_velocity_mm_per_s",
    "rear_velocity_mm_per_s",
    "pulse_count",
    "interval_us",
    "speed_m_s",
    "speed_kph",
    "speed_mph",
    "stat_code",
    "stat_name",
    "stat_flags",
    "stat_value0",
    "stat_value1",
    "imu_burst_index",
    "imu_burst_host_timestamp_us",
    "imu_frame_index",
    "imu_tag_raw",
    "imu_tag_sensor",
    "imu_tag_name",
    "imu_tag_counter",
    "imu_tag_parity",
    "imu_sample_timestamp_raw",
    "imu_sample_timestamp_us",
    "imu_x_raw",
    "imu_y_raw",
    "imu_z_raw",
    "imu_accel_x_g",
    "imu_accel_y_g",
    "imu_accel_z_g",
    "imu_gyro_x_dps",
    "imu_gyro_y_dps",
    "imu_gyro_z_dps",
    "config_analog_rate_hz",
    "config_analog_resolution_bits",
    "config_analog_averaging",
    "config_button_debounce_ms",
    "config_button_hold_start_ms",
    "config_button_hold_stop_ms",
    "config_reed_debounce_us",
    "config_wheel_circumference_m",
    "config_imu_accel_range_g",
    "config_imu_gyro_range_dps",
    "config_imu_odr_hz",
    "config_imu_watermark_frames",
    "config_imu_fifo_mode",
    "config_imu_timestamp_decimation",
]

HOST_TIMESTAMP_WRAP_US = 1 << 32
HOST_TIMESTAMP_WRAP_THRESHOLD_US = 1 << 31


def normalize_value(value):
    if value is None:
        return ""
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: normalize_value(row.get(name)) for name in fieldnames})


def resolve_mm_per_count(explicit_scale: float | None, full_scale_mm: float | None, analog_resolution_bits: int) -> float | None:
    if explicit_scale is not None:
        return explicit_scale
    if full_scale_mm is None:
        return None
    adc_max_count = (1 << analog_resolution_bits) - 1
    return full_scale_mm / adc_max_count


def load_config_record(session: dict) -> tuple[dict | None, dict | None]:
    for record in session["records"]:
        if record["header"]["tag"] == "CFG ":
            return record["header"], record["payload"]
    return None, None


def derivative_per_second(rows: list[dict], index: int, key: str) -> float | None:
    if len(rows) < 2:
        return None
    if index == 0:
        left = rows[0]
        right = rows[1]
    elif index == len(rows) - 1:
        left = rows[-2]
        right = rows[-1]
    else:
        left = rows[index - 1]
        right = rows[index + 1]

    dt_us = right["host_timestamp_us"] - left["host_timestamp_us"]
    if dt_us <= 0:
        return None
    return (right[key] - left[key]) * 1_000_000.0 / dt_us


def build_unwrapped_host_timestamps(session: dict) -> dict[int, int]:
    unwrapped: dict[int, int] = {}
    wrap_offset = 0
    previous_raw: int | None = None
    for record in session["records"]:
        header = record["header"]
        raw_timestamp = int(header["timestamp_us"])
        if previous_raw is not None and raw_timestamp < previous_raw:
            if previous_raw - raw_timestamp > HOST_TIMESTAMP_WRAP_THRESHOLD_US:
                wrap_offset += HOST_TIMESTAMP_WRAP_US
        unwrapped[int(header["seq"])] = raw_timestamp + wrap_offset
        previous_raw = raw_timestamp
    return unwrapped


def record_host_timestamp_us(header: dict, unwrapped_timestamps: dict[int, int] | None = None) -> int:
    if unwrapped_timestamps is None:
        return int(header["timestamp_us"])
    return int(unwrapped_timestamps.get(int(header["seq"]), int(header["timestamp_us"])))


def build_analog_rows(
    session: dict,
    front_mm_per_count: float | None,
    rear_mm_per_count: float | None,
    front_zero_count: int,
    rear_zero_count: int,
    front_sign: int,
    rear_sign: int,
    unwrapped_timestamps: dict[int, int] | None = None,
) -> list[dict]:
    analog_resolution_bits = session["header"]["analog_resolution_bits"]
    adc_max_count = (1 << analog_resolution_bits) - 1
    rows: list[dict] = []

    for record in session["records"]:
        if record["header"]["tag"] != "ANLG":
            continue
        header = record["header"]
        payload = record["payload"]
        host_timestamp_us = record_host_timestamp_us(header, unwrapped_timestamps)
        front_position_counts = front_sign * (payload["front_raw"] - front_zero_count)
        rear_position_counts = rear_sign * (payload["rear_raw"] - rear_zero_count)
        row = {
            "record_seq": header["seq"],
            "record_flags": header["flags"],
            "host_timestamp_us": host_timestamp_us,
            "host_time_s": host_timestamp_us / 1_000_000.0,
            "dt_us": None if not rows else host_timestamp_us - rows[-1]["host_timestamp_us"],
            "sample_index": payload["sample_index"],
            "front_raw": payload["front_raw"],
            "rear_raw": payload["rear_raw"],
            "front_position_counts": front_position_counts,
            "rear_position_counts": rear_position_counts,
            "front_norm": payload["front_raw"] / adc_max_count,
            "rear_norm": payload["rear_raw"] / adc_max_count,
        }
        if front_mm_per_count is not None:
            row["front_mm"] = front_position_counts * front_mm_per_count
        if rear_mm_per_count is not None:
            row["rear_mm"] = rear_position_counts * rear_mm_per_count
        rows.append(row)

    for index, row in enumerate(rows):
        row["front_velocity_counts_per_s"] = derivative_per_second(rows, index, "front_position_counts")
        row["rear_velocity_counts_per_s"] = derivative_per_second(rows, index, "rear_position_counts")
        if front_mm_per_count is not None and row["front_velocity_counts_per_s"] is not None:
            row["front_velocity_mm_per_s"] = row["front_velocity_counts_per_s"] * front_mm_per_count
        if rear_mm_per_count is not None and row["rear_velocity_counts_per_s"] is not None:
            row["rear_velocity_mm_per_s"] = row["rear_velocity_counts_per_s"] * rear_mm_per_count
    return rows


def build_wheel_rows(session: dict, unwrapped_timestamps: dict[int, int] | None = None) -> list[dict]:
    wheel_circumference_m = None
    if session["config"] is not None:
        wheel_circumference_m = session["config"].get("wheel_circumference_m")

    rows = []
    for record in session["records"]:
        if record["header"]["tag"] != "WSPD":
            continue
        header = record["header"]
        payload = record["payload"]
        host_timestamp_us = record_host_timestamp_us(header, unwrapped_timestamps)
        interval_us = payload["interval_us"]
        speed_m_s = None
        if interval_us > 0 and wheel_circumference_m is not None:
            speed_m_s = wheel_circumference_m / (interval_us / 1_000_000.0)
        rows.append(
            {
                "record_seq": header["seq"],
                "record_flags": header["flags"],
                "host_timestamp_us": host_timestamp_us,
                "host_time_s": host_timestamp_us / 1_000_000.0,
                "pulse_count": payload["pulse_count"],
                "interval_us": interval_us,
                "period_s": None if interval_us <= 0 else interval_us / 1_000_000.0,
                "speed_m_s": speed_m_s,
                "speed_kph": None if speed_m_s is None else speed_m_s * 3.6,
                "speed_mph": None if speed_m_s is None else speed_m_s * 2.2369362920544,
            }
        )
    return rows


def build_stat_rows(session: dict, unwrapped_timestamps: dict[int, int] | None = None) -> list[dict]:
    rows = []
    for record in session["records"]:
        if record["header"]["tag"] != "STAT":
            continue
        header = record["header"]
        payload = record["payload"]
        host_timestamp_us = record_host_timestamp_us(header, unwrapped_timestamps)
        rows.append(
            {
                "record_seq": header["seq"],
                "record_flags": header["flags"],
                "host_timestamp_us": host_timestamp_us,
                "host_time_s": host_timestamp_us / 1_000_000.0,
                "code": payload["code"],
                "name": payload["name"],
                "flags": payload["flags"],
                "value0": payload["value0"],
                "value1": payload["value1"],
            }
        )
    return rows


def build_imu_rows(session: dict, unwrapped_timestamps: dict[int, int] | None = None) -> tuple[list[dict], list[dict]]:
    burst_rows = []
    frame_rows = []

    for record in session["records"]:
        if record["header"]["tag"] != "IMU ":
            continue
        header = record["header"]
        payload = record["payload"]
        burst_host_timestamp_us = record_host_timestamp_us(header, unwrapped_timestamps)

        burst_rows.append(
            {
                "record_seq": header["seq"],
                "record_flags": header["flags"],
                "burst_index": payload["burst_index"],
                "burst_host_timestamp_us": burst_host_timestamp_us,
                "burst_host_time_s": burst_host_timestamp_us / 1_000_000.0,
                "frame_count": payload["frame_count"],
                "accel_frames": payload["accel_frames"],
                "gyro_frames": payload["gyro_frames"],
                "timestamp_frames": payload["timestamp_frames"],
                "fifo_level_before": payload["fifo_level_before"],
                "fifo_level_after": payload["fifo_level_after"],
                "burst_flags": payload["flags"],
                "first_timestamp_raw": payload["first_timestamp"],
                "first_timestamp_us": payload["first_timestamp_us"],
                "last_timestamp_raw": payload["last_timestamp"],
                "last_timestamp_us": payload["last_timestamp_us"],
            }
        )

        for frame in payload.get("frames", []):
            frame_rows.append(
                {
                    "record_seq": header["seq"],
                    "record_flags": header["flags"],
                    "burst_index": payload["burst_index"],
                    "burst_host_timestamp_us": burst_host_timestamp_us,
                    "burst_host_time_s": burst_host_timestamp_us / 1_000_000.0,
                    "frame_index": frame["frame_index"],
                    "tag_raw": frame["tag_raw"],
                    "tag_sensor": frame["tag_sensor"],
                    "tag_name": frame["tag_name"],
                    "tag_counter": frame["tag_counter"],
                    "tag_parity": frame["tag_parity"],
                    "sample_timestamp_raw": frame.get("sample_timestamp_raw", frame.get("timestamp_raw")),
                    "sample_timestamp_us": frame.get("sample_timestamp_us", frame.get("timestamp_us")),
                    "estimated_host_timestamp_us": frame.get("estimated_host_timestamp_us"),
                    "estimated_host_time_s": None
                    if frame.get("estimated_host_timestamp_us") is None
                    else frame["estimated_host_timestamp_us"] / 1_000_000.0,
                    "x_raw": frame.get("x_raw"),
                    "y_raw": frame.get("y_raw"),
                    "z_raw": frame.get("z_raw"),
                    "accel_x_g": frame.get("x_g"),
                    "accel_y_g": frame.get("y_g"),
                    "accel_z_g": frame.get("z_g"),
                    "gyro_x_dps": frame.get("x_dps"),
                    "gyro_y_dps": frame.get("y_dps"),
                    "gyro_z_dps": frame.get("z_dps"),
                    "raw_hex": frame.get("raw_hex"),
                }
            )

    return burst_rows, frame_rows


def build_config_rows(session: dict, unwrapped_timestamps: dict[int, int] | None = None) -> list[dict]:
    record_header, config = load_config_record(session)
    if record_header is None or config is None:
        return []
    host_timestamp_us = record_host_timestamp_us(record_header, unwrapped_timestamps)
    return [
        {
            "record_seq": record_header["seq"],
            "record_flags": record_header["flags"],
            "host_timestamp_us": host_timestamp_us,
            "host_time_s": host_timestamp_us / 1_000_000.0,
            **config,
        }
    ]


def build_timeline_rows(
    config_rows: list[dict],
    analog_rows: list[dict],
    wheel_rows: list[dict],
    stat_rows: list[dict],
    imu_frame_rows: list[dict],
) -> list[dict]:
    rows = []

    for row in config_rows:
        rows.append(
            {
                "event_type": "config",
                "record_seq": row["record_seq"],
                "record_tag": "CFG ",
                "record_flags": row["record_flags"],
                "host_timestamp_us": row["host_timestamp_us"],
                "host_time_s": row["host_time_s"],
                "config_analog_rate_hz": row["analog_rate_hz"],
                "config_analog_resolution_bits": row["analog_resolution_bits"],
                "config_analog_averaging": row["analog_averaging"],
                "config_button_debounce_ms": row["button_debounce_ms"],
                "config_button_hold_start_ms": row["button_hold_start_ms"],
                "config_button_hold_stop_ms": row["button_hold_stop_ms"],
                "config_reed_debounce_us": row["reed_debounce_us"],
                "config_wheel_circumference_m": row["wheel_circumference_m"],
                "config_imu_accel_range_g": row["imu_accel_range_g"],
                "config_imu_gyro_range_dps": row["imu_gyro_range_dps"],
                "config_imu_odr_hz": row["imu_odr_hz"],
                "config_imu_watermark_frames": row["imu_watermark_frames"],
                "config_imu_fifo_mode": row["imu_fifo_mode"],
                "config_imu_timestamp_decimation": row["imu_timestamp_decimation"],
            }
        )

    for row in stat_rows:
        rows.append(
            {
                "event_type": "status",
                "record_seq": row["record_seq"],
                "record_tag": "STAT",
                "record_flags": row["record_flags"],
                "host_timestamp_us": row["host_timestamp_us"],
                "host_time_s": row["host_time_s"],
                "stat_code": row["code"],
                "stat_name": row["name"],
                "stat_flags": row["flags"],
                "stat_value0": row["value0"],
                "stat_value1": row["value1"],
            }
        )

    for row in analog_rows:
        rows.append(
            {
                "event_type": "analog",
                "record_seq": row["record_seq"],
                "record_tag": "ANLG",
                "record_flags": row["record_flags"],
                "host_timestamp_us": row["host_timestamp_us"],
                "host_time_s": row["host_time_s"],
                "sample_index": row["sample_index"],
                "front_raw": row["front_raw"],
                "rear_raw": row["rear_raw"],
                "front_position_counts": row["front_position_counts"],
                "rear_position_counts": row["rear_position_counts"],
                "front_norm": row["front_norm"],
                "rear_norm": row["rear_norm"],
                "front_velocity_counts_per_s": row["front_velocity_counts_per_s"],
                "rear_velocity_counts_per_s": row["rear_velocity_counts_per_s"],
                "front_mm": row.get("front_mm"),
                "rear_mm": row.get("rear_mm"),
                "front_velocity_mm_per_s": row.get("front_velocity_mm_per_s"),
                "rear_velocity_mm_per_s": row.get("rear_velocity_mm_per_s"),
            }
        )

    for row in wheel_rows:
        rows.append(
            {
                "event_type": "wheel",
                "record_seq": row["record_seq"],
                "record_tag": "WSPD",
                "record_flags": row["record_flags"],
                "host_timestamp_us": row["host_timestamp_us"],
                "host_time_s": row["host_time_s"],
                "pulse_count": row["pulse_count"],
                "interval_us": row["interval_us"],
                "speed_m_s": row["speed_m_s"],
                "speed_kph": row["speed_kph"],
                "speed_mph": row["speed_mph"],
            }
        )

    for row in imu_frame_rows:
        host_timestamp_us = row["estimated_host_timestamp_us"]
        if host_timestamp_us is None:
            host_timestamp_us = row["burst_host_timestamp_us"]
        rows.append(
            {
                "event_type": f"imu_{row['tag_name'].lower()}",
                "record_seq": row["record_seq"],
                "record_tag": "IMU ",
                "record_flags": row["record_flags"],
                "host_timestamp_us": host_timestamp_us,
                "host_time_s": host_timestamp_us / 1_000_000.0,
                "imu_burst_index": row["burst_index"],
                "imu_burst_host_timestamp_us": row["burst_host_timestamp_us"],
                "imu_frame_index": row["frame_index"],
                "imu_tag_raw": row["tag_raw"],
                "imu_tag_sensor": row["tag_sensor"],
                "imu_tag_name": row["tag_name"],
                "imu_tag_counter": row["tag_counter"],
                "imu_tag_parity": row["tag_parity"],
                "imu_sample_timestamp_raw": row["sample_timestamp_raw"],
                "imu_sample_timestamp_us": row["sample_timestamp_us"],
                "imu_x_raw": row["x_raw"],
                "imu_y_raw": row["y_raw"],
                "imu_z_raw": row["z_raw"],
                "imu_accel_x_g": row["accel_x_g"],
                "imu_accel_y_g": row["accel_y_g"],
                "imu_accel_z_g": row["accel_z_g"],
                "imu_gyro_x_dps": row["gyro_x_dps"],
                "imu_gyro_y_dps": row["gyro_y_dps"],
                "imu_gyro_z_dps": row["gyro_z_dps"],
            }
        )

    rows.sort(key=lambda row: (row["host_timestamp_us"], row["record_seq"], row["event_type"]))
    return rows


def summarize_session(
    session: dict,
    analog_rows: list[dict],
    wheel_rows: list[dict],
    stat_rows: list[dict],
    imu_burst_rows: list[dict],
    imu_frame_rows: list[dict],
    output_dir: Path,
    front_mm_per_count: float | None,
    rear_mm_per_count: float | None,
    unwrapped_timestamps: dict[int, int] | None = None,
) -> dict:
    all_host_timestamps = [
        record_host_timestamp_us(record["header"], unwrapped_timestamps) for record in session["records"]
    ]
    source_path = Path(session["path"])
    source_stat = source_path.stat() if source_path.exists() else None
    summary = {
        "source_path": str(source_path),
        "file_size_bytes": session["file_size_bytes"],
        "source_mtime_ns": None if source_stat is None else source_stat.st_mtime_ns,
        "output_dir": str(output_dir),
        "header": session["header"],
        "config": session["config"],
        "record_counts": dict(session["record_counts"]),
        "derived_scales": {
            "front_mm_per_count": front_mm_per_count,
            "rear_mm_per_count": rear_mm_per_count,
        },
        "counts": {
            "analog_rows": len(analog_rows),
            "wheel_rows": len(wheel_rows),
            "stat_rows": len(stat_rows),
            "imu_burst_rows": len(imu_burst_rows),
            "imu_frame_rows": len(imu_frame_rows),
        },
        "timing": {
            "first_host_timestamp_us": min(all_host_timestamps) if all_host_timestamps else None,
            "last_host_timestamp_us": max(all_host_timestamps) if all_host_timestamps else None,
            "duration_us": None if not all_host_timestamps else max(all_host_timestamps) - min(all_host_timestamps),
        },
    }
    if analog_rows:
        summary["analog_extents"] = {
            "front_raw_min": min(row["front_raw"] for row in analog_rows),
            "front_raw_max": max(row["front_raw"] for row in analog_rows),
            "rear_raw_min": min(row["rear_raw"] for row in analog_rows),
            "rear_raw_max": max(row["rear_raw"] for row in analog_rows),
        }
    return summary


def export_session(
    session: dict,
    output_dir: Path,
    front_mm_per_count: float | None,
    rear_mm_per_count: float | None,
    front_zero_count: int,
    rear_zero_count: int,
    front_sign: int,
    rear_sign: int,
) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    unwrapped_timestamps = build_unwrapped_host_timestamps(session)

    analog_rows = build_analog_rows(
        session,
        front_mm_per_count=front_mm_per_count,
        rear_mm_per_count=rear_mm_per_count,
        front_zero_count=front_zero_count,
        rear_zero_count=rear_zero_count,
        front_sign=front_sign,
        rear_sign=rear_sign,
        unwrapped_timestamps=unwrapped_timestamps,
    )
    wheel_rows = build_wheel_rows(session, unwrapped_timestamps=unwrapped_timestamps)
    stat_rows = build_stat_rows(session, unwrapped_timestamps=unwrapped_timestamps)
    imu_burst_rows, imu_frame_rows = build_imu_rows(session, unwrapped_timestamps=unwrapped_timestamps)
    config_rows = build_config_rows(session, unwrapped_timestamps=unwrapped_timestamps)
    timeline_rows = build_timeline_rows(config_rows, analog_rows, wheel_rows, stat_rows, imu_frame_rows)

    write_csv(output_dir / "config.csv", CONFIG_FIELDS, config_rows)
    write_csv(output_dir / "analog.csv", ANALOG_FIELDS, analog_rows)
    write_csv(output_dir / "wheel.csv", WHEEL_FIELDS, wheel_rows)
    write_csv(output_dir / "status.csv", STAT_FIELDS, stat_rows)
    write_csv(output_dir / "imu_bursts.csv", IMU_BURST_FIELDS, imu_burst_rows)
    write_csv(output_dir / "imu_frames.csv", IMU_FRAME_FIELDS, imu_frame_rows)
    write_csv(output_dir / "timeline.csv", TIMELINE_FIELDS, timeline_rows)

    summary = summarize_session(
        session,
        analog_rows=analog_rows,
        wheel_rows=wheel_rows,
        stat_rows=stat_rows,
        imu_burst_rows=imu_burst_rows,
        imu_frame_rows=imu_frame_rows,
        output_dir=output_dir,
        front_mm_per_count=front_mm_per_count,
        rear_mm_per_count=rear_mm_per_count,
        unwrapped_timestamps=unwrapped_timestamps,
    )
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    return summary


def export_bin_to_directory(
    bin_path: Path,
    output_dir: Path | None = None,
    front_mm_per_count: float | None = None,
    rear_mm_per_count: float | None = None,
    front_full_scale_mm: float | None = None,
    rear_full_scale_mm: float | None = None,
    front_zero_count: int = 0,
    rear_zero_count: int = 0,
    front_invert: bool = False,
    rear_invert: bool = False,
) -> tuple[dict, Path]:
    session = parse_log(bin_path)
    analog_resolution_bits = session["header"]["analog_resolution_bits"]
    resolved_front_mm = resolve_mm_per_count(front_mm_per_count, front_full_scale_mm, analog_resolution_bits)
    resolved_rear_mm = resolve_mm_per_count(rear_mm_per_count, rear_full_scale_mm, analog_resolution_bits)

    if output_dir is None:
        output_dir = Path("exports") / bin_path.stem

    summary = export_session(
        session=session,
        output_dir=output_dir,
        front_mm_per_count=resolved_front_mm,
        rear_mm_per_count=resolved_rear_mm,
        front_zero_count=front_zero_count,
        rear_zero_count=rear_zero_count,
        front_sign=-1 if front_invert else 1,
        rear_sign=-1 if rear_invert else 1,
    )
    return summary, output_dir
