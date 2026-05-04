#!/usr/bin/env python3
import collections
import pathlib
import struct


HEADER_STRUCT = struct.Struct("<8sHHII24sIII4B4H8B6I")
RECORD_HEADER_STRUCT = struct.Struct("<4sHHII")

CONFIG_STRUCT = struct.Struct("<HBB4If4HBBH")
STAT_STRUCT = struct.Struct("<HHii")
ANLG_STRUCT = struct.Struct("<IHH")
WSPD_STRUCT = struct.Struct("<II")
IMU_BURST_HEADER_STRUCT = struct.Struct("<I7H2I")
IMU_FRAME_STRUCT = struct.Struct("<B6s")
IMU_VECTOR_STRUCT = struct.Struct("<hhh")
IMU_TIMESTAMP_STRUCT = struct.Struct("<I")

TIMESTAMP_TICK_US = 25

STAT_CODES = {
    1: "BOOT",
    2: "RTC_VALID",
    3: "RTC_INVALID",
    4: "SD_READY",
    5: "SD_ERROR",
    6: "IMU_READY",
    7: "IMU_ERROR",
    8: "SESSION_START",
    9: "SESSION_STOP",
    10: "CLEAN_CLOSE",
    11: "QUEUE_OVERFLOW",
    12: "ANALOG_OVERRUN",
    13: "REED_OVERFLOW",
    14: "IMU_FIFO_OVERFLOW",
    15: "WRITER_ERROR",
    16: "IMU_READ_ERROR",
}

SESSION_STOP_REASON_NAMES = {
    0: "UNKNOWN",
    1: "SWITCH_OFF",
    2: "VALIDATION_AUTORUN",
}

IMU_TAG_NAMES = {
    1: "GYRO_NC",
    2: "ACCEL_NC",
    3: "TEMPERATURE",
    4: "TIMESTAMP",
    5: "CFG_CHANGE",
    6: "ACCEL_NC_T_2",
    7: "ACCEL_NC_T_1",
    8: "ACCEL_2XC",
    9: "ACCEL_3XC",
    10: "GYRO_NC_T_2",
    11: "GYRO_NC_T_1",
    12: "GYRO_2XC",
    13: "GYRO_3XC",
    14: "SENSORHUB_SLAVE0",
    15: "SENSORHUB_SLAVE1",
    16: "SENSORHUB_SLAVE2",
    17: "SENSORHUB_SLAVE3",
    18: "STEP_COUNTER",
    19: "GAME_ROTATION",
    20: "GEOMAG_ROTATION",
    21: "ROTATION",
    25: "SENSORHUB_NACK",
}

ACCEL_MG_PER_LSB = {
    2: 0.061,
    4: 0.122,
    8: 0.244,
    16: 0.488,
}

GYRO_MDPS_PER_LSB = {
    125: 4.375,
    250: 8.75,
    500: 17.5,
    1000: 35.0,
    2000: 70.0,
    4000: 140.0,
}


def fourcc(value: bytes) -> str:
    return value.decode("ascii", errors="replace")


def decode_header(data: bytes) -> dict:
    values = HEADER_STRUCT.unpack(data)
    return {
        "magic": values[0].rstrip(b"\0").decode("ascii", errors="replace"),
        "format_version": values[1],
        "header_size": values[2],
        "firmware_version": values[3],
        "build_epoch": values[4],
        "build_id": values[5].rstrip(b"\0").decode("ascii", errors="replace"),
        "start_epoch": values[6],
        "start_micros": values[7],
        "session_index": values[8],
        "rtc_valid": values[9],
        "enabled_mask": values[10],
        "analog_resolution_bits": values[11],
        "analog_averaging": values[12],
        "analog_rate_hz": values[13],
        "imu_rate_hz": values[14],
        "imu_watermark_frames": values[15],
        "record_header_size": values[16],
        "button_pin": values[17],
        "led_pin": values[18],
        "reed_pin": values[19],
        "imu_int1_pin": values[20],
        "front_pot_pin": values[21],
        "rear_pot_pin": values[22],
        "imu_cs_pin": values[23],
    }


def unpack_record_header(data: bytes) -> dict:
    tag, payload_len, flags, seq, timestamp_us = RECORD_HEADER_STRUCT.unpack(data)
    return {
        "tag": fourcc(tag),
        "payload_len": payload_len,
        "flags": flags,
        "seq": seq,
        "timestamp_us": timestamp_us,
    }


def decode_config(payload: bytes) -> dict:
    values = CONFIG_STRUCT.unpack(payload)
    return {
        "analog_rate_hz": values[0],
        "analog_resolution_bits": values[1],
        "analog_averaging": values[2],
        "button_debounce_ms": values[3],
        "button_hold_start_ms": values[4],
        "button_hold_stop_ms": values[5],
        "reed_debounce_us": values[6],
        "wheel_circumference_m": values[7],
        "imu_accel_range_g": values[8],
        "imu_gyro_range_dps": values[9],
        "imu_odr_hz": values[10],
        "imu_watermark_frames": values[11],
        "imu_fifo_mode": values[12],
        "imu_timestamp_decimation": values[13],
    }


def decode_stat(payload: bytes) -> dict:
    code, flags, value0, value1 = STAT_STRUCT.unpack(payload)
    decoded = {
        "code": code,
        "name": STAT_CODES.get(code, f"UNKNOWN_{code}"),
        "flags": flags,
        "value0": value0,
        "value1": value1,
    }
    if code == 9:
        decoded["stop_reason_name"] = SESSION_STOP_REASON_NAMES.get(value1, f"UNKNOWN_{value1}")
    return decoded


def decode_anlg(payload: bytes) -> dict:
    sample_index, front_raw, rear_raw = ANLG_STRUCT.unpack(payload)
    return {
        "sample_index": sample_index,
        "front_raw": front_raw,
        "rear_raw": rear_raw,
    }


def decode_wspd(payload: bytes) -> dict:
    pulse_count, interval_us = WSPD_STRUCT.unpack(payload)
    return {
        "pulse_count": pulse_count,
        "interval_us": interval_us,
    }


def accel_raw_to_g(raw_value: int, accel_range_g: int | None) -> float | None:
    mg_per_lsb = ACCEL_MG_PER_LSB.get(accel_range_g)
    if mg_per_lsb is None:
        return None
    return raw_value * mg_per_lsb / 1000.0


def gyro_raw_to_dps(raw_value: int, gyro_range_dps: int | None) -> float | None:
    mdps_per_lsb = GYRO_MDPS_PER_LSB.get(gyro_range_dps)
    if mdps_per_lsb is None:
        return None
    return raw_value * mdps_per_lsb / 1000.0


def estimate_host_timestamp_us(record_header: dict, burst_header: dict, sample_timestamp_raw: int | None) -> int | None:
    if sample_timestamp_raw is None:
        return None
    last_timestamp_raw = burst_header.get("last_timestamp")
    if not last_timestamp_raw:
        return None
    delta_ticks = last_timestamp_raw - sample_timestamp_raw
    return int(record_header["timestamp_us"] - delta_ticks * TIMESTAMP_TICK_US)


def decode_imu(payload: bytes, record_header: dict, config: dict | None) -> dict:
    header = IMU_BURST_HEADER_STRUCT.unpack(payload[: IMU_BURST_HEADER_STRUCT.size])
    frame_bytes = payload[IMU_BURST_HEADER_STRUCT.size :]
    accel_range_g = None if config is None else config.get("imu_accel_range_g")
    gyro_range_dps = None if config is None else config.get("imu_gyro_range_dps")

    burst = {
        "burst_index": header[0],
        "frame_count": header[1],
        "accel_frames": header[2],
        "gyro_frames": header[3],
        "timestamp_frames": header[4],
        "fifo_level_before": header[5],
        "fifo_level_after": header[6],
        "flags": header[7],
        "first_timestamp": header[8],
        "last_timestamp": header[9],
        "first_timestamp_us": header[8] * TIMESTAMP_TICK_US if header[8] else None,
        "last_timestamp_us": header[9] * TIMESTAMP_TICK_US if header[9] else None,
        "frames": [],
    }

    active_timestamp_raw = None
    for frame_index, offset in enumerate(range(0, len(frame_bytes), IMU_FRAME_STRUCT.size)):
        chunk = frame_bytes[offset : offset + IMU_FRAME_STRUCT.size]
        if len(chunk) < IMU_FRAME_STRUCT.size:
            break

        tag_raw, raw_bytes = IMU_FRAME_STRUCT.unpack(chunk)
        tag_sensor = tag_raw >> 3
        frame = {
            "frame_index": frame_index,
            "tag_raw": tag_raw,
            "tag_sensor": tag_sensor,
            "tag_name": IMU_TAG_NAMES.get(tag_sensor, f"UNKNOWN_{tag_sensor}"),
            "tag_counter": (tag_raw >> 1) & 0x03,
            "tag_parity": tag_raw & 0x01,
            "raw_hex": raw_bytes.hex(),
        }

        if tag_sensor == 4:
            timestamp_raw = IMU_TIMESTAMP_STRUCT.unpack(raw_bytes[:4])[0]
            active_timestamp_raw = timestamp_raw
            frame["timestamp_raw"] = timestamp_raw
            frame["timestamp_us"] = timestamp_raw * TIMESTAMP_TICK_US
            frame["estimated_host_timestamp_us"] = estimate_host_timestamp_us(record_header, burst, timestamp_raw)
        elif tag_sensor in (1, 2):
            x_raw, y_raw, z_raw = IMU_VECTOR_STRUCT.unpack(raw_bytes)
            frame["sample_timestamp_raw"] = active_timestamp_raw
            frame["sample_timestamp_us"] = None if active_timestamp_raw is None else active_timestamp_raw * TIMESTAMP_TICK_US
            frame["estimated_host_timestamp_us"] = estimate_host_timestamp_us(record_header, burst, active_timestamp_raw)
            frame["x_raw"] = x_raw
            frame["y_raw"] = y_raw
            frame["z_raw"] = z_raw
            if tag_sensor == 1:
                frame["x_dps"] = gyro_raw_to_dps(x_raw, gyro_range_dps)
                frame["y_dps"] = gyro_raw_to_dps(y_raw, gyro_range_dps)
                frame["z_dps"] = gyro_raw_to_dps(z_raw, gyro_range_dps)
            else:
                frame["x_g"] = accel_raw_to_g(x_raw, accel_range_g)
                frame["y_g"] = accel_raw_to_g(y_raw, accel_range_g)
                frame["z_g"] = accel_raw_to_g(z_raw, accel_range_g)
        else:
            frame["sample_timestamp_raw"] = active_timestamp_raw
            frame["sample_timestamp_us"] = None if active_timestamp_raw is None else active_timestamp_raw * TIMESTAMP_TICK_US
            frame["estimated_host_timestamp_us"] = estimate_host_timestamp_us(record_header, burst, active_timestamp_raw)

        burst["frames"].append(frame)

    burst["frames_preview"] = burst["frames"][:5]
    return burst


def decode_record_payload(tag: str, payload: bytes, record_header: dict, config: dict | None) -> dict:
    if tag == "CFG " and len(payload) == CONFIG_STRUCT.size:
        return decode_config(payload)
    if tag == "STAT" and len(payload) == STAT_STRUCT.size:
        return decode_stat(payload)
    if tag == "ANLG" and len(payload) == ANLG_STRUCT.size:
        return decode_anlg(payload)
    if tag == "WSPD" and len(payload) == WSPD_STRUCT.size:
        return decode_wspd(payload)
    if tag == "IMU " and len(payload) >= IMU_BURST_HEADER_STRUCT.size:
        return decode_imu(payload, record_header, config)
    return {"raw_len": len(payload), "raw_hex": payload.hex()}


def parse_log(path: pathlib.Path) -> dict:
    raw = path.read_bytes()
    if len(raw) < HEADER_STRUCT.size:
        raise ValueError("file too small for header")

    header = decode_header(raw[: HEADER_STRUCT.size])
    if header["magic"] != "STLOG1":
        raise ValueError(f"unexpected header magic: {header['magic']!r}")
    if header["header_size"] != HEADER_STRUCT.size:
        raise ValueError(
            f"unexpected header size: file says {header['header_size']}, parser expects {HEADER_STRUCT.size}"
        )
    if header["record_header_size"] != RECORD_HEADER_STRUCT.size:
        raise ValueError(
            "unexpected record header size: "
            f"file says {header['record_header_size']}, parser expects {RECORD_HEADER_STRUCT.size}"
        )

    offset = HEADER_STRUCT.size
    counts = collections.Counter()
    records = []
    config = None

    while offset + RECORD_HEADER_STRUCT.size <= len(raw):
        record_header = unpack_record_header(raw[offset : offset + RECORD_HEADER_STRUCT.size])
        offset += RECORD_HEADER_STRUCT.size

        payload_end = offset + record_header["payload_len"]
        if payload_end > len(raw):
            raise ValueError(
                f"truncated payload for record seq {record_header['seq']} tag {record_header['tag']}"
            )

        payload = raw[offset:payload_end]
        offset = payload_end
        decoded_payload = decode_record_payload(record_header["tag"], payload, record_header, config)
        if record_header["tag"] == "CFG " and "raw_len" not in decoded_payload:
            config = decoded_payload

        records.append(
            {
                "header": record_header,
                "payload": decoded_payload,
            }
        )
        counts[record_header["tag"]] += 1

    if offset != len(raw):
        raise ValueError(f"trailing {len(raw) - offset} byte(s) after last complete record")

    return {
        "path": path,
        "file_size_bytes": len(raw),
        "header": header,
        "config": config,
        "records": records,
        "record_counts": counts,
    }
