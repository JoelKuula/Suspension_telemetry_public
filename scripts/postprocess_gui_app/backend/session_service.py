from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from .analysis_service import build_derived_analog, estimate_default_zero_count, load_export_analog
from .export_service import export_bin_to_directory
from .session_metadata_service import load_session_metadata
from .session_config import load_or_create_session_config, save_session_config

WHEEL_SPEED_MAX_VALID_KPH = 200.0

REQUIRED_EXPORT_FILES = (
    "summary.json",
    "analog.csv",
    "config.csv",
    "status.csv",
    "timeline.csv",
)


@dataclass
class SessionBundle:
    source_path: Path | None
    export_dir: Path
    summary: dict[str, Any]
    status_rows: list[dict[str, str]]
    config_rows: list[dict[str, str]]
    session_config: dict[str, Any]
    derived_df: pl.DataFrame
    analog_df: pl.DataFrame
    wheel_df: pl.DataFrame
    imu_frame_df: pl.DataFrame
    session_metadata: dict[str, Any] = field(default_factory=dict)


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_summary(export_dir: Path) -> dict[str, Any]:
    summary_path = export_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"missing summary export: {summary_path}")
    with summary_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_optional_csv_frame(path: Path) -> pl.DataFrame:
    if not path.exists():
        return pl.DataFrame()
    return pl.read_csv(path, null_values=[""], infer_schema_length=1000)


def _float_column(frame: pl.DataFrame, column: str) -> np.ndarray | None:
    if column not in frame.columns:
        return None
    return np.asarray(frame.get_column(column).fill_null(np.nan).to_numpy(), dtype=np.float64)


def sanitize_wheel_speed_frame(
    wheel_df: pl.DataFrame,
    max_speed_kph: float = WHEEL_SPEED_MAX_VALID_KPH,
) -> pl.DataFrame:
    if wheel_df.is_empty() or "speed_kph" not in wheel_df.columns:
        return wheel_df

    speed_kph_raw = _float_column(wheel_df, "speed_kph_raw" if "speed_kph_raw" in wheel_df.columns else "speed_kph")
    if speed_kph_raw is None:
        return wheel_df

    threshold = float(max_speed_kph)
    rejected = np.isfinite(speed_kph_raw) & ((speed_kph_raw < 0.0) | (speed_kph_raw > threshold))
    speed_kph = speed_kph_raw.copy()
    speed_kph[rejected] = np.nan

    columns = [
        pl.Series("speed_kph_raw", speed_kph_raw),
        pl.Series("speed_kph", speed_kph),
        pl.Series("speed_sanity_rejected", rejected),
    ]

    speed_m_s_raw = _float_column(wheel_df, "speed_m_s_raw" if "speed_m_s_raw" in wheel_df.columns else "speed_m_s")
    if speed_m_s_raw is not None:
        speed_m_s = speed_m_s_raw.copy()
        speed_m_s[rejected] = np.nan
        columns.extend([pl.Series("speed_m_s_raw", speed_m_s_raw), pl.Series("speed_m_s", speed_m_s)])

    speed_mph_raw = _float_column(wheel_df, "speed_mph_raw" if "speed_mph_raw" in wheel_df.columns else "speed_mph")
    if speed_mph_raw is not None:
        speed_mph = speed_mph_raw.copy()
        speed_mph[rejected] = np.nan
        columns.extend([pl.Series("speed_mph_raw", speed_mph_raw), pl.Series("speed_mph", speed_mph)])

    return wheel_df.with_columns(columns)


def export_dir_from_bin(bin_path: Path) -> Path:
    return Path("exports") / bin_path.stem


def _summary_source_matches(summary_source: Any, bin_path: Path) -> bool:
    if not summary_source:
        return False
    summary_path = Path(str(summary_source))
    if summary_path.is_absolute():
        try:
            return summary_path.resolve() == bin_path.resolve()
        except OSError:
            return False
    return summary_path.name == bin_path.name


def bin_export_is_current(bin_path: Path, export_dir: Path) -> bool:
    if not bin_path.exists():
        return False
    if any(not (export_dir / filename).exists() for filename in REQUIRED_EXPORT_FILES):
        return False

    summary_path = export_dir / "summary.json"
    try:
        summary = _load_summary(export_dir)
    except (FileNotFoundError, json.JSONDecodeError):
        return False

    bin_stat = bin_path.stat()
    if int(summary.get("file_size_bytes", -1)) != int(bin_stat.st_size):
        return False
    if not _summary_source_matches(summary.get("source_path"), bin_path):
        return False

    source_mtime_ns = summary.get("source_mtime_ns")
    if source_mtime_ns is not None:
        try:
            return int(source_mtime_ns) == int(bin_stat.st_mtime_ns)
        except (TypeError, ValueError):
            return False
    return int(summary_path.stat().st_mtime_ns) >= int(bin_stat.st_mtime_ns)


def _prepare_bundle(
    export_dir: Path,
    source_path: Path | None,
    config_override: dict[str, Any] | None = None,
    force_derived: bool = False,
) -> SessionBundle:
    summary = _load_summary(export_dir)
    analog_df = load_export_analog(export_dir)
    wheel_df = sanitize_wheel_speed_frame(_read_optional_csv_frame(export_dir / "wheel.csv"))
    imu_frame_df = _read_optional_csv_frame(export_dir / "imu_frames.csv")

    front_zero_count = estimate_default_zero_count(analog_df.get_column("front_raw").fill_null(0).to_numpy())
    rear_zero_count = estimate_default_zero_count(analog_df.get_column("rear_raw").fill_null(0).to_numpy())
    session_config = load_or_create_session_config(
        export_dir=export_dir,
        source_path=source_path,
        front_zero_count=front_zero_count,
        rear_zero_count=rear_zero_count,
    )
    if config_override is not None:
        session_config = config_override

    session_config["analog_resolution_bits"] = int(summary["header"]["analog_resolution_bits"])
    derived_df, session_config = build_derived_analog(
        export_dir=export_dir,
        config=session_config,
        analog_df=analog_df,
        force=force_derived,
    )
    save_session_config(export_dir, session_config)

    if source_path is None:
        source_raw = session_config.get("source_path")
        source_path = None if not source_raw else Path(source_raw)
    session_metadata = load_session_metadata(export_dir, summary, source_path)

    return SessionBundle(
        source_path=source_path,
        export_dir=export_dir,
        summary=summary,
        status_rows=_read_csv_rows(export_dir / "status.csv"),
        config_rows=_read_csv_rows(export_dir / "config.csv"),
        session_config=session_config,
        derived_df=derived_df,
        analog_df=analog_df,
        wheel_df=wheel_df,
        imu_frame_df=imu_frame_df,
        session_metadata=session_metadata,
    )


def open_bin_session(bin_path: Path) -> SessionBundle:
    export_dir = export_dir_from_bin(bin_path)
    if not bin_export_is_current(bin_path, export_dir):
        export_bin_to_directory(bin_path=bin_path, output_dir=export_dir)
    return _prepare_bundle(export_dir, source_path=bin_path)


def open_export_session(export_dir: Path) -> SessionBundle:
    return _prepare_bundle(export_dir, source_path=None)


def rebuild_session_analysis(export_dir: Path, session_config: dict[str, Any]) -> SessionBundle:
    source_raw = session_config.get("source_path")
    source_path = None if not source_raw else Path(source_raw)
    return _prepare_bundle(
        export_dir=export_dir,
        source_path=source_path,
        config_override=session_config,
        force_derived=True,
    )
