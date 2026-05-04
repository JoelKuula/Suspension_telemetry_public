from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .session_config import analysis_dir


SCHEMA_VERSION = 2
METADATA_FILENAME = "session_metadata.json"


@dataclass
class SessionDatabaseEntry:
    session_id: str
    export_dir: Path
    source_path: Path | None
    metadata: dict[str, Any]
    summary: dict[str, Any]
    sort_timestamp: int


def session_metadata_path(export_dir: Path) -> Path:
    return analysis_dir(export_dir) / METADATA_FILENAME


def _default_date_text(summary: dict[str, Any]) -> str:
    start_epoch = summary.get("header", {}).get("start_epoch")
    if start_epoch in (None, "", 0):
        return ""
    try:
        timestamp = datetime.fromtimestamp(int(start_epoch), tz=timezone.utc)
    except (OverflowError, OSError, TypeError, ValueError):
        return ""
    return timestamp.strftime("%Y-%m-%d")


def default_session_metadata(
    export_dir: Path,
    summary: dict[str, Any],
    source_path: Path | None,
) -> dict[str, Any]:
    session_id = export_dir.name
    return {
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "source_path": None if source_path is None else str(source_path),
        "export_dir": str(export_dir),
        "date": _default_date_text(summary),
        "track": "",
        "set_label": session_id,
        "set_label_auto": True,
        "comment": "",
        "updated_at": None,
    }


def _merge_metadata(defaults: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = dict(defaults)
    if overrides:
        merged.update(overrides)
    merged["schema_version"] = SCHEMA_VERSION
    merged["session_id"] = defaults["session_id"]
    merged["source_path"] = defaults["source_path"]
    merged["export_dir"] = defaults["export_dir"]
    if "set_label_auto" not in merged:
        merged["set_label_auto"] = str(merged.get("set_label", "")).strip() in ("", defaults["session_id"])
    else:
        merged["set_label_auto"] = bool(merged["set_label_auto"])
    for key in ("date", "track", "set_label", "comment"):
        value = merged.get(key, "")
        merged[key] = "" if value is None else str(value).strip()
    return merged


def load_session_metadata(
    export_dir: Path,
    summary: dict[str, Any],
    source_path: Path | None,
) -> dict[str, Any]:
    defaults = default_session_metadata(export_dir, summary, source_path)
    path = session_metadata_path(export_dir)
    if not path.exists():
        return defaults

    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError):
        return defaults
    return _merge_metadata(defaults, loaded)


def save_session_metadata(export_dir: Path, metadata: dict[str, Any]) -> Path:
    path = session_metadata_path(export_dir)
    path.parent.mkdir(parents=True, exist_ok=True)

    normalized = dict(metadata)
    normalized["schema_version"] = SCHEMA_VERSION
    normalized["updated_at"] = datetime.now(timezone.utc).isoformat()
    for key in ("date", "track", "set_label", "comment"):
        value = normalized.get(key, "")
        normalized[key] = "" if value is None else str(value).strip()
    normalized["set_label_auto"] = bool(normalized.get("set_label_auto", False))

    with path.open("w", encoding="utf-8") as handle:
        json.dump(normalized, handle, indent=2)
        handle.write("\n")
    return path


def _load_summary(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _source_path_from_summary(summary: dict[str, Any]) -> Path | None:
    raw = summary.get("source_path")
    if not raw:
        return None
    return Path(str(raw))


def scan_session_database(exports_root: Path = Path("exports")) -> list[SessionDatabaseEntry]:
    entries: list[SessionDatabaseEntry] = []
    if not exports_root.exists():
        return entries

    for export_dir in sorted(path for path in exports_root.iterdir() if path.is_dir()):
        summary_path = export_dir / "summary.json"
        if not summary_path.exists():
            continue
        try:
            summary = _load_summary(summary_path)
        except (OSError, ValueError, json.JSONDecodeError):
            continue

        source_path = _source_path_from_summary(summary)
        metadata = load_session_metadata(export_dir, summary, source_path)
        start_epoch = summary.get("header", {}).get("start_epoch")
        try:
            sort_timestamp = int(start_epoch) if start_epoch not in (None, "") else 0
        except (TypeError, ValueError):
            sort_timestamp = 0
        entries.append(
            SessionDatabaseEntry(
                session_id=export_dir.name,
                export_dir=export_dir,
                source_path=source_path,
                metadata=metadata,
                summary=summary,
                sort_timestamp=sort_timestamp,
            )
        )

    entries.sort(key=lambda entry: (entry.sort_timestamp, entry.session_id), reverse=True)
    return entries


def auto_assign_set_labels(exports_root: Path = Path("exports")) -> dict[Path, dict[str, Any]]:
    entries = scan_session_database(exports_root)
    by_date: dict[str, list[SessionDatabaseEntry]] = {}
    for entry in entries:
        date_key = str(entry.metadata.get("date", "")).strip() or "undated"
        by_date.setdefault(date_key, []).append(entry)

    updated: dict[Path, dict[str, Any]] = {}
    for day_entries in by_date.values():
        day_entries.sort(key=lambda entry: (entry.sort_timestamp, entry.session_id))
        for index, entry in enumerate(day_entries, start=1):
            metadata = dict(entry.metadata)
            current_label = str(metadata.get("set_label", "")).strip()
            is_auto_label = bool(metadata.get("set_label_auto", False))
            if not is_auto_label and current_label:
                continue
            metadata["set_label"] = f"Set {index}"
            metadata["set_label_auto"] = True
            save_session_metadata(entry.export_dir, metadata)
            updated[entry.export_dir.resolve()] = metadata
    return updated
