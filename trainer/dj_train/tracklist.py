from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


TIMESTAMP = re.compile(
    r"^\s*(?:[-*#]\s*)?(?P<stamp>(?:\d{1,2}:)?\d{1,2}:\d{2})(?:\s*[-–—|:]\s*|\s+)(?P<title>.+?)\s*$"
)


@dataclass(frozen=True)
class TracklistEntry:
    seconds: float
    title: str
    raw: str


def timestamp_seconds(value: str) -> float:
    fields = [int(field) for field in value.split(":")]
    if len(fields) == 2:
        minutes, seconds = fields
        hours = 0
    elif len(fields) == 3:
        hours, minutes, seconds = fields
    else:
        raise ValueError(f"Unsupported timestamp: {value}")
    if seconds >= 60 or (len(fields) == 3 and minutes >= 60):
        raise ValueError(f"Invalid timestamp: {value}")
    return float(hours * 3600 + minutes * 60 + seconds)


def parse_tracklist(text: str) -> list[TracklistEntry]:
    entries: list[TracklistEntry] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = TIMESTAMP.match(stripped)
        if not match:
            continue
        entries.append(
            TracklistEntry(
                seconds=timestamp_seconds(match.group("stamp")),
                title=match.group("title").strip(),
                raw=stripped,
            )
        )
    entries.sort(key=lambda item: item.seconds)
    deduplicated: list[TracklistEntry] = []
    for entry in entries:
        if deduplicated and entry.seconds == deduplicated[-1].seconds:
            deduplicated[-1] = entry
        else:
            deduplicated.append(entry)
    if len(deduplicated) < 2:
        raise ValueError("A timed tracklist needs at least two timestamped tracks")
    return deduplicated


def read_tracklist(path: Path) -> list[TracklistEntry]:
    return parse_tracklist(path.read_text(encoding="utf-8"))
