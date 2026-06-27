"""Tolerant NMEA-0183 radio-altimeter parser."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True, slots=True)
class RadioAltimeterSample:
    timestamp: float
    radio_altitude_m: float


def nmea_checksum(sentence_body: str) -> int:
    value = 0
    for character in sentence_body:
        value ^= ord(character)
    return value


def checksum_valid(line: str) -> bool:
    stripped = line.strip()
    if not stripped.startswith("$") or "*" not in stripped:
        return False
    body, supplied = stripped[1:].rsplit("*", 1)
    try:
        return nmea_checksum(body) == int(supplied[:2], 16)
    except ValueError:
        return False


def parse_nmea_timestamp(value: str) -> float:
    """Convert hhmmss.sss to seconds since start of day."""

    raw = value.strip()
    if len(raw) < 6:
        raise ValueError("NMEA timestamp is missing or too short")
    hours = int(raw[:2])
    minutes = int(raw[2:4])
    seconds = float(raw[4:])
    if hours > 23 or minutes > 59 or not 0 <= seconds < 60:
        raise ValueError("Invalid NMEA timestamp")
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_nmea_line(
    line: str,
    *,
    verify_checksum: bool = False,
) -> RadioAltimeterSample | None:
    """Parse GGA field 9 as radio altitude, tolerating malformed records."""

    stripped = line.strip()
    if not stripped.startswith("$"):
        return None
    if verify_checksum and not checksum_valid(stripped):
        return None
    payload = stripped.split("*", 1)[0]
    fields = payload.split(",")
    if len(fields) <= 9 or not fields[0].endswith("GGA"):
        return None
    try:
        timestamp = parse_nmea_timestamp(fields[1])
        altitude = float(fields[9])
    except (ValueError, TypeError):
        return None
    if altitude < 0:
        return None
    return RadioAltimeterSample(timestamp, altitude)


def parse_nmea_lines(
    lines: Iterable[str],
    *,
    verify_checksum: bool = False,
) -> list[RadioAltimeterSample]:
    samples: list[RadioAltimeterSample] = []
    day_offset = 0.0
    previous_time: float | None = None
    for line in lines:
        sample = parse_nmea_line(line, verify_checksum=verify_checksum)
        if sample is None:
            continue
        current = sample.timestamp
        if previous_time is not None and current + day_offset < previous_time - 12 * 3600:
            day_offset += 24 * 3600
        absolute = current + day_offset
        if previous_time is not None and absolute <= previous_time:
            continue
        samples.append(RadioAltimeterSample(absolute, sample.radio_altitude_m))
        previous_time = absolute
    return samples


def read_nmea(
    path: str | Path,
    *,
    verify_checksum: bool = False,
) -> list[RadioAltimeterSample]:
    with Path(path).open("r", encoding="ascii", errors="ignore") as stream:
        return parse_nmea_lines(stream, verify_checksum=verify_checksum)

