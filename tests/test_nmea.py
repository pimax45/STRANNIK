from terrain_nav.nmea import (
    RadioAltimeterSample,
    checksum_valid,
    nmea_checksum,
    parse_nmea_line,
    parse_nmea_lines,
)


def _sentence(timestamp: str, altitude: str) -> str:
    body = f"GPGGA,{timestamp},,,,,,,,{altitude},M,,M,,"
    return f"${body}*{nmea_checksum(body):02X}"


def test_parses_gga_altitude_field_and_timestamp():
    line = _sentence("123519.111", "545.4")
    sample = parse_nmea_line(line, verify_checksum=True)

    assert sample == RadioAltimeterSample(12 * 3600 + 35 * 60 + 19.111, 545.4)
    assert checksum_valid(line)


def test_tolerates_bad_checksum_and_empty_fields():
    line = _sentence("000001.000", "123.5")[:-2] + "00"
    assert parse_nmea_line(line) is not None
    assert parse_nmea_line(line, verify_checksum=True) is None
    assert parse_nmea_line("$GPGGA,000002.000,,,,,,,,,M,,,,*00") is None


def test_rollover_and_malformed_lines_are_handled():
    samples = parse_nmea_lines(
        [
            _sentence("235959.900", "100.0"),
            "garbage",
            _sentence("000000.100", "101.0"),
        ]
    )
    assert len(samples) == 2
    assert samples[1].timestamp > samples[0].timestamp

