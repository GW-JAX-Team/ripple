"""Unit tests for the LALPulsar ephemeris reader."""

import gzip

import numpy as np
import pytest

from ripplegw.cw.ephemeris import read_ephemeris_file


def _write_synthetic(path, gzipped):
    """Write a tiny synthetic ephemeris file in LAL's 1-line-per-entry format."""
    gps0, dt, n = 630720013, 7200.0, 3
    lines = ["# a comment line", f"{gps0} {dt} {n}"]
    rows = []
    for i in range(n):
        gps = gps0 + i * dt
        # arbitrary but distinct pos/vel/acc values
        vals = [gps] + [float(i + j) for j in range(9)]
        rows.append(vals)
        lines.append(" ".join(repr(v) for v in vals))
    text = "\n".join(lines) + "\n"
    if gzipped:
        with gzip.open(path, "wt") as fh:
            fh.write(text)
    else:
        with open(path, "w") as fh:
            fh.write(text)
    return gps0, dt, n, np.array(rows)


@pytest.mark.parametrize("gzipped", [False, True])
def test_reader_roundtrip(tmp_path, gzipped):
    """The reader recovers header metadata and pos/vel/acc columns."""
    path = str(tmp_path / ("eph.dat.gz" if gzipped else "eph.dat"))
    gps0, dt, n, rows = _write_synthetic(path, gzipped)

    eph = read_ephemeris_file(path)

    assert eph.n_entries == n
    assert eph.gps0 == gps0
    assert eph.dt == dt
    assert eph.gps_end == gps0 + (n - 1) * dt
    np.testing.assert_allclose(eph.pos, rows[:, 1:4])
    np.testing.assert_allclose(eph.vel, rows[:, 4:7])
    np.testing.assert_allclose(eph.acc, rows[:, 7:10])


def test_reader_appends_gz(tmp_path):
    """Requesting the bare name finds the .gz file (LAL behaviour)."""
    path = str(tmp_path / "eph.dat")
    _write_synthetic(path + ".gz", gzipped=True)
    eph = read_ephemeris_file(path)  # note: no .gz suffix
    assert eph.n_entries == 3


def test_reader_missing_file(tmp_path):
    """A missing file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        read_ephemeris_file(str(tmp_path / "does_not_exist.dat"))


def test_reader_inconsistent_count(tmp_path):
    """A header entry-count that disagrees with the data raises ValueError."""
    path = str(tmp_path / "bad.dat")
    with open(path, "w") as fh:
        fh.write("630720013 7200.0 5\n")  # claims 5 entries
        fh.write("630720013 " + " ".join(["0.0"] * 9) + "\n")  # only 1
    with pytest.raises(ValueError):
        read_ephemeris_file(path)
