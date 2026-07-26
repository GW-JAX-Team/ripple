"""Unit tests for the LALPulsar ephemeris reader."""

import gzip
import io
from unittest.mock import patch

import numpy as np
import pytest

from ripplegw.waveforms.cw.ephemeris import read_ephemeris_file, resolve_ephemeris_path


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


class _FakeResponse(io.BytesIO):
    """A minimal stand-in for ``urllib.request.urlopen``'s context-manager result."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False


def test_resolve_existing_path_is_returned_without_network(tmp_path):
    """An existing path is returned as-is; no download is attempted."""
    path = str(tmp_path / "eph.dat")
    _write_synthetic(path, gzipped=False)
    with patch("ripplegw.waveforms.cw.ephemeris.urllib.request.urlopen") as mock_urlopen:
        assert resolve_ephemeris_path(path) == path
    mock_urlopen.assert_not_called()


def test_resolve_path_with_directory_never_downloads(tmp_path):
    """A missing file with a directory component is not treated as a bare
    well-known name -- it must raise, not attempt a download."""
    missing = str(tmp_path / "earth00-40-DE405.dat.gz")
    with (
        patch("ripplegw.waveforms.cw.ephemeris.urllib.request.urlopen") as mock_urlopen,
        pytest.raises(FileNotFoundError),
    ):
        resolve_ephemeris_path(missing)
    mock_urlopen.assert_not_called()


def test_resolve_non_standard_name_never_downloads(tmp_path, monkeypatch):
    """A bare name that isn't a recognised earth*/sun* pattern raises without
    attempting a download -- ripple never guesses at an arbitrary URL."""
    monkeypatch.setenv("RIPPLEGW_CACHE_DIR", str(tmp_path / "cache"))
    with (
        patch("ripplegw.waveforms.cw.ephemeris.urllib.request.urlopen") as mock_urlopen,
        pytest.raises(FileNotFoundError),
    ):
        resolve_ephemeris_path("not-a-real-ephemeris-name.dat")
    mock_urlopen.assert_not_called()


def test_resolve_standard_name_downloads_and_caches(tmp_path, monkeypatch):
    """A recognised standard name not found locally is downloaded once,
    cached under RIPPLEGW_CACHE_DIR, and reused on a second call."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("RIPPLEGW_CACHE_DIR", str(cache_dir))
    gps0, dt, n = 630720013, 7200.0, 2
    lines = [f"{gps0} {dt} {n}"]
    for i in range(n):
        vals = [gps0 + i * dt] + [float(i + j) for j in range(9)]
        lines.append(" ".join(repr(v) for v in vals))
    payload = ("\n".join(lines) + "\n").encode()

    with patch(
        "ripplegw.waveforms.cw.ephemeris.urllib.request.urlopen",
        return_value=_FakeResponse(payload),
    ) as mock_urlopen:
        resolved = resolve_ephemeris_path("earth00-40-DE405.dat")
    mock_urlopen.assert_called_once()
    assert resolved == str(cache_dir / "ripplegw" / "ephemeris" / "earth00-40-DE405.dat")
    eph = read_ephemeris_file(resolved)
    assert eph.n_entries == n

    # Second call reuses the cached file -- no further network access, and no
    # leftover ".part" temp file from the first download.
    with patch(
        "ripplegw.waveforms.cw.ephemeris.urllib.request.urlopen",
        return_value=_FakeResponse(payload),
    ) as mock_urlopen_again:
        resolve_ephemeris_path("earth00-40-DE405.dat")
    mock_urlopen_again.assert_not_called()
    leftovers = list((cache_dir / "ripplegw" / "ephemeris").glob("*.part"))
    assert leftovers == []


def test_resolve_download_failure_raises_file_not_found_and_cleans_up(tmp_path, monkeypatch):
    """A failed download surfaces as FileNotFoundError with no partial file left behind."""
    cache_dir = tmp_path / "cache"
    monkeypatch.setenv("RIPPLEGW_CACHE_DIR", str(cache_dir))
    with patch(
        "ripplegw.waveforms.cw.ephemeris.urllib.request.urlopen",
        side_effect=OSError("network unreachable"),
    ), pytest.raises(FileNotFoundError):
        resolve_ephemeris_path("sun00-40-DE405.dat.gz")
    ephemeris_dir = cache_dir / "ripplegw" / "ephemeris"
    leftovers = list(ephemeris_dir.glob("*")) if ephemeris_dir.exists() else []
    assert leftovers == []
