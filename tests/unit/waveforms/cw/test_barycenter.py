"""Unit tests for CW barycentering geometry."""

import numpy as np

from ripplegw.waveforms.cw.barycenter import C, _detector_geocentric, source_unit_vector


def test_source_unit_vector():
    """The source unit vector matches the analytic expression and is a unit."""
    alpha, delta = 1.1, -0.4
    n = np.asarray(source_unit_vector(alpha, delta))
    expected = np.array(
        [np.cos(delta) * np.cos(alpha), np.cos(delta) * np.sin(alpha), np.sin(delta)]
    )
    np.testing.assert_allclose(n, expected, rtol=0, atol=1e-14)
    np.testing.assert_allclose(np.linalg.norm(n), 1.0, atol=1e-14)


def test_detector_geocentric_at_the_geocentre_is_finite_and_has_no_site_offset():
    """The geocentre is a supported site, and its site contribution vanishes exactly.

    ``lz / rd`` is 0/0 there, and an unguarded ``arccos`` makes every downstream sample NaN.
    Asserting only finiteness would accept a wrong-but-finite site offset, so this pins the
    physical value too: at the geocentre the detector *is* the Earth centre, so ``rd`` must be
    exactly zero and the rotation term it scales must vanish.
    """
    rd, _longitude, latitude, sin_lat, cos_lat = _detector_geocentric((0.0, 0.0, 0.0))

    for name, value in (
        ("rd", rd),
        ("latitude", latitude),
        ("sin_lat", sin_lat),
        ("cos_lat", cos_lat),
    ):
        assert np.isfinite(np.asarray(value)).all(), (
            f"{name} is not finite at the geocentre"
        )
    assert float(rd) == 0.0, "the geocentre must sit at zero geocentric radius"


def test_detector_geocentric_is_unchanged_for_a_real_site():
    """The geocentre guard must be inert everywhere else.

    ``jnp.where(rd > 0, ...)`` engages only at exactly zero, so a real detector must reproduce
    the plain spherical conversion bit for bit -- otherwise the fix has moved existing signals.
    """
    location = (-2161414.93, -3834695.35, 4600350.23)  # LIGO Hanford, metres
    rd, longitude, latitude, sin_lat, cos_lat = _detector_geocentric(location)

    lx, ly, lz = (np.asarray(c) / C for c in location)
    expected_rd = np.sqrt(lx * lx + ly * ly + lz * lz)
    expected_lat = np.pi / 2.0 - np.arccos(lz / expected_rd)

    assert float(rd) == float(expected_rd)
    assert float(longitude) == float(np.arctan2(ly, lx))
    assert float(latitude) == float(expected_lat)
    assert float(sin_lat) == float(np.sin(expected_lat))
    assert float(cos_lat) == float(np.cos(expected_lat))


def test_differentiating_at_the_geocentre_is_not_supported():
    """Pins the documented limitation, so that lifting it is a deliberate act.

    Spherical coordinates are singular at the origin: ``longitude = arctan2(0, 0)`` has no
    derivative, so the gradient is NaN there however the radius and latitude are guarded. The
    forward value is unaffected. If a future change makes this finite -- for instance by moving
    the rotation into Cartesian coordinates -- this test fails and the docstring's note needs
    removing along with it.
    """
    import jax
    import jax.numpy as jnp

    def total(location):
        rd, longitude, latitude, _sin_lat, _cos_lat = _detector_geocentric(
            (location[0], location[1], location[2])
        )
        return rd + longitude + latitude

    gradient = jax.grad(total)(jnp.zeros(3))

    assert not np.isfinite(np.asarray(gradient)).all(), (
        "the geocentre gradient is finite; if that is intended, drop the forward-only note "
        "from _detector_geocentric's docstring"
    )
    assert np.isfinite(
        np.asarray(jax.grad(total)(jnp.array([-2161414.93, -3834695.35, 4600350.23])))
    ).all()
