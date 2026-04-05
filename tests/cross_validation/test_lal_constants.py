"""Tests that ripplegw constants exactly match the live LAL constants.

These tests import lal directly so that if LALSuite updates its constants we
will detect the discrepancy immediately.  The full cross-validation suite
requires LALSuite to be installed; these tests are included in CI via the
explicit path tests/cross_validation/test_lal_constants.py.
"""

import pytest

import ripplegw.constants as const

lal = pytest.importorskip("lal", reason="LALSuite required for constants cross-check")


def test_msun():
    assert const.MSUN == lal.MSUN_SI


def test_mrsun():
    assert const.MRSUN == lal.MRSUN_SI


def test_mtsun():
    assert const.MTSUN == lal.MTSUN_SI


def test_G():
    assert const.G == lal.G_SI


def test_C():
    assert const.C == lal.C_SI


def test_PI():
    assert const.PI == lal.PI


def test_TWO_PI():
    assert const.TWO_PI == lal.TWOPI


def test_MPC():
    # LAL_PC_SI is one parsec in metres; 1 Mpc = 1e6 pc
    assert const.MPC == 1e6 * lal.PC_SI


def test_EULERGAMMA():
    assert const.EULERGAMMA == lal.GAMMA
