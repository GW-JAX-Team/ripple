"""Tests for ripplegw.conversions: mass and tidal parameter transforms."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

from ripplegw.conversions import (
    Mc_eta_to_ms,
    lambda_tildes_to_lambdas,
    lambda_tildes_to_lambdas_from_q,
    lambdas_to_lambda_tildes,
    lambdas_to_lambda_tildes_from_q,
    ms_to_Mc_eta,
)

# --- mass conversions --------------------------------------------------------


def test_ms_to_Mc_eta_known_value():
    # Equal-mass 30+30: Mc = 30 * 2**(-1/5), eta = 0.25.
    Mc, eta = ms_to_Mc_eta(jnp.array([30.0, 30.0]))
    assert Mc == pytest.approx(30.0 * 2 ** (-1 / 5))
    assert eta == pytest.approx(0.25)


def test_ms_to_Mc_eta_symmetric_in_masses():
    a = ms_to_Mc_eta(jnp.array([30.0, 25.0]))
    b = ms_to_Mc_eta(jnp.array([25.0, 30.0]))
    assert a[0] == pytest.approx(b[0])
    assert a[1] == pytest.approx(b[1])


@pytest.mark.parametrize(
    "m1,m2", [(30.0, 25.0), (1.4, 1.3), (100.0, 5.0), (30.0, 30.0)]
)
def test_mass_roundtrip(m1, m2):
    Mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    m1_back, m2_back = Mc_eta_to_ms(jnp.array([Mc, eta]))
    # Mc_eta_to_ms returns m1 >= m2 by convention; compare as sets.
    assert sorted([float(m1_back), float(m2_back)]) == pytest.approx(sorted([m1, m2]))


def test_Mc_eta_to_ms_orders_m1_geq_m2():
    m1, m2 = Mc_eta_to_ms(jnp.array([30.0 * 2 ** (-1 / 5), 0.2]))
    assert m1 >= m2


# --- tidal conversions --------------------------------------------------------


@pytest.mark.parametrize(
    "l1,l2,m1,m2",
    [(500.0, 400.0, 1.4, 1.3), (0.0, 0.0, 1.4, 1.4), (1000.0, 0.0, 1.6, 1.2)],
)
def test_lambda_roundtrip(l1, l2, m1, m2):
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([l1, l2, m1, m2]))
    l1_back, l2_back = lambda_tildes_to_lambdas(jnp.array([lt, dlt, m1, m2]))
    assert float(l1_back) == pytest.approx(l1, abs=1e-8)
    assert float(l2_back) == pytest.approx(l2, abs=1e-8)


def test_lambda_zero_deformability_gives_zero_tildes():
    lt, dlt = lambdas_to_lambda_tildes(jnp.array([0.0, 0.0, 1.4, 1.3]))
    assert float(lt) == pytest.approx(0.0)
    assert float(dlt) == pytest.approx(0.0)


def test_lambdas_to_lambda_tildes_from_q_matches_mass_form():
    m1, m2 = 1.5, 1.2
    q = m2 / m1
    lt_q, dlt_q = lambdas_to_lambda_tildes_from_q(jnp.array([500.0, 400.0, q]))
    lt_m, dlt_m = lambdas_to_lambda_tildes(jnp.array([500.0, 400.0, m1, m2]))
    assert float(lt_q) == pytest.approx(float(lt_m))
    assert float(dlt_q) == pytest.approx(float(dlt_m))


def test_lambda_tildes_to_lambdas_from_q_matches_mass_form():
    m1, m2 = 1.5, 1.2
    q = m2 / m1
    l1_q, l2_q = lambda_tildes_to_lambdas_from_q(jnp.array([300.0, 20.0, q]))
    l1_m, l2_m = lambda_tildes_to_lambdas(jnp.array([300.0, 20.0, m1, m2]))
    assert float(l1_q) == pytest.approx(float(l1_m))
    assert float(l2_q) == pytest.approx(float(l2_m))
