"""Recoverable NSBH domain checks, anchored to LAL's model boundaries."""

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import checkify

import ripplegw
from ripplegw.conversions import ms_to_Mc_eta
from tests.helpers.params import canonical_params

nsbh = importlib.import_module("ripplegw.waveforms.cbc.IMRPhenom_NRTidal.IMRPhenomNSBH")
F = jnp.array([20.0, 40.0, 80.0])


def theta(m1=7.0, m2=1.4, lam=400.0, chi=0.3):
    mc, eta = ms_to_Mc_eta(jnp.array([m1, m2]))
    return jnp.array([mc, eta, chi, 0.0, lam, 200.0, 0.0, 0.0])


evaluate = jax.jit(lambda p: nsbh.gen_IMRPhenomNSBH(F, p, 20.0))


@pytest.mark.parametrize("lam", [-1.0, 5000.0001])
def test_invalid_tides_return_nan_without_raising(lam):
    assert np.isnan(evaluate(theta(lam=lam))).all()
    assert np.isfinite(evaluate(theta())).all()


def test_mass_ratio_above_100_is_rejected():
    assert np.isnan(evaluate(theta(m1=141.4))).all()


def test_mass_ratio_endpoint_in_chirp_coordinates():
    boundary = 100.0 / 101.0**2
    p = theta().at[1].set(boundary)
    assert np.isfinite(evaluate(p)).all()
    q = 100.01
    assert np.isnan(evaluate(p.at[1].set(q / (1.0 + q) ** 2))).all()


@pytest.mark.parametrize("m2", [1.3, 1.4, 2.0])
def test_mass_ratio_100_from_component_masses_is_accepted(m2):
    # LAL compares the masses (mBH <= 100 mNS); converting q = 100 exactly to
    # eta can round one ulp below 100/101**2, which must not reject the sample.
    assert np.isfinite(evaluate(theta(m1=100.0 * m2, m2=m2))).all()


def test_mass_ratio_100_rounding_reaches_below_boundary():
    # Keeps the test above discriminating: some exact q = 100 pairs round low.
    etas = [float(ms_to_Mc_eta(jnp.array([100.0 * m, m]))[1]) for m in (1.3, 1.4)]
    assert min(etas) < 100.0 / 101.0**2


@pytest.mark.parametrize("eta", [0.0, -0.1, 0.2501, np.nan])
def test_invalid_eta_is_not_hidden_by_mass_conversion(eta):
    assert np.isnan(evaluate(theta().at[1].set(eta))).all()


@pytest.mark.parametrize("m2", [0.7, 4.2])
def test_no_neutron_star_mass_cut(m2):
    assert np.isfinite(evaluate(theta(m1=5 * m2, m2=m2))).all()


def test_redshift_scaling_is_preserved_above_mass_cut():
    # Fixed dimensionless parameters: h(f/k, k*m)/h(f, m) = k**2 at fixed distance.
    scale = 3.0
    original = nsbh.gen_IMRPhenomNSBH(F, theta(), 20.0)
    redshifted = nsbh.gen_IMRPhenomNSBH(
        F / scale, theta(m1=7.0 * scale, m2=1.4 * scale), 20.0 / scale
    )
    np.testing.assert_allclose(redshifted, scale**2 * original, atol=0.0, rtol=1e-9)


@pytest.mark.parametrize("lam", [0.0, 5000.0])
def test_tidal_endpoints_are_inclusive(lam):
    assert np.isfinite(evaluate(theta(lam=lam))).all()


def test_checkify_returns_error_without_aborting():
    checked = jax.jit(
        checkify.checkify(
            lambda p: nsbh.gen_IMRPhenomNSBH(F, p, 20.0, diagnostics=True)
        )
    )
    err, output = checked(theta(lam=5001.0))
    assert err.get() is not None and "lambda_2" in err.get()
    assert np.isnan(output).all()
    err, output = checked(theta())
    assert err.get() is None
    assert np.isfinite(output).all()


def test_mixed_batch_rejects_only_invalid_samples_inside_scan():
    batch = jnp.stack([theta(), theta(lam=5001.0), theta(m2=4.2, m1=21.0)])

    def loglike(p):
        h = nsbh.gen_IMRPhenomNSBH(F, p, 20.0)
        valid = jnp.all(jnp.isfinite(h))
        safe_h = jnp.where(valid, h, 0.0)
        return jnp.where(valid, -jnp.sum(jnp.abs(safe_h * 1e23) ** 2), -jnp.inf)

    @jax.jit
    def run(samples):
        return jax.lax.scan(lambda c, p: (c, loglike(p)), None, samples)[1]

    output = run(batch)
    assert np.isfinite(np.asarray(output)[[0, 2]]).all()
    assert np.isneginf(output[1])
    mapped = jax.jit(jax.vmap(loglike))(batch)
    np.testing.assert_allclose(mapped, output, atol=0.0, rtol=1e-12)


def test_negative_derivative_only_rejected_below_matching_frequency():
    cc = nsbh.phenomC_amp_coeffs(0.14, 0.3)
    props = nsbh.nsbh_merger_properties(7.0, 1.4, 0.3, 400.0)
    # Force xdot negative on both sides of a known matching frequency.
    cc = {**cc, "xdota2": -1e6, "Mf1": 0.01}
    amplitude = jax.jit(nsbh.nsbh_amplitude)
    assert np.isnan(amplitude(jnp.array([0.005, 0.02]), cc, props)).all()
    assert np.isfinite(amplitude(jnp.array([0.01, 0.02]), cc, props)).all()


@pytest.mark.parametrize("method", ["__call__", "amplitude", "phase", "strain"])
def test_all_public_interfaces_reject_invalid_sample(method):
    wf = ripplegw.waveform("IMRPhenomNSBH")
    params = canonical_params(wf, lambda_2=5001.0)
    output = jax.jit(getattr(wf, method))(F, params)
    for array in jax.tree.leaves(output):
        assert np.isnan(array).all()


def test_valid_gradient_survives_checkify():
    def power(p):
        return jnp.sum(
            jnp.abs(nsbh.gen_IMRPhenomNSBH(F, p, 20.0, diagnostics=True) * 1e23) ** 2
        )

    checked = jax.jit(checkify.checkify(jax.value_and_grad(power)))
    err, (value, gradient) = checked(theta())
    assert err.get() is None
    assert np.isfinite(value)
    assert np.isfinite(gradient).all()


def test_lal_parameter_rejection_and_valid_amplitude():
    lal = pytest.importorskip("lal")
    lalsim = pytest.importorskip("lalsimulation")
    freq = lal.CreateREAL8Sequence(len(F))
    freq.data[:] = np.asarray(F)

    def reference(m1, lam):
        extra = lal.CreateDict()
        lalsim.SimInspiralWaveformParamsInsertTidalLambda2(extra, lam)
        return lalsim.SimIMRPhenomNSBHFrequencySequence(
            freq,
            0.0,
            20.0,
            200.0e6 * lal.PC_SI,
            m1 * lal.MSUN_SI,
            1.4 * lal.MSUN_SI,
            0.3,
            0.0,
            extra,
        ).data.data

    for m1, lam in [(141.4, 400.0), (7.0, -1.0), (7.0, 5001.0)]:
        with pytest.raises(RuntimeError):
            reference(m1, lam)
        assert np.isnan(evaluate(theta(m1=m1, lam=lam))).all()
    expected = reference(7.0, 400.0)
    actual = nsbh.gen_IMRPhenomNSBH(F, theta(), 20.0, f_max=float(F[-1]))
    np.testing.assert_allclose(np.abs(actual), np.abs(expected), atol=0.0, rtol=1e-7)


def test_default_has_no_error_effects():
    assert not jax.make_jaxpr(evaluate)(theta()).effects
    wf = ripplegw.waveform("IMRPhenomNSBH", diagnostics=True)
    checked = jax.jit(checkify.checkify(wf))
    err, output = checked(F, canonical_params(wf, lambda_2=-1.0))
    assert err.get() is not None and "lambda_2" in err.get()
    assert np.isnan(output["p"]).all()


def test_checkified_vmap_preserves_good_samples():
    checked = jax.jit(
        jax.vmap(
            checkify.checkify(
                lambda p: nsbh.gen_IMRPhenomNSBH(F, p, 20.0, diagnostics=True)
            )
        )
    )
    errors, output = checked(jnp.stack([theta(), theta(lam=-1.0), theta()]))
    assert "mapped index 1" in errors.get()
    assert np.isnan(output[1]).all()
    assert np.isfinite(np.asarray(output)[[0, 2]]).all()


def test_empty_frequency_array_is_rejected_at_trace_time():
    with pytest.raises(ValueError, match="nonempty frequency"):
        jax.jit(nsbh.gen_IMRPhenomNSBH)(jnp.array([]), theta(), 20.0)


@pytest.mark.parametrize("case", ["eta", "tides", "spin", "derivative"])
@pytest.mark.parametrize("diagnostics", [False, True])
@pytest.mark.parametrize("layout", ["scalar", "scan"])
def test_invalid_parameters_skip_merger_and_phase(
    monkeypatch, case, diagnostics, layout
):
    """Runtime callbacks distinguish a skipped branch from a NaN mask."""
    calls = []
    merger = nsbh.nsbh_merger_properties
    phase = nsbh.get_coeffs

    def observed_merger(*args, **kwargs):
        jax.debug.callback(lambda: calls.append("merger"))
        return merger(*args, **kwargs)

    def observed_phase(*args, **kwargs):
        jax.debug.callback(lambda: calls.append("phase"))
        return phase(*args, **kwargs)

    monkeypatch.setattr(nsbh, "nsbh_merger_properties", observed_merger)
    monkeypatch.setattr(nsbh, "get_coeffs", observed_phase)
    coeffs = nsbh._inspiral_coeffs
    if case == "derivative":

        def negative_derivative(*args, **kwargs):
            return {**coeffs(*args, **kwargs), "xdota2": -1e6, "Mf1": 0.01}

        monkeypatch.setattr(nsbh, "_inspiral_coeffs", negative_derivative)
    p = theta()
    if case == "eta":
        p = p.at[1].set(0.3)
    elif case == "tides":
        p = p.at[4].set(-1.0)
    elif case == "spin":
        p = p.at[2].set(4.0)
    fn = lambda p: nsbh.gen_IMRPhenomNSBH(F, p, 20.0, diagnostics=diagnostics)
    if diagnostics:
        fn = checkify.checkify(fn)
    if layout == "scan":
        scalar = fn
        fn = lambda batch: jax.lax.scan(lambda c, x: (c, scalar(x)), None, batch)[1]
        p = jnp.stack([p, p])
    result = jax.jit(fn)(p)
    jax.block_until_ready(result)
    jax.effects_barrier()
    output = result[1] if diagnostics else result
    assert np.isnan(output).all()
    if diagnostics:
        assert result[0].get() is not None
    assert calls == [], f"Invalid {case} still executed {calls}"


@pytest.mark.parametrize("case", ["eta", "tides", "spin", "valid"])
def test_domain_stages_execute_in_order(monkeypatch, case):
    calls = []
    for name in (
        "Mc_eta_to_ms",
        "_remnant_spin",
        "_inspiral_coeffs",
        "_inspiral_quantities",
        "_amplitude_coeffs",
        "nsbh_merger_properties",
        "get_coeffs",
    ):
        original = getattr(nsbh, name)

        def observed(*args, _original=original, _name=name, **kwargs):
            jax.debug.callback(lambda: calls.append(_name), ordered=True)
            return _original(*args, **kwargs)

        monkeypatch.setattr(nsbh, name, observed)
    p = theta()
    if case == "eta":
        p = p.at[1].set(0.3)
    elif case == "tides":
        p = p.at[4].set(-1.0)
    elif case == "spin":
        p = p.at[2].set(4.0)
    output = jax.jit(lambda p: nsbh.gen_IMRPhenomNSBH(F, p, 20.0))(p)
    jax.block_until_ready(output)
    jax.effects_barrier()
    stages = [
        "Mc_eta_to_ms",
        "_remnant_spin",
        "_inspiral_coeffs",
        "_inspiral_quantities",
        "_amplitude_coeffs",
        "nsbh_merger_properties",
        "get_coeffs",
    ]
    assert calls == (
        stages if case == "valid" else stages[:2] if case == "spin" else []
    )


@pytest.mark.parametrize("dtype", [jnp.float32, jnp.float64])
@pytest.mark.parametrize("parameter_dtype", [jnp.float32, jnp.float64])
def test_mixed_frequency_and_parameter_dtypes(dtype, parameter_dtype):
    fn = jax.jit(lambda p: nsbh.gen_IMRPhenomNSBH(F.astype(dtype), p, 20.0))
    assert np.isfinite(fn(theta().astype(parameter_dtype))).all()
    assert np.isnan(fn(theta(lam=-1.0).astype(parameter_dtype))).all()


@pytest.mark.parametrize("method", ["__call__", "amplitude", "phase", "strain"])
def test_public_output_construction_is_skipped_on_rejection(monkeypatch, method):
    calls = []
    original = nsbh._format_output

    def observed(*args):
        jax.debug.callback(lambda: calls.append("output"))
        return original(*args)

    monkeypatch.setattr(nsbh, "_format_output", observed)
    wf = ripplegw.waveform("IMRPhenomNSBH")
    fn = jax.jit(getattr(wf, method))
    bad = fn(F, canonical_params(wf, lambda_2=-1.0))
    jax.block_until_ready(bad)
    jax.effects_barrier()
    assert calls == []
    assert all(np.isnan(x).all() for x in jax.tree.leaves(bad))
    good = fn(F, canonical_params(wf))
    jax.block_until_ready(good)
    jax.effects_barrier()
    assert calls == ["output"]
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(good))


@pytest.mark.parametrize("method", ["__call__", "amplitude", "phase", "strain"])
@pytest.mark.parametrize("field", ["phase_c", "d_L", "lambda_2"])
def test_individual_parameter_field_can_promote_independently(method, field):
    wf = ripplegw.waveform("IMRPhenomNSBH")
    params = {
        k: jnp.asarray(v, dtype=jnp.float32) for k, v in canonical_params(wf).items()
    }
    params[field] = params[field].astype(jnp.float64)
    result = jax.jit(getattr(wf, method))(F.astype(jnp.float32), params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(result))


def test_reference_frequency_can_promote_independently():
    result = jax.jit(nsbh.gen_IMRPhenomNSBH)(
        F.astype(jnp.float32), theta().astype(jnp.float32), jnp.float64(20.0)
    )
    assert np.isfinite(result).all()


def test_shared_strain_matches_separate_amplitude_and_phase():
    wf = ripplegw.waveform("IMRPhenomNSBH")
    params = canonical_params(wf)
    amplitude = jax.jit(wf.amplitude)(F, params)
    phase = jax.jit(wf.phase)(F, params)
    actual = jax.jit(wf.strain)(F, params)
    expected = amplitude * jnp.cos(phase) + 1j * amplitude * jnp.sin(phase)
    np.testing.assert_allclose(actual, expected, atol=0.0, rtol=1e-10)


@pytest.mark.parametrize("method", ["__call__", "amplitude", "phase", "strain"])
@pytest.mark.parametrize("strong_fields", [False, True])
def test_python_chirp_coordinates_preserve_array_conversion_dtype(
    method, strong_fields
):
    wf = ripplegw.waveform("IMRPhenomNSBH")
    params = {k: float(v) for k, v in canonical_params(wf).items()}
    if strong_fields:
        params = {
            k: v if k in ("M_c", "eta") else jnp.float32(v) for k, v in params.items()
        }
    result = jax.jit(getattr(wf, method))(F.astype(jnp.float32), params)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(result))
