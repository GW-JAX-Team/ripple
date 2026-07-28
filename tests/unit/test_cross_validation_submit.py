"""Focused, scheduler-free tests for the unified cross-validation launcher."""

from pathlib import Path

import pytest

from tests.cross_validation import submit


def test_known_waveforms_resolve_to_their_native_tests():
    fd = submit.test_target("IMRPhenomD")
    sine = submit.test_target("SineGaussian")
    cw = submit.test_target("PulsarSignal")
    exact = submit.test_target("ExactPulsarSignal")

    assert fd.adapter == "lal-frequency-domain"
    assert len(fd.pytest_nodeids) == 2  # overlap plus absolute-phase check
    assert sine.adapter == "sinegaussian-lalsimulation"
    assert sine.resources.gpus == 0
    assert cw.adapter == "cw-makefakedata"
    assert cw.pytest_args == ("--cw-waveform", "PulsarSignal")
    assert cw.requires_ephemerides
    assert exact.adapter == "exact-pulsar-lal"
    assert exact.pytest_args == ()
    assert exact.requires_ephemerides


def test_precessing_fd_target_avoids_nonexistent_phase_node():
    target = submit.test_target("IMRPhenomXP")

    assert target.adapter == "lal-frequency-domain"
    assert len(target.pytest_nodeids) == 1
    assert "test_reference_overlap" in target.pytest_nodeids[0]


def test_pytest_command_keeps_target_specific_arguments(tmp_path):
    target = submit.TestTarget(
        waveform="PulsarSignal",
        adapter="cw-makefakedata",
        pytest_nodeids=("test.py::test_large_scale",),
        pytest_args=("--cw-waveform", "PulsarSignal"),
        resources=submit.CW_RESOURCES,
        requires_ephemerides=True,
    )

    command = submit.build_pytest_command(
        target,
        python="/venv/bin/python",
        n_samples=25,
        outdir=tmp_path / "PulsarSignal",
        plots=True,
    )

    assert command[:4] == (
        "/venv/bin/python",
        "-m",
        "pytest",
        "test.py::test_large_scale",
    )
    assert "--cw-waveform" in command
    assert "PulsarSignal" in command
    assert "--plots" in command
    assert "--cache-reference" not in command


def test_slurm_command_carries_cw_ephemerides_and_cpu_resources(tmp_path):
    target = submit.TestTarget(
        waveform="PulsarSignal",
        adapter="cw-makefakedata",
        pytest_nodeids=("test.py::test_large_scale",),
        pytest_args=(),
        resources=submit.CW_RESOURCES,
        requires_ephemerides=True,
    )
    command = submit.build_slurm_command(
        target,
        command=("/venv/bin/python", "-m", "pytest", "test.py::test_large_scale"),
        outdir=tmp_path,
        repo_dir=Path("/repo"),
        resources=submit.CW_RESOURCES,
        ephemerides=("/data/earth.dat", "/data/sun.dat"),
    )

    assert "--cpus-per-task=64" in command
    assert not any(arg.startswith("--gpus-per-task") for arg in command)
    assert any(
        arg
        == "--export=ALL,RIPPLE_EARTH_EPHEMERIS=/data/earth.dat,RIPPLE_SUN_EPHEMERIS=/data/sun.dat"
        for arg in command
    )


def test_condor_description_uses_getenv_and_cpu_only_td_resources(tmp_path):
    target = submit.TestTarget(
        waveform="SineGaussian",
        adapter="sinegaussian-lalsimulation",
        pytest_nodeids=("test.py::test_large_scale",),
        pytest_args=(),
        resources=submit.TD_RESOURCES,
    )
    text = submit.build_condor_submit(
        target,
        command=("/venv/bin/python", "-m", "pytest", "test.py::test_large_scale"),
        outdir=tmp_path / "results with space",
        repo_dir=Path("/repo"),
        resources=submit.TD_RESOURCES,
    )

    assert "getenv = True" in text
    assert "request_CPUs = 16" in text
    assert "request_memory = 16GB" in text
    assert "request_GPUs" not in text


def test_cw_ephemerides_require_both_real_files(tmp_path):
    earth = tmp_path / "earth.dat"
    sun = tmp_path / "sun.dat"
    earth.touch()
    sun.touch()

    assert submit.cw_ephemerides(
        {
            "RIPPLE_EARTH_EPHEMERIS": str(earth),
            "RIPPLE_SUN_EPHEMERIS": str(sun),
        },
        validate_files=True,
    ) == (str(earth), str(sun))
    with pytest.raises(ValueError, match="RIPPLE_EARTH_EPHEMERIS"):
        submit.cw_ephemerides({}, validate_files=True)


def test_dry_run_uses_absolute_output_and_never_calls_scheduler(tmp_path, capsys):
    def no_scheduler(*_args, **_kwargs):
        raise AssertionError("dry run must not invoke a scheduler")

    outdir = tmp_path / "relative-output"
    status = submit.main(
        [
            "--scheduler",
            "condor",
            "--waveform",
            "SineGaussian",
            "--n-samples",
            "2",
            "--outdir",
            str(outdir),
            "--dry-run",
        ],
        run=no_scheduler,
    )

    rendered = capsys.readouterr().out
    assert status == 0
    assert str(outdir.resolve() / "SineGaussian" / "test.sub") in rendered
    assert not outdir.exists()
