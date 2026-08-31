"""Run a selected large-scale cross-validation test locally or submit it to a scheduler.

The suite has deliberately different reference methods for different waveform
families: CBC models use frequency-domain LAL overlaps, while time-domain models use
aligned, direct sampled-time comparisons.  This module gives them one launcher without
forcing them through the same metric or resource request.

For example::

    python -m tests.cross_validation.submit --scheduler slurm \\
        --waveform IMRPhenomD --n-samples 1000 --plots

``--waveform all`` submits one independently configured job for every waveform with a
registered large-scale test adapter.  Adding another family means adding one
``TestAdapter`` below; the local, Slurm, and HTCondor code stays unchanged.
"""

from __future__ import annotations

import argparse
import importlib
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_REPO_DIR = Path(__file__).resolve().parents[2]

_FD_OVERLAP = "tests/cross_validation/fd/test_overlap.py::test_reference_overlap"
_FD_PHASE = (
    "tests/cross_validation/fd/test_phase_convention.py::"
    "test_reference_phase_convention"
)
_CW_TEST_NODEID = (
    "tests/cross_validation/cw/test_makefakedata_v5_large_scale.py::"
    "test_makefakedata_v5_large_scale"
)
_EXACT_PULSAR_TEST_NODEID = (
    "tests/cross_validation/cw/test_exact_pulsar_signal_large_scale.py::"
    "test_exact_pulsar_signal_lal_large_scale"
)
_SINEGAUSSIAN_TEST_NODEID = (
    "tests/cross_validation/td/test_sinegaussian.py::"
    "test_sinegaussian_lalsimulation_large_scale"
)


@dataclass(frozen=True)
class Resources:
    """Resource defaults for one large-scale test method.

    ``memory`` remains a scheduler-facing string so site conventions such as ``16G``
    or ``16000M`` can be supplied unchanged.  The common ``16G`` spelling is converted
    to HTCondor's ``16GB`` form when its submit description is written.
    """

    partition: str | None
    cpus: int
    gpus: int
    memory: str
    walltime: str


FD_RESOURCES = Resources("gpu_h100", 4, 1, "16G", "04:00:00")
"""Default LAL FD resources; override them for a different cluster."""

TD_RESOURCES = Resources("genoa", 16, 0, "16G", "04:00:00")
"""Default generic time-domain resources; these tests are CPU-only."""

CW_RESOURCES = Resources("genoa", 64, 0, "32G", "04:00:00")
"""Default CW resources; LALPulsar generation benefits from many CPU cores."""


@dataclass(frozen=True)
class TestTarget:
    """A waveform and the exact pytest test that validates it."""

    waveform: str
    adapter: str
    pytest_nodeids: tuple[str, ...]
    pytest_args: tuple[str, ...]
    resources: Resources
    requires_ephemerides: bool = False


Predicate = Callable[[str, Mapping[str, Any]], bool]
NodeidsBuilder = Callable[[str, Mapping[str, Any]], tuple[str, ...]]
PytestArgsBuilder = Callable[[str, Mapping[str, Any]], tuple[str, ...]]


@dataclass(frozen=True)
class TestAdapter:
    """A registry entry joining a waveform family to one large-scale test.

    A future adapter supplies only its matching predicate, pytest node IDs, optional
    pytest arguments, and resources.  Scheduler code does not need a new branch.
    """

    name: str
    matches: Predicate
    pytest_nodeids: NodeidsBuilder
    pytest_args: PytestArgsBuilder
    resources: Resources
    requires_ephemerides: bool = False

    def build_test_target(
        self, waveform: str, metadata: Mapping[str, Any]
    ) -> TestTarget:
        """Build the selected target for one matching registered waveform."""

        return TestTarget(
            waveform=waveform,
            adapter=self.name,
            pytest_nodeids=self.pytest_nodeids(waveform, metadata),
            pytest_args=self.pytest_args(waveform, metadata),
            resources=self.resources,
            requires_ephemerides=self.requires_ephemerides,
        )


def _ripplegw_for_discovery():
    """Import the registry without probing CUDA on a CPU-only login node.

    The launcher itself never evaluates a waveform.  Limiting only this short-lived
    process to CPU avoids a noisy CUDA-plugin warning during ``--help``/``--dry-run``;
    the environment is restored before a local command is run or a batch job is
    submitted, so the selected FD job still uses its requested GPU normally.
    """

    if "ripplegw" in sys.modules:
        return importlib.import_module("ripplegw")
    previous = os.environ.get("JAX_PLATFORMS")
    if previous is None:
        os.environ["JAX_PLATFORMS"] = "cpu"
    try:
        return importlib.import_module("ripplegw")
    finally:
        if previous is None:
            os.environ.pop("JAX_PLATFORMS", None)


def _lal_supports(waveform: str) -> bool:
    """Ask LAL's declarative adapter map, without requiring LAL to be installed."""

    _ripplegw_for_discovery()
    from tests.cross_validation.reference import get_backend

    return get_backend("lal").supports(waveform)


def _fd_nodeids(waveform: str, metadata: Mapping[str, Any]) -> tuple[str, ...]:
    """Every FD target gets overlap; only non-precessing models get phase checks."""

    nodeids = [f"{_FD_OVERLAP}[{waveform}]"]
    if metadata.get("is_precessing") is False:
        nodeids.append(f"{_FD_PHASE}[{waveform}]")
    return tuple(nodeids)


def _no_pytest_args(_waveform: str, _metadata: Mapping[str, Any]) -> tuple[str, ...]:
    return ()


def _cw_pytest_args(waveform: str, _metadata: Mapping[str, Any]) -> tuple[str, ...]:
    return ("--cw-waveform", waveform)


TEST_ADAPTERS: tuple[TestAdapter, ...] = (
    TestAdapter(
        name="lal-frequency-domain",
        matches=lambda waveform, metadata: (
            metadata.get("domain") == "FD" and _lal_supports(waveform)
        ),
        pytest_nodeids=_fd_nodeids,
        pytest_args=_no_pytest_args,
        resources=FD_RESOURCES,
    ),
    TestAdapter(
        name="sinegaussian-lalsimulation",
        matches=lambda waveform, _metadata: waveform == "SineGaussian",
        pytest_nodeids=lambda _waveform, _metadata: (_SINEGAUSSIAN_TEST_NODEID,),
        pytest_args=_no_pytest_args,
        resources=TD_RESOURCES,
    ),
    TestAdapter(
        name="cw-makefakedata",
        matches=lambda waveform, _metadata: (
            waveform in {"PulsarSignal", "BinaryPulsarSignal"}
        ),
        pytest_nodeids=lambda _waveform, _metadata: (_CW_TEST_NODEID,),
        pytest_args=_cw_pytest_args,
        resources=CW_RESOURCES,
        requires_ephemerides=True,
    ),
    TestAdapter(
        name="exact-pulsar-lal",
        matches=lambda waveform, _metadata: waveform == "ExactPulsarSignal",
        pytest_nodeids=lambda _waveform, _metadata: (_EXACT_PULSAR_TEST_NODEID,),
        pytest_args=_no_pytest_args,
        resources=CW_RESOURCES,
        requires_ephemerides=True,
    ),
)
"""Ordered adapter registry; append a new waveform-family test here."""


_UNSUPPORTED_REASONS: dict[str, str] = {}


class UnsupportedWaveformError(ValueError):
    """A registered waveform has no suitable large-scale reference adapter."""


def _unsupported_reason(waveform: str, metadata: Mapping[str, Any]) -> str:
    if waveform in _UNSUPPORTED_REASONS:
        return _UNSUPPORTED_REASONS[waveform]
    return (
        "no large-scale reference adapter is registered for "
        f"domain={metadata.get('domain', 'unknown')!r}, "
        f"source_type={metadata.get('source_type', 'unknown')!r}"
    )


def test_target(waveform: str) -> TestTarget:
    """Resolve one user-selected waveform to its registered test adapter."""

    ripplegw = _ripplegw_for_discovery()

    try:
        metadata = ripplegw.get_waveform_metadata(waveform)
    except ValueError as exc:
        available = ", ".join(ripplegw.list_waveforms())
        raise ValueError(
            f"Unknown waveform {waveform!r}. Available: {available}"
        ) from exc

    for adapter in TEST_ADAPTERS:
        if adapter.matches(waveform, metadata):
            return adapter.build_test_target(waveform, metadata)
    raise UnsupportedWaveformError(
        f"{waveform} has no large-scale test: "
        f"{_unsupported_reason(waveform, metadata)}."
    )


def all_test_targets() -> tuple[tuple[TestTarget, ...], dict[str, str]]:
    """Return supported targets plus explicit exclusions for ``--waveform all``."""

    ripplegw = _ripplegw_for_discovery()

    targets: list[TestTarget] = []
    excluded: dict[str, str] = {}
    for waveform in ripplegw.list_waveforms():
        try:
            targets.append(test_target(waveform))
        except UnsupportedWaveformError:
            metadata = ripplegw.get_waveform_metadata(waveform)
            excluded[waveform] = _unsupported_reason(waveform, metadata)
    return tuple(targets), excluded


def select_targets(selection: str) -> tuple[tuple[TestTarget, ...], dict[str, str]]:
    """Select one target, or every currently supported adapter target."""

    if selection == "all":
        return all_test_targets()
    return (test_target(selection),), {}


def resolve_resources(
    defaults: Resources,
    *,
    partition: str | None = None,
    cpus: int | None = None,
    gpus: int | None = None,
    memory: str | None = None,
    walltime: str | None = None,
) -> Resources:
    """Overlay user resource choices on method-specific defaults."""

    resources = Resources(
        partition=defaults.partition if partition is None else partition,
        cpus=defaults.cpus if cpus is None else cpus,
        gpus=defaults.gpus if gpus is None else gpus,
        memory=defaults.memory if memory is None else memory,
        walltime=defaults.walltime if walltime is None else walltime,
    )
    if resources.cpus < 1:
        raise ValueError("--cpus must be at least 1")
    if resources.gpus < 0:
        raise ValueError("--gpus cannot be negative")
    if not resources.memory:
        raise ValueError("--memory cannot be empty")
    if not resources.walltime:
        raise ValueError("--time cannot be empty")
    return resources


def waveform_outdir(base_outdir: Path, waveform: str) -> Path:
    """Keep logs, caches, and results separate for every selected waveform."""

    return base_outdir / waveform


def build_pytest_command(
    target: TestTarget,
    *,
    python: str,
    n_samples: int,
    outdir: Path,
    plots: bool,
) -> tuple[str, ...]:
    """Build exact pytest node IDs, avoiding broad marker or ``-k`` selection.

    This preserves test-level assertion behavior while ensuring an FD submission does
    not accidentally collect the expensive CW or time-domain tests.
    """

    if n_samples < 1:
        raise ValueError("--n-samples must be at least 1")
    command = [
        python,
        "-m",
        "pytest",
        *target.pytest_nodeids,
        "-v",
        "-s",
        "--n-samples",
        str(n_samples),
        "--outdir",
        str(outdir),
        *target.pytest_args,
    ]
    if target.adapter == "lal-frequency-domain":
        command.extend(("--reference", "lal", "--cache-reference"))
    if plots:
        command.append("--plots")
    return tuple(command)


def _job_slug(waveform: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", waveform).strip("-")


def cw_ephemerides(
    environ: Mapping[str, str], *, validate_files: bool
) -> tuple[str, str]:
    """Read CW ephemerides, validating real submissions but not dry-run previews."""

    keys = ("RIPPLE_EARTH_EPHEMERIS", "RIPPLE_SUN_EPHEMERIS")
    if not validate_files:
        return (
            environ.get(keys[0], f"<set-{keys[0]}>"),
            environ.get(keys[1], f"<set-{keys[1]}>"),
        )
    missing = [key for key in keys if not environ.get(key)]
    if missing:
        raise ValueError(
            "CW tests require " + ", ".join(missing) + " to name local ephemeris files"
        )
    earth, sun = environ[keys[0]], environ[keys[1]]
    absent = [path for path in (earth, sun) if not Path(path).is_file()]
    if absent:
        raise ValueError("CW ephemeris file(s) not found: " + ", ".join(absent))
    return earth, sun


def build_slurm_command(
    target: TestTarget,
    *,
    command: Sequence[str],
    outdir: Path,
    repo_dir: Path,
    resources: Resources,
    ephemerides: tuple[str, str] | None = None,
) -> tuple[str, ...]:
    """Build one ``sbatch`` argv sequence without invoking a local shell."""

    args = ["sbatch"]
    if resources.partition:
        args.append(f"--partition={resources.partition}")
    args.extend(
        (
            f"--time={resources.walltime}",
            "--ntasks=1",
            f"--cpus-per-task={resources.cpus}",
            f"--mem={resources.memory}",
        )
    )
    if resources.gpus:
        args.append(f"--gpus-per-task={resources.gpus}")
    if target.requires_ephemerides:
        if ephemerides is None:
            raise ValueError("CW Slurm target needs explicit ephemerides")
        earth, sun = ephemerides
        args.append(
            f"--export=ALL,RIPPLE_EARTH_EPHEMERIS={earth},RIPPLE_SUN_EPHEMERIS={sun}"
        )
    else:
        args.append("--export=ALL")
    wrapped = f"cd {shlex.quote(str(repo_dir))} && {shlex.join(command)}"
    args.extend(
        (
            f"--job-name=ripple-cv-{_job_slug(target.waveform)}",
            f"--output={outdir / 'test-%j.out'}",
            "--wrap",
            wrapped,
        )
    )
    return tuple(args)


def _condor_memory(memory: str) -> str:
    """Translate Slurm's common ``16G`` form to HTCondor's ``16GB`` form."""

    return f"{memory[:-1]}GB" if re.fullmatch(r"\d+[Gg]", memory) else memory


def _condor_argument_line(command: Sequence[str]) -> str:
    """Render direct argv elements as one HTCondor ``arguments`` value."""

    def quote(argument: str) -> str:
        escaped = argument.replace("\\", "\\\\").replace('"', '\\"')
        return (
            f'\\"{escaped}\\"'
            if not argument or any(c.isspace() for c in argument)
            else escaped
        )

    return '"' + " ".join(quote(arg) for arg in command[1:]) + '"'


def build_condor_submit(
    target: TestTarget,
    *,
    command: Sequence[str],
    outdir: Path,
    repo_dir: Path,
    resources: Resources,
) -> str:
    """Build one inspectable HTCondor submit description.

    ``getenv = True`` is deliberate: it propagates the explicitly validated CW
    ephemeris variables (and ordinary environment setup) into the execute node.
    """

    lines = [
        f"executable = {command[0]}",
        f"arguments = {_condor_argument_line(command)}",
        f"initialdir = {repo_dir}",
        f"output = {outdir / 'test-$(ClusterId).$(ProcId).out'}",
        f"error = {outdir / 'test-$(ClusterId).$(ProcId).err'}",
        f"log = {outdir / 'condor.log'}",
        "getenv = True",
        f"request_CPUs = {resources.cpus}",
        f"request_memory = {_condor_memory(resources.memory)}",
    ]
    if resources.gpus:
        lines.append(f"request_GPUs = {resources.gpus}")
    lines.append("queue")
    return "\n".join(lines) + "\n"


RunCommand = Callable[..., subprocess.CompletedProcess[Any]]


def _run_command(
    command: Sequence[str], *, cwd: Path | None = None
) -> subprocess.CompletedProcess[Any]:
    """Small injection point for scheduler-free tests."""

    return subprocess.run(command, cwd=cwd, check=False)


def _print_exclusions(excluded: Mapping[str, str]) -> None:
    if excluded:
        print("Registered waveforms without a large-scale adapter (not submitted):")
        for waveform, reason in excluded.items():
            print(f"  {waveform}: {reason}")


def _submit_one(
    scheduler: str,
    *,
    target: TestTarget,
    command: tuple[str, ...],
    outdir: Path,
    repo_dir: Path,
    resources: Resources,
    environ: Mapping[str, str],
    dry_run: bool,
    run: RunCommand,
) -> int:
    """Run or submit one target and return the immediate launcher status."""

    ephemerides = (
        cw_ephemerides(environ, validate_files=not dry_run)
        if target.requires_ephemerides
        else None
    )
    if scheduler == "local":
        print(f"Running {target.waveform} locally:\n  {shlex.join(command)}")
        return 0 if dry_run else run(command, cwd=repo_dir).returncode
    if scheduler == "slurm":
        submit_command = build_slurm_command(
            target,
            command=command,
            outdir=outdir,
            repo_dir=repo_dir,
            resources=resources,
            ephemerides=ephemerides,
        )
        print(f"Submitting {target.waveform} to Slurm:\n  {shlex.join(submit_command)}")
        return 0 if dry_run else run(submit_command, cwd=repo_dir).returncode
    if scheduler == "condor":
        submit_text = build_condor_submit(
            target,
            command=command,
            outdir=outdir,
            repo_dir=repo_dir,
            resources=resources,
        )
        submit_file = outdir / "test.sub"
        print(f"HTCondor submit file for {target.waveform}: {submit_file}")
        print(submit_text, end="")
        if dry_run:
            return 0
        submit_file.write_text(submit_text)
        return run(("condor_submit", str(submit_file)), cwd=repo_dir).returncode
    raise ValueError(f"Unknown scheduler {scheduler!r}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scheduler",
        choices=("local", "slurm", "condor"),
        default="local",
        help="Where to run the selected test (default: local).",
    )
    parser.add_argument(
        "--waveform",
        required=True,
        metavar="NAME|all",
        help="Registered waveform to validate, or 'all' for every available adapter.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=1000,
        help="Random trials per selected waveform (default: 1000).",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Base result directory; a subdirectory is created per selected waveform.",
    )
    parser.add_argument("--plots", action="store_true", help="Write optional plots.")
    parser.add_argument(
        "--partition",
        default=None,
        help="Partition override (default depends on method).",
    )
    parser.add_argument(
        "--cpus",
        type=int,
        default=None,
        help="CPU override (default depends on method).",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        default=None,
        help="GPU override (default depends on method).",
    )
    parser.add_argument("--memory", default=None, help="Memory override, e.g. 16G.")
    parser.add_argument(
        "--time",
        dest="walltime",
        default=None,
        help="Wall-time override, e.g. 04:00:00.",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Interpreter made available to batch jobs (default: this interpreter).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands/submit descriptions without running or submitting them.",
    )
    return parser


def _default_outdir() -> Path:
    return (
        _REPO_DIR
        / "tests"
        / "cross_validation"
        / "results"
        / datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    run: RunCommand = _run_command,
) -> int:
    """CLI entry point; ``run`` is injectable so tests never need a scheduler."""

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.n_samples < 1:
        parser.error("--n-samples must be at least 1")
    if args.cpus is not None and args.cpus < 1:
        parser.error("--cpus must be at least 1")
    if args.gpus is not None and args.gpus < 0:
        parser.error("--gpus cannot be negative")
    try:
        targets, excluded = select_targets(args.waveform)
    except ValueError as exc:
        parser.error(str(exc))

    env = os.environ if environ is None else environ
    base_outdir = Path(args.outdir) if args.outdir else _default_outdir()
    base_outdir = base_outdir.expanduser().resolve()
    _print_exclusions(excluded)

    statuses = []
    for target in targets:
        outdir = waveform_outdir(base_outdir, target.waveform)
        try:
            resources = resolve_resources(
                target.resources,
                partition=args.partition,
                cpus=args.cpus,
                gpus=args.gpus,
                memory=args.memory,
                walltime=args.walltime,
            )
            command = build_pytest_command(
                target,
                # Do not resolve a virtualenv's ``python`` symlink: resolving it
                # changes ``sys.prefix`` in the batch job and drops the venv's
                # installed test dependencies.  Make a relative override absolute
                # while preserving the spelling of an absolute interpreter path.
                python=str(
                    Path(args.python).expanduser()
                    if Path(args.python).expanduser().is_absolute()
                    else Path.cwd() / Path(args.python).expanduser()
                ),
                n_samples=args.n_samples,
                outdir=outdir,
                plots=args.plots,
            )
            if not args.dry_run:
                outdir.mkdir(parents=True, exist_ok=True)
            statuses.append(
                _submit_one(
                    args.scheduler,
                    target=target,
                    command=command,
                    outdir=outdir,
                    repo_dir=_REPO_DIR,
                    resources=resources,
                    environ=env,
                    dry_run=args.dry_run,
                    run=run,
                )
            )
        except ValueError as exc:
            parser.error(str(exc))
    return 1 if any(status != 0 for status in statuses) else 0


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
