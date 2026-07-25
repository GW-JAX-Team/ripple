"""Guards on tests/cross_validation/tolerances.toml, no reference backend needed.

1. Every waveform an available backend claims to ``supports()`` has a
   tolerance entry -- a new model cannot land with its agreement unrecorded.
2. The overlap-loss thresholds match the documented table in
   docs/dev/lal_agreement.md, so that table cannot silently drift from what
   is actually enforced.
"""

import re
from pathlib import Path

import pytest

import ripplegw
from tests.cross_validation.campaign import load_tolerances
from tests.helpers.reference import REFERENCE_BACKENDS, get_backend

_TOLERANCES = load_tolerances()
_REPO_ROOT = Path(__file__).parents[2]
_LAL_AGREEMENT_MD = _REPO_ROOT / "docs" / "dev" / "lal_agreement.md"


@pytest.mark.parametrize("backend_name", sorted(REFERENCE_BACKENDS))
def test_every_supported_waveform_has_a_tolerance_entry(backend_name):
    backend = get_backend(backend_name)
    backend_table = _TOLERANCES.get(backend_name, {})
    missing = [
        name
        for name in ripplegw.list_waveforms()
        if backend.supports(name) and name not in backend_table
    ]
    assert not missing, (
        f"{backend_name}: {missing} are supported but have no "
        f"[{backend_name}.<name>] block in tests/cross_validation/tolerances.toml"
    )


def _parse_lal_agreement_thresholds() -> dict[str, float]:
    """Extract the ``| Waveform | Threshold | ... |`` summary table."""
    text = _LAL_AGREEMENT_MD.read_text()
    thresholds = {}
    for line in text.splitlines():
        m = re.match(r"\|\s*(\w[\w.]*)\s*\|\s*([0-9.eE+-]+)\s*\|", line)
        if m:
            thresholds[m.group(1)] = float(m.group(2))
    return thresholds


def test_lal_agreement_doc_matches_tolerances_toml():
    documented = _parse_lal_agreement_thresholds()
    lal_table = _TOLERANCES["lal"]
    for name, threshold in documented.items():
        if name not in lal_table:
            continue  # non-waveform table rows (there are none today, but be lenient)
        assert lal_table[name]["overlap_loss"] == threshold, (
            f"docs/dev/lal_agreement.md documents {name} threshold={threshold:.0e}, "
            f"but tests/cross_validation/tolerances.toml has "
            f"{lal_table[name]['overlap_loss']:.0e} -- update whichever is stale."
        )
