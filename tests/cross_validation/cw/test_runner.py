"""Pure selection behavior for the selected CW large-scale test."""

import pytest

from tests.cross_validation.cw.runner import sample_trials


@pytest.mark.parametrize(
    ("waveform", "is_binary"),
    (("PulsarSignal", False), ("BinaryPulsarSignal", True)),
)
def test_selected_cw_large_scale_test_samples_only_the_requested_population(
    waveform, is_binary
):
    trials = sample_trials(5, waveform=waveform)

    assert [trial.index for trial in trials] == list(range(5))
    assert all(trial.is_binary is is_binary for trial in trials)


def test_unselected_cw_large_scale_test_preserves_mixed_population():
    trials = sample_trials(6)

    assert [trial.is_binary for trial in trials] == [
        False,
        True,
        False,
        True,
        False,
        True,
    ]


def test_selected_cw_large_scale_test_rejects_unknown_waveform():
    with pytest.raises(ValueError, match="Unknown CW test waveform"):
        sample_trials(1, waveform="ExactPulsarSignal")
