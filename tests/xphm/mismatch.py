# %%
import numpy as np
import matplotlib.pyplot as plt
import bilby
import lalsimulation as lalsim
import jax.numpy as jnp
from ripplegw.waveforms import IMRPhenomXPHM
import lal
from tqdm import tqdm


def compute_overlap(frequency_series_1, frequency_series_2):
    norm1 = np.sum(frequency_series_1 * np.conj(frequency_series_1)) ** 0.5
    norm2 = np.sum(frequency_series_2 * np.conj(frequency_series_2)) ** 0.5

    inner_product = np.sum(frequency_series_1 * np.conj(frequency_series_2))
    return inner_product / (norm1 * norm2)


def setup_injection_parameters(N_WAVEFORMS):
    # Waveform parameters
    """
    Set up the injection parameters from bilby prior
    """
    population = bilby.gw.prior.BBHPriorDict()
    population["chirp_mass"] = bilby.core.prior.Uniform(10, 100)
    population["mass_ratio"] = bilby.core.prior.Uniform(0.25, 1)
    population.pop("mass_1")
    population.pop("mass_2")

    _injection_parameters = bilby.gw.conversion.generate_component_masses(
        population.sample(N_WAVEFORMS)
    )
    _injection_parameters["reference_frequency"] = np.ones(N_WAVEFORMS) * 50

    _injection_parameters = bilby.gw.conversion.generate_component_spins(
        _injection_parameters
    )

    injection_parameters = {
        key: jnp.array(value) for key, value in _injection_parameters.items()
    }

    return injection_parameters


def plot_mismatch_histogram(collect_mismatch, output="mismatch_histogram.pdf"):
    collect_mismatch = abs(np.real(np.array(collect_mismatch)))
    fig, ax = plt.subplots(1, 1)
    _close_to_zero = np.log10(1e-20) - 1
    minimum_match = (
        _close_to_zero
        if min(collect_mismatch) == 0.0
        else (np.log10(min(collect_mismatch)) - 1)
    )
    maximum_match = np.log10(max(collect_mismatch)) + 1
    ax.hist(
        collect_mismatch,
        cumulative=1,
        histtype="step",
        lw=2,
        bins=10.0 ** np.arange(minimum_match, maximum_match, 0.2),
    )
    ax.set_ylabel("Fraction of events")
    ax.set_title(f"N = {len(collect_mismatch)}, XPHM")
    ax.set_xlabel("Mismatch")
    ax.set_xscale("log")
    ax.grid(alpha=0.5)
    fig.savefig(output)


def generate_lalsimulation_xphm_waveform(
    injection_parameters,
    minimum_frequency,
    maximum_frequency,
    reference_frequency,
    duration,
    modes,
):
    """
    Wrapper to call the lalsimulation XPHM waveform.
    """
    lalparams = lal.CreateDict()

    ModeArray = lalsim.SimInspiralCreateModeArray()
    for mm in modes:
        lalsim.SimInspiralModeArrayActivateMode(ModeArray, int(mm[0]), int(mm[1]))

    lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, ModeArray)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(lalparams, 1)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMMBandVersion(lalparams, 0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lalparams, 0.0)
    lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalparams, 223)

    hp, hc = lalsim.SimIMRPhenomXPHM(
        injection_parameters["mass_1_SI"],
        injection_parameters["mass_2_SI"],
        injection_parameters["chi1x"],
        injection_parameters["chi1y"],
        injection_parameters["chi1z"],
        injection_parameters["chi2x"],
        injection_parameters["chi2y"],
        injection_parameters["chi2z"],
        injection_parameters["distance_SI"],
        injection_parameters["iota"],
        injection_parameters["Phicoal"],
        minimum_frequency,
        maximum_frequency,
        1 / duration,
        reference_frequency,
        lalparams,
    )
    return hp, hc


def generate_ripple_xphm_waveform(
    injection_parameters,
    minimum_frequency,
    maximum_frequency,
    reference_frequency,
    duration,
):
    """
    Wrapper to call the ripple XPHM waveform.
    """
    frequency_array = jnp.arange(minimum_frequency, maximum_frequency, 1 / duration)
    hp, hc = IMRPhenomXPHM.generate_xphm(
        injection_parameters["mass_1"],
        injection_parameters["mass_2"],
        injection_parameters["chi1x"],
        injection_parameters["chi1y"],
        injection_parameters["chi1z"],
        injection_parameters["chi2x"],
        injection_parameters["chi2y"],
        injection_parameters["chi2z"],
        injection_parameters["distance"],
        injection_parameters["iota"],
        injection_parameters["Phicoal"],
        frequency_array,
        reference_frequency,
    )
    return hp, hc


def compute_mismatch_loop(
    batch_injection_parameters,
    minimum_frequency,
    maximum_frequency,
    reference_frequency,
    duration,
    modes,
):
    "Main loop to call the lalsimulation and ripple XPHM waveform"
    "Compute and collect mismatch"
    collect_mismatch = []
    N_injections = len(batch_injection_parameters["mass_1"])
    for ii in tqdm(range(N_injections)):
        injection_parameters = {
            key: value[ii] for key, value in batch_injection_parameters.items()
        }

        lalsim_plus, _ = generate_lalsimulation_xphm_waveform(
            injection_parameters,
            minimum_frequency,
            maximum_frequency,
            reference_frequency,
            duration,
            modes,
        )

        ripple_plus, _ = generate_ripple_xphm_waveform(
            injection_parameters,
            minimum_frequency,
            maximum_frequency,
            reference_frequency,
            duration,
        )

        mask = int(minimum_frequency * duration)
        plus_overlap = compute_overlap(
            ripple_plus, np.array(lalsim_plus.data.data[mask:-1])
        )

        mismatch = 1 - plus_overlap
        collect_mismatch.append(mismatch)
    return collect_mismatch


def main():
    N_injections = int(100)
    seed = 3232
    np.random.seed(seed)
    bilby.core.utils.random.seed(seed)
    # Frequency settings
    minimum_frequency = 20.0
    maximum_frequency = 512.0
    duration = 4.0
    reference_frequency = 50.0
    modes = jnp.array([[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]], dtype=jnp.int32)

    # Waveform parameters batch
    batch_injection_parameters = setup_injection_parameters(N_injections)
    collect_mismatch = compute_mismatch_loop(
        batch_injection_parameters,
        minimum_frequency,
        maximum_frequency,
        reference_frequency,
        duration,
        modes,
    )

    plot_mismatch_histogram(collect_mismatch)


if __name__ == "__main__":
    main()
