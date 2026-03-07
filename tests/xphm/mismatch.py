# %%
import numpy as np
import matplotlib.pyplot as plt
import bilby
import lalsimulation as lalsim
import jax.numpy as jnp
from ripplegw.waveforms import IMRPhenomXPHM
import lal
from ripplegw.constants import MSUN
from bilby.gw.utils import noise_weighted_inner_product
from tqdm import tqdm
import scienceplots

scienceplots
plt.style.use(["science"])
psd = bilby.gw.detector.PowerSpectralDensity(psd_file="ET_D_psd.txt")

"""
Script to compute mismatch between lalsimulation and ripple implementation of IMRPhenomXPHM waveform.
The script will force lalsimulation to use MSA prescription. If lalsimulation falls back on NNLO, the script will ignore that waveform. 
"""


def compute_overlap(
    frequency_series_1,
    frequency_series_2,
    minimum_frequency,
    maximum_frequency,
    duration,
):
    frequency_array = np.arange(minimum_frequency, maximum_frequency, 1 / duration)
    interpolated_psd = psd.power_spectral_density_interpolated(frequency_array)

    inner_product = noise_weighted_inner_product(
        frequency_series_1, frequency_series_2, interpolated_psd, duration
    )

    norm_1 = noise_weighted_inner_product(
        frequency_series_1, frequency_series_1, interpolated_psd, duration
    )
    norm_2 = noise_weighted_inner_product(
        frequency_series_2, frequency_series_2, interpolated_psd, duration
    )

    overlap = inner_product / np.power(norm_1 * norm_2, 0.5)
    overlap_real = np.real(overlap)

    return abs(overlap_real)


def setup_injection_parameters(N_WAVEFORMS, reference_frequency):
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
    _injection_parameters["reference_frequency"] = (
        np.ones(N_WAVEFORMS) * reference_frequency
    )

    _injection_parameters = bilby.gw.conversion.generate_component_spins(
        _injection_parameters
    )

    _injection_parameters["luminosity_distance_SI"] = (
        _injection_parameters["luminosity_distance"] * 3.0856775814913673e22
    )  # In meters
    _injection_parameters["mass_1_SI"] = _injection_parameters["mass_1"] * MSUN
    _injection_parameters["mass_2_SI"] = _injection_parameters["mass_2"] * MSUN
    injection_parameters = {
        key: jnp.array(value) for key, value in _injection_parameters.items()
    }

    return injection_parameters


def plot_mismatch_histogram(collect_mismatch, output="mismatch_histogram.pdf"):
    collect_mismatch = np.real(np.array(collect_mismatch))
    fig, ax = plt.subplots(1, 1)
    _close_to_zero = np.log10(1e-20) - 1
    minimum_match = (
        _close_to_zero
        if min(collect_mismatch) == 0.0
        else (np.log10(min(collect_mismatch)) - 1)
    )
    maximum_match = np.log10(max(collect_mismatch))
    ax.hist(
        collect_mismatch,
        cumulative=1,
        histtype="step",
        lw=2,
        bins=10.0 ** np.linspace(minimum_match, maximum_match, 40),
    )
    ax.set_ylabel("Fraction of waveforms")
    ax.set_title(f"N = {len(collect_mismatch)}")
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

    try:
        hp, hc = lalsim.SimIMRPhenomXPHM(
            injection_parameters["mass_1_SI"],
            injection_parameters["mass_2_SI"],
            injection_parameters["spin_1x"],
            injection_parameters["spin_1y"],
            injection_parameters["spin_1z"],
            injection_parameters["spin_2x"],
            injection_parameters["spin_2y"],
            injection_parameters["spin_2z"],
            injection_parameters["luminosity_distance_SI"],
            injection_parameters["iota"],
            injection_parameters["phase"],
            minimum_frequency,
            maximum_frequency,
            1 / duration,
            reference_frequency,
            lalparams,
        )

    except Exception as e:
        print(f"Failed with: {e}")

        print(
            "Evaluating the waveform failed with an error.\n"
            + "The parameters were {}\n".format(injection_parameters)
        )

        frequency_array_length = int(duration * maximum_frequency) + 1

        hp = lal.CreateCOMPLEX16FrequencySeries(
            "hplus",
            lal.LIGOTimeGPS(0),
            0.0,
            1 / duration,
            lal.DimensionlessUnit,
            frequency_array_length,
        )
        hc = lal.CreateCOMPLEX16FrequencySeries(
            "hcross",
            lal.LIGOTimeGPS(0),
            0.0,
            1 / duration,
            lal.DimensionlessUnit,
            frequency_array_length,
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
        injection_parameters["spin_1x"],
        injection_parameters["spin_1y"],
        injection_parameters["spin_1z"],
        injection_parameters["spin_2x"],
        injection_parameters["spin_2y"],
        injection_parameters["spin_2z"],
        injection_parameters["luminosity_distance"],
        injection_parameters["iota"],
        injection_parameters["phase"],
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
            key: float(value[ii]) for key, value in batch_injection_parameters.items()
        }

        try:
            lalsim_plus, _ = generate_lalsimulation_xphm_waveform(
                injection_parameters,
                minimum_frequency,
                maximum_frequency,
                reference_frequency,
                duration,
                modes,
            )
        except Exception:
            print("The lalsimulation waveform generation failed.")
            continue

        ripple_plus, _ = generate_ripple_xphm_waveform(
            injection_parameters,
            minimum_frequency,
            maximum_frequency,
            reference_frequency,
            duration,
        )

        mask = int(minimum_frequency * duration)
        plus_overlap = compute_overlap(
            ripple_plus,
            np.array(lalsim_plus.data.data[mask:-1]),
            minimum_frequency,
            maximum_frequency,
            duration,
        )

        if plus_overlap > 1:
            continue
        mismatch = 1.0 - plus_overlap
        collect_mismatch.append(mismatch)

        debug = False
        if debug:
            log_mismatch = np.log10(abs(np.real(mismatch)))
            if log_mismatch > -4:
                print(f"High Mismatch {log_mismatch}")
                print("Injection parameters")
                print(injection_parameters)
                print("----------\n")

    return collect_mismatch


def main():
    N_injections = int(1e4)
    seed = 2
    np.random.seed(seed)
    bilby.core.utils.random.seed(seed)
    # Frequency settings
    minimum_frequency = 10.0
    maximum_frequency = 1024.0
    duration = 8.0
    reference_frequency = 50.0
    modes = jnp.array([[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]], dtype=jnp.int32)

    # Waveform parameters batch
    batch_injection_parameters = setup_injection_parameters(
        N_injections, reference_frequency
    )
    batch_injection_parameters["phase"] = np.zeros(N_injections)
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

# %%
