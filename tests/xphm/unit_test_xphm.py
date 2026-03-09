import lalsimulation as lalsim
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import lal
import matplotlib.pyplot as plt
from ripplegw.waveforms import IMRPhenomXPHM
import bilby
from bilby.gw.utils import noise_weighted_inner_product

psd = bilby.gw.detector.PowerSpectralDensity(psd_file="./ET_D_psd.txt")

"""
Unit test to 

1. generate one XPHM waveform using ripple and lalsimulation.
2. Plot them
3. Compute match between them. 

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

    return inner_product / np.power(norm_1 * norm_2, 0.5)


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


def plot_xphm_comparison(
    ripple_hp, lal_hp, minimum_frequency, maximum_frequency, duration, output="xphm.pdf"
):
    N = int(minimum_frequency * duration)
    f = np.arange(minimum_frequency, maximum_frequency, 1.0 / duration)
    lal_f = np.arange(0.0, maximum_frequency, 1.0 / duration)
    lal_hp_data = lal_hp.data.data[:-1]

    ripple_amp = np.abs(ripple_hp)
    ripple_phase = np.unwrap(np.angle(ripple_hp))
    lal_amp = np.abs(lal_hp_data)
    lal_phase = np.unwrap(np.angle(lal_hp_data))

    fig, ax = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    ax[0].plot(f, ripple_amp, label="ripple")
    ax[0].plot(lal_f, lal_amp, label="lalsim", linestyle="--")
    ax[0].plot(
        f, abs(ripple_amp - np.abs(lal_amp[N:])), label="difference", color="black"
    )
    ax[0].set_yscale("log")
    ax[0].set_xlim(15, 100)
    ax[0].set_ylabel("Amplitude")
    ax[0].legend()
    ax[0].set_title("Amplitude XPHM")

    ax[1].plot(f, ripple_phase, label="ripple")
    ax[1].plot(lal_f, lal_phase, label="lalsim", linestyle="--")
    ax[1].plot(
        f, abs(ripple_phase - lal_phase[N:]), label="phase difference", color="black"
    )
    ax[1].set_xlim(15, 100)
    ax[1].set_ylabel("Phase [rad]")
    ax[1].legend()
    ax[1].set_yscale("log")
    ax[1].set_title("Phase XPHM")

    ax[2].plot(f, np.real(ripple_hp), label="ripple")
    ax[2].plot(lal_f, np.real(lal_hp_data), label="lalsim", linestyle="--")
    ax[2].set_xlim(5, 50)
    ax[2].set_xlabel("Frequency [Hz]")
    ax[2].set_ylabel("Real(h+)")
    ax[2].legend()
    ax[2].set_title("Full XPHM Waveform (Real)")

    plt.tight_layout()
    fig.savefig(output)


def main():
    print("Device", jax.devices())

    """

    injection_parameters = {}
    injection_parameters["mass_1"] = 54.1
    injection_parameters["mass_2"] = 39.6
    injection_parameters["mass_1_SI"] = injection_parameters["mass_1"] * MSUN
    injection_parameters["mass_2_SI"] = injection_parameters["mass_2"] * MSUN
    injection_parameters["distance"] = 1.0  # In Mpc
    injection_parameters["distance_SI"] = (
        injection_parameters["distance"] * 3.0856775814913673e22
    )  # In meters
    injection_parameters["theta"] = 0.5
    injection_parameters["phi"] = 0.0
    injection_parameters["iota"] = 0.2
    injection_parameters["psi"] = 1.2
    injection_parameters["phase"] = 0.0
    injection_parameters["spin_1x"] = -0.02
    injection_parameters["spin_1y"] = 0.564
    injection_parameters["spin_1z"] = -0.3
    injection_parameters["spin_2x"] = -0.86
    injection_parameters["spin_2y"] = -0.45
    injection_parameters["spin_2z"] = -0.08
    """

    injection_parameters = {
        "mass_ratio": 0.5376651738645067,
        "chirp_mass": 93.88321707875129,
        "luminosity_distance": 4320.569794212493,
        "dec": -0.5340624003795874,
        "ra": 1.3155913002794608,
        "theta_jn": 0.8231599643967205,
        "psi": 1.8350045296023523,
        "phase": 0.0,
        "a_1": 0.8397132152576712,
        "a_2": 0.8404086003139519,
        "tilt_1": 1.991385939862487,
        "tilt_2": 1.6627955891549548,
        "phi_12": 1.5821686909994463,
        "phi_jl": 3.1850539903859567,
        "total_mass": 228.30475292075536,
        "mass_1": 148.47494552209497,
        "mass_2": 79.82980739866036,
        "reference_frequency": 50.0,
        "iota": 1.0208893498397191,
        "spin_1x": -0.7474522736209998,
        "spin_1y": 0.16995474427151677,
        "spin_1z": -0.3428538572581289,
        "spin_2x": -0.17625493757902752,
        "spin_2y": -0.8180829693890946,
        "spin_2z": -0.07720795073171265,
        "phi_1": 2.9180152641002444,
        "phi_2": 4.50018395509969,
        "luminosity_distance_SI": 1.3331885353270261e26,
        "mass_1_SI": 2.9522904722748898e32,
        "mass_2_SI": 1.5873437700742054e32,
    }

    minimum_frequency = 10
    maximum_frequency = 1024
    duration = 8.0
    reference_frequency = 50
    modes = jnp.array([[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]], dtype=jnp.int32)

    lalsim_plus, lalsim_cross = generate_lalsimulation_xphm_waveform(
        injection_parameters,
        minimum_frequency,
        maximum_frequency,
        reference_frequency,
        duration,
        modes,
    )

    ripple_plus, ripple_cross = generate_ripple_xphm_waveform(
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
    print("plus log10 mismatch", np.log10(np.abs(np.real(1 - plus_overlap))))

    plot_xphm_comparison(
        ripple_plus, lalsim_plus, minimum_frequency, maximum_frequency, duration
    )


if __name__ == "__main__":
    main()
