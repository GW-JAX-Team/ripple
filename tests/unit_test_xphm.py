import lalsimulation as lalsim
import jax
import jax.numpy as jnp
import numpy as np
from ripplegw.constants import MSUN
import lal
import matplotlib.pyplot as plt
from ripplegw.waveforms import IMRPhenomXPHM


def compute_overlap(frequency_series_1, frequency_series_2):
    normass_1 = np.sum(frequency_series_1 * np.conj(frequency_series_1)) ** 0.5
    normass_2 = np.sum(frequency_series_2 * np.conj(frequency_series_2)) ** 0.5
    inner_product = np.sum(frequency_series_1 * np.conj(frequency_series_2))
    return inner_product / (normass_1 * normass_2)


def generate_lalsimulation_xphm_waveform(
    injection_parameters,
    minimum_frequency,
    maximum_frequency,
    reference_frequency,
    duration,
    modes,
):
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
        injection_parameters["mass_1_SI"][0],
        injection_parameters["mass_2_SI"][0],
        injection_parameters["chi1x"][0],
        injection_parameters["chi1y"][0],
        injection_parameters["chi1z"][0],
        injection_parameters["chi2x"][0],
        injection_parameters["chi2y"][0],
        injection_parameters["chi2z"][0],
        injection_parameters["distance_SI"][0],
        injection_parameters["iota"][0],
        injection_parameters["Phicoal"][0],
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
    frequency_array = jnp.arange(minimum_frequency, maximum_frequency, 1 / duration)
    hp, hc = IMRPhenomXPHM.generate_xphm(
        injection_parameters["mass_1"][0],
        injection_parameters["mass_2"][0],
        injection_parameters["chi1x"][0],
        injection_parameters["chi1y"][0],
        injection_parameters["chi1z"][0],
        injection_parameters["chi2x"][0],
        injection_parameters["chi2y"][0],
        injection_parameters["chi2z"][0],
        injection_parameters["distance"][0],
        injection_parameters["iota"][0],
        injection_parameters["Phicoal"][0],
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

    fig, ax = plt.subplots(3, 1, figsize=(10, 12))

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
    ax[1].set_title("Phase XPHM")

    ax[2].plot(f, np.real(ripple_hp), label="ripple")
    ax[2].plot(lal_f, np.real(lal_hp_data), label="lalsim", linestyle="--")
    ax[2].set_xlim(15, 80)
    ax[2].set_xlabel("Frequency [Hz]")
    ax[2].set_ylabel("Real(h+)")
    ax[2].legend()
    ax[2].set_title("Full XPHM Waveform (Real)")

    plt.tight_layout()
    fig.savefig(output)


def main():
    print("Device", jax.devices())

    injection_parameters = {}
    injection_parameters["mass_1"] = np.array([36.0])
    injection_parameters["mass_2"] = np.array([9.0])
    injection_parameters["mass_1_SI"] = injection_parameters["mass_1"] * MSUN
    injection_parameters["mass_2_SI"] = injection_parameters["mass_2"] * MSUN
    injection_parameters["distance"] = np.array([1])  # In Mpc
    injection_parameters["distance_SI"] = np.array(
        [injection_parameters["distance"][0] * 3.0856775814913673e22]
    )  # In meters
    injection_parameters["theta"] = np.array([0.5])
    injection_parameters["phi"] = np.array([0.0])
    injection_parameters["iota"] = np.array([0.2])
    injection_parameters["psi"] = np.array([1.2])
    injection_parameters["Phicoal"] = np.array([0.0])
    injection_parameters["chi1x"] = np.array([0.1])
    injection_parameters["chi1y"] = np.array([0.2])
    injection_parameters["chi1z"] = np.array([0.3])
    injection_parameters["chi2x"] = np.array([0.3])
    injection_parameters["chi2y"] = np.array([0.2])
    injection_parameters["chi2z"] = np.array([0.1])

    minimum_frequency = 20
    maximum_frequency = 1024
    duration = 8.0
    reference_frequency = 50
    modes = jnp.array([[2, 1], [2, 2], [3, 2], [3, 3], [4, 4]])

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
        ripple_plus, np.array(lalsim_plus.data.data[mask:-1])
    )
    print("Plus overlap percentage", 100 * (1 - plus_overlap))

    plot_xphm_comparison(
        ripple_plus, lalsim_plus, minimum_frequency, maximum_frequency, duration
    )


if __name__ == "__main__":
    main()
