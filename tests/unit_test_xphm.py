import lalsimulation as lalsim
import jax
import jax.numpy as jnp
import numpy as np
from ripplegw.constants import C, MSUN
import lal
import matplotlib.pyplot as plt
from ripplegw.waveforms import IMRPhenomXPHM
import bilby
from utils import GPSt_to_LMST
print("Device", jax.devices())

def compute_overlap(frequency_series_1, frequency_series_2):

    norm1 = np.sum(frequency_series_1*np.conj(frequency_series_1))**0.5
    norm2 = np.sum(frequency_series_2*np.conj(frequency_series_2))**0.5

    inner_product = np.sum(frequency_series_1*np.conj(frequency_series_2))
    return inner_product / (norm1*norm2)

injection_parameters = {}
injection_parameters['m1'] = np.array([36.0])
injection_parameters['m2'] = np.array([9.0])

injection_parameters['m1_SI'] = injection_parameters['m1'] * MSUN
injection_parameters['m2_SI'] = injection_parameters['m2'] * MSUN


injection_parameters['chirp_mass'] = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['m1'], 
                                                                                injection_parameters['m2'])

injection_parameters['distance'] = np.array([0.001]) # In GPc

injection_parameters['distance_SI'] = np.array([0.001 * 3.0856775814913673e25])
injection_parameters['theta'] = np.array([0.5])

injection_parameters['phi'] = np.array([0.])

injection_parameters['iota'] = np.array([0])

injection_parameters['psi'] = np.array([0.])

injection_parameters['eta'] = injection_parameters['m1'] * injection_parameters['m2'] / (injection_parameters['m1'] + injection_parameters['m2'])**2

injection_parameters['Phicoal'] = np.array([0.])

injection_parameters['chi1x'] = np.array([.1])
injection_parameters['chi1y'] = np.array([.2])
injection_parameters['chi1z'] = np.array([.3])

injection_parameters['chi2x'] = np.array([.3])
injection_parameters['chi2y'] = np.array([.2])
injection_parameters['chi2z'] = np.array([.1])


minimum_frequency = 20
maximum_frequency = 1024
duration = 8.
df = 1/duration
reference_frequency = 50
modes = jnp.array([[2,1],[2,2],[3,2],[3,3],[4,4]])

f = np.arange(minimum_frequency, maximum_frequency, df)
lalparams = lal.CreateDict()

ModeArray = lalsim.SimInspiralCreateModeArray()

for mm in modes:
    lalsim.SimInspiralModeArrayActivateMode(ModeArray, int(mm[0]), int(mm[1]))





lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, ModeArray)
lalsim.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(lalparams, 1)
lalsim.SimInspiralWaveformParamsInsertPhenomXPHMMBandVersion(lalparams, 0)
lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lalparams, 0.0)
lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalparams, 223)

lal_hp_xphm, lal_hc_xphm = lalsim.SimIMRPhenomXPHM(injection_parameters['m1_SI'][0],                       
                                               injection_parameters['m2_SI'][0],                    
                                               injection_parameters['chi1x'][0],                        #/**< x-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
                                               injection_parameters['chi1y'][0],                        #/**< y-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
                                               injection_parameters['chi1z'][0],                        #/**< z-component of the dimensionless spin of object 1 w.r.t. Lhat = (0,0,1) */
                                               injection_parameters['chi2x'][0],                        #/**< x-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
                                               injection_parameters['chi2y'][0],                        #/**< y-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
                                               injection_parameters['chi2z'][0],                        #/**< z-component of the dimensionless spin of object 2 w.r.t. Lhat = (0,0,1) */
                                               injection_parameters['distance_SI'][0],                     #/**< Distance of source (m) */
                                               injection_parameters['iota'][0],                  #/**< inclination of source (rad) */
                                               injection_parameters['Phicoal'][0],                       #/**< Orbital phase (rad) at reference frequency */
                                               minimum_frequency,                        #/**< Starting GW frequency (Hz) */
                                               maximum_frequency,                        #/**< Ending GW frequency (Hz); Defaults to Mf = 0.3 if no f_max is specified. */
                                               df,                       #/**< Sampling frequency (Hz). To use non-uniform frequency grid, set deltaF <= 0. */
                                               reference_frequency,                      #/**< Reference frequency (Hz) */
                                               lalparams                  #/**< LAL Dictionary struct */
                                               )

###### jax code
tGPS = 3600
#model = IMRPhenomXPHM.IMRPhenomXPHM(apply_fcut = True, reference_frequency=reference_frequency)

make_ripple_hlms = True
if make_ripple_hlms:


    extra_params = {"ModeArray": modes}
    
    hlms_ripple = IMRPhenomXPHM.XLALSimIMRPhenomHMGethlmModes(
    f,
    injection_parameters['m1_SI'][0],
    injection_parameters['m2_SI'][0],
    injection_parameters['chi1x'][0],
    injection_parameters['chi1y'][0],
    injection_parameters['chi1z'][0],
    injection_parameters['chi2x'][0],
    injection_parameters['chi2y'][0],
    injection_parameters['chi2z'][0],
    0.0,
    df,
    reference_frequency,
    extra_params)
    
    
    Mtot = injection_parameters['m1'][0] + injection_parameters['m2'][0]
    dist_m = injection_parameters['distance_SI'][0]
    amp0 = Mtot * lal.MRSUN_SI * Mtot * lal.MTSUN_SI / dist_m
    
    print("ripple amp0", amp0)
    ells = modes[:, 0]
    minus1l = jnp.where(ells % 2 != 0, -1, 1)
    hlms_ripple_final = minus1l[:, None] * hlms_ripple * amp0



run_jim_xphm = True

if run_jim_xphm:


    ripple_hp_xphm, ripple_hc_xphm = IMRPhenomXPHM.generate_xphm(injection_parameters['m1'][0],
                                           injection_parameters['m2'][0],
                                            injection_parameters['chi1x'][0],
                                            injection_parameters['chi1y'][0],
                                            injection_parameters['chi1z'][0],
                                            injection_parameters['chi2x'][0],
                                            injection_parameters['chi2y'][0],
                                            injection_parameters['chi2z'][0],
                                            injection_parameters['distance'][0],
                                            injection_parameters['iota'][0],
                                            injection_parameters['Phicoal'][0],
                                            duration,
                                            minimum_frequency, maximum_frequency, reference_frequency,
                                            modes)


    N = int(minimum_frequency * duration)

    plus_overlap = compute_overlap(ripple_hp_xphm, np.array(lal_hp_xphm.data.data[N:-1]))
    print("Plus overlap percentage", 100*(1-plus_overlap))



    lal_f = np.arange(0., maximum_frequency, 1./duration)
    plot_xphm_hp = lal_hp_xphm.data.data[:-1]




    # Compute amplitude and phase
    ripple_amp = np.abs(ripple_hp_xphm)
    ripple_phase = np.unwrap(np.angle(ripple_hp_xphm))
    lal_amp = np.abs(plot_xphm_hp)
    lal_phase = np.unwrap(np.angle(plot_xphm_hp))

    #diff = ripple_phase - lal_phase[int(duration*f_min):]
    
    amplitude_difference = abs(ripple_amp - np.abs(lal_amp[N:]))
    fig, ax = plt.subplots(3, 1, figsize=(10, 12))
    # Amplitude
    ax[0].plot(f, ripple_amp, label='ripple')
    ax[0].plot(lal_f, lal_amp, label='lalsim', linestyle='--')
    ax[0].plot(f, amplitude_difference, label = 'difference', color = 'black')
    ax[0].set_yscale('log')
    ax[0].set_xlim(15, 100)
    ax[0].set_ylabel('Amplitude')
    ax[0].legend()
    ax[0].set_title('Amplitude XPHM')

    # Phase
    ax[1].plot(f, ripple_phase, label='ripple')
    ax[1].plot(lal_f, lal_phase, label='lalsim', linestyle='--')
    phase_difference = abs(ripple_phase - lal_phase[int(minimum_frequency*duration):])
    ax[1].plot(f, phase_difference, label = 'phase difference', color = 'black')
    #ax[1].plot(f, diff, label = 'difference')
    #ax[1].set_yscale('log')
    ax[1].set_xlim(15, 100)
    ax[1].set_ylabel('Phase [rad]')
    ax[1].legend()
    ax[1].set_title('Phase XPHM')

    # Full waveform (real part)
    ax[2].plot(f, np.real(ripple_hp_xphm), label='ripple')
    ax[2].plot(lal_f, np.real(plot_xphm_hp), label='lalsim', linestyle='--')
    ax[2].set_xlim(15, 100)
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_ylabel('Real(h+)')
    ax[2].legend()
    ax[2].set_title('Full XPHM Waveform (Real)')
    ax[2].set_xlim(15, 80)

    plt.tight_layout()
    fig.savefig('xphm.pdf')

    


# Save each mode of hlm to separate files
modes_info = [(int(ell), int(m), i) for i, (ell, m) in enumerate(modes)]

save_ripple_waveforms = False
if save_ripple_waveforms:
    # Saving ripple waveforms
    for ell, m, col_idx in modes_info:
        mode_data = hlms_ripple_final[:, col_idx]
        ripple_amp = np.abs(mode_data)
        ripple_phase = np.angle(mode_data)

        # Save as: frequency, amplitude, phase
        output = np.column_stack([f, ripple_amp, ripple_phase])
        filename = f'ripple_hlm{ell}{m}.dat'
        np.savetxt(filename, output, header='frequency amplitude phase', fmt='%.3f %.5e %.5e')



fig, ax = plt.subplots(5, 4, figsize = (18, 11), sharex = True)
for ell, emm, col_idx in modes_info:

    phase_ripple = jnp.unwrap(jnp.angle(hlms_ripple_final[col_idx, :]))
    amp_ripple = jnp.abs(hlms_ripple_final[col_idx, :])

    hlms_lal = np.genfromtxt(f"lalsim_htildelm_{ell}{emm}.dat", skip_header=1)
    freq_lal = hlms_lal[:, 1]
    hlms_lal_complex = hlms_lal[:, 2] + 1j * hlms_lal[:, 3]

    # Select rows whose frequency matches the ripple frequency array
    mask = (freq_lal >= minimum_frequency) & (freq_lal < maximum_frequency)
    hlms_lal_masked = hlms_lal_complex[mask]
    phase_lal = jnp.unwrap(jnp.angle(hlms_lal_masked))
    amp_lal = jnp.abs(hlms_lal_masked)

    # Phase
    ax[col_idx, 0].plot(f, phase_lal, label = "LAL")
    ax[col_idx, 0].plot(f, phase_ripple, linestyle = "--", label = "Ripple")
    ax[col_idx, 0].set_title(f"Phase ({ell},{emm})")
    ax[col_idx, 0].legend()

    dphase = phase_ripple - phase_lal
    ax[col_idx, 1].plot(f, dphase)
    ax[col_idx, 1].set_title(r"$\Delta \phi$" + f" ({ell},{emm})")

    # Amplitude
    ax[col_idx, 2].plot(f, amp_lal, label = "LAL")
    ax[col_idx, 2].plot(f, amp_ripple, linestyle = "--", label = "Ripple")
    ax[col_idx, 2].set_title(f"Amplitude ({ell},{emm})")
    ax[col_idx, 2].legend()

    damp = amp_ripple - amp_lal
    ax[col_idx, 3].plot(f, damp)
    ax[col_idx, 3].set_title(r"$\Delta A$" + f" ({ell},{emm})")



fig.tight_layout()
fig.savefig("phase_difference.pdf")

   


# Twisting-up angles comparison: ripple vs lalsimulation
# lalsim files: Mf alpha epsilon cos_beta
# ripple files: Mf alpha epsilon beta

Mtot_SI = (injection_parameters['m1'][0] + injection_parameters['m2'][0]) * lal.MTSUN_SI

fig_ang, ax_ang = plt.subplots(5, 6, figsize=(24, 14), sharex=True)

for row, (ell, emm, col_idx) in enumerate(modes_info):
    lal_data    = np.loadtxt(f"lalsim_angles_{ell}{emm}.dat")
    ripple_data = np.loadtxt(f"ripple_angles_{emm}.dat", skiprows=1)

    # Convert Mf -> Hz
    f_lal    = lal_data[:, 0]    / Mtot_SI
    f_ripple = ripple_data[:, 0] / Mtot_SI

    alpha_lal,   epsilon_lal,   cosbeta_lal   = lal_data[:, 1],    lal_data[:, 2],    lal_data[:, 3]
    alpha_ripple, epsilon_ripple, beta_ripple  = ripple_data[:, 1], ripple_data[:, 2], ripple_data[:, 3]

    beta_lal = np.arccos(cosbeta_lal)

    for col, (y_lal, y_ripple, label) in enumerate([
        (alpha_lal,   alpha_ripple,   r"$\alpha$"),
        (epsilon_lal, epsilon_ripple, r"$\epsilon$"),
        (beta_lal,    beta_ripple,    r"$\beta$"),
    ]):
        ax_ang[row, 2*col].plot(f_lal,    y_lal,    label="LAL")
        ax_ang[row, 2*col].plot(f_ripple, y_ripple, linestyle="--", label="Ripple")
        ax_ang[row, 2*col].set_title(f"{label} ({ell},{emm})")
        ax_ang[row, 2*col].legend(fontsize=6)

        diff = y_lal[:(len(y_ripple))] - y_ripple[:len(y_ripple)]
        ax_ang[row, 2*col+1].plot(f_ripple, diff)
        ax_ang[row, 2*col+1].set_title(f"$\Delta${label} ({ell},{emm})")

ax_ang[-1, 0].set_xlabel("Frequency [Hz]")
fig_ang.tight_layout()
fig_ang.savefig("spin_angles_comparison.pdf")