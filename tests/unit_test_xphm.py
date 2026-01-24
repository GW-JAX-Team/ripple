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

def compute_overlap(frequency_series_1, frequency_series_2, df):

    norm1 = np.sum(frequency_series_1*np.conj(frequency_series_1))**0.5
    norm2 = np.sum(frequency_series_2*np.conj(frequency_series_2))**0.5

    inner_product = np.sum(frequency_series_1*np.conj(frequency_series_2))
    return inner_product / (norm1*norm2)

injection_parameters = {}
injection_parameters['m1'] = np.array([36.0])
injection_parameters['m2'] = np.array([29.0])

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

injection_parameters['Phicoal'] = np.array([0.5])

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

f = np.arange(minimum_frequency, maximum_frequency, df)
lalparams = lal.CreateDict()

ModeArray = lalsim.SimInspiralCreateModeArray()

lalsim.SimInspiralModeArrayActivateMode(ModeArray, 2, 1)
lalsim.SimInspiralModeArrayActivateMode(ModeArray, 2, 2)

lalsim.SimInspiralModeArrayActivateMode(ModeArray, 3, 2)
lalsim.SimInspiralModeArrayActivateMode(ModeArray, 3, 3)

#lalsim.SimInspiralModeArrayActivateMode(ModeArray, 4, 3)
lalsim.SimInspiralModeArrayActivateMode(ModeArray, 4, 4)




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
run_jim = True
tGPS = 3600
model = IMRPhenomXPHM.IMRPhenomXPHM(apply_fcut = True, reference_frequency=reference_frequency)

if run_jim:


    hp_xphm, hc_xphm = model.generate_xphm(injection_parameters['m1'][0],
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
                                            minimum_frequency, maximum_frequency, reference_frequency)


    N = int(minimum_frequency * duration)

    plus_overlap = compute_overlap(hp_xphm, np.array(lal_hp_xphm.data.data[N:-1]), df = 1/duration)
    print("Plus overlap", plus_overlap)



    lal_f = np.arange(0., maximum_frequency, 1./duration)
    plot_xphm_hp = lal_hp_xphm.data.data[:-1]




    # Compute amplitude and phase
    ripple_amp = np.abs(hp_xphm)
    ripple_phase = np.unwrap(np.angle(hp_xphm))
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
    ax[2].plot(f, np.real(hp_xphm), label='ripple')
    ax[2].plot(lal_f, np.real(plot_xphm_hp), label='lalsim', linestyle='--')
    ax[2].set_xlim(15, 100)
    ax[2].set_xlabel('Frequency [Hz]')
    ax[2].set_ylabel('Real(h+)')
    ax[2].legend()
    ax[2].set_title('Full XPHM Waveform (Real)')
    ax[2].set_xlim(15, 80)

    plt.tight_layout()
    fig.savefig('xphm.pdf')

    

# Convert all parameters to JAX arrays to avoid type mixing issues
print('\n Calling the hphc directly')
hlm = IMRPhenomXPHM.hphc(frequency_array = jnp.arange(minimum_frequency, maximum_frequency, df),
                        chirp_mass = injection_parameters['chirp_mass'][0],
                        eta = injection_parameters['eta'][0],
                        chi1x = injection_parameters['chi1x'][0],
                        chi1y = injection_parameters['chi1y'][0],
                        chi1z = injection_parameters['chi1z'][0],
                        chi2x = injection_parameters['chi2x'][0],
                        chi2y = injection_parameters['chi2y'][0],
                        chi2z = injection_parameters['chi2z'][0],
                        luminosity_distance= injection_parameters['distance'][0],
                        iota = injection_parameters['iota'][0],
                        initial_phase= injection_parameters['Phicoal'][0],
                        reference_frequency = reference_frequency
                        )

# Save each mode of hlm to separate files
modes_info = [
    (2, 1, 0),  # (ell, m, column_index)
    (2, 2, 1),
    (3, 2, 2),
    (3, 3, 3),
    (4, 4, 4)
]

for ell, m, col_idx in modes_info:
    mode_data = hlm[:, col_idx]
    amp = np.abs(mode_data)
    phase = np.angle(mode_data)

    # Save as: frequency, amplitude, phase
    output = np.column_stack([f, amp, phase])
    filename = f'ripple_hlm{ell}{m}.dat'
    np.savetxt(filename, output, header='frequency amplitude phase', fmt='%.3f %.5e %.5e')
    #print(f"Saved {filename}")



ripple_phase = np.genfromtxt("./PhisAllModes_ripple.dat", skip_header=True)

ripple_phase[:, 1] += np.pi/2
ripple_phase[:, 4] -= np.pi/2
ripple_phase[:, 5] += np.pi

fig, ax = plt.subplots(5, 2, figsize = (6, 8), sharex = True)
for ell, emm, column_index in modes_info:
    lalsim_phase = np.genfromtxt(f"./lalsim_phases_{ell}{emm}.dat", skip_header=True)
    si_frequency = lalsim_phase[:, 1]

    ripple_phase_mode = ripple_phase[:, column_index+1][:len(si_frequency)] 


    ax[column_index, 0].plot(si_frequency, lalsim_phase[:, 2], label = 'lalsim')
    ax[column_index, 0].plot(si_frequency, ripple_phase_mode, label = 'ripple', ls = '--')
    ax[column_index, 0].legend()
    ax[column_index, 0].grid()


    diff = lalsim_phase[:, 2] - ripple_phase_mode
    ax[column_index, 1].plot(si_frequency, diff, label = 'difference')
    #ax[column_index, 1].set_yscale('symlog')
    #ax[column_index, 1].legend()
    ax[column_index, 1].grid()
    ax[column_index, 1].set_ylim(-0.05, 0.06)
    ax[column_index, 1].set_title(f"mode {ell} {emm}")


    for _ax in ax.flatten():
        _ax.axvline(x = 56.2224314, color = 'black', ls ='--')
        _ax.axvline(x = 146.12671021, color = 'black', ls ='--')
        _ax.axvline(x = 624.69368224, color = 'black', ls ='--')
        _ax.axvline(x = 20, color = 'grey', ls = ':')

fig.tight_layout()
fig.savefig(f'mode_all_phase.pdf')