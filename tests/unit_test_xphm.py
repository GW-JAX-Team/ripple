import lalsimulation as lalsim
import jax
import jax.numpy as jnp
import numpy as np
from ripplegw.constants import MSUN
import lal
import matplotlib.pyplot as plt
from ripplegw.waveforms import IMRPhenomXPHM
from utils import GPSt_to_LMST
print("Device", jax.devices())

def compute_overlap(frequency_series_1, frequency_series_2):

    normass_1 = np.sum(frequency_series_1*np.conj(frequency_series_1))**0.5
    normass_2 = np.sum(frequency_series_2*np.conj(frequency_series_2))**0.5

    inner_product = np.sum(frequency_series_1*np.conj(frequency_series_2))
    return inner_product / (normass_1*normass_2)

injection_parameters = {}
injection_parameters['mass_1'] = np.array([36.0])
injection_parameters['mass_2'] = np.array([9.0])

injection_parameters['mass_1_SI'] = injection_parameters['mass_1'] * MSUN
injection_parameters['mass_2_SI'] = injection_parameters['mass_2'] * MSUN
injection_parameters['distance'] = np.array([1]) # In Mpc
injection_parameters['distance_SI'] = np.array([1 * 3.0856775814913673e22]) # In meters
injection_parameters['theta'] = np.array([0.5])
injection_parameters['phi'] = np.array([0.])
injection_parameters['iota'] = np.array([0.2])
injection_parameters['psi'] = np.array([1.2])
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
reference_frequency = 50
modes = jnp.array([[2,1],[2,2],[3,2],[3,3],[4,4]])

f = np.arange(minimum_frequency, maximum_frequency, 1/duration)
lalparams = lal.CreateDict()

ModeArray = lalsim.SimInspiralCreateModeArray()

for mm in modes:
    lalsim.SimInspiralModeArrayActivateMode(ModeArray, int(mm[0]), int(mm[1]))

lalsim.SimInspiralWaveformParamsInsertModeArray(lalparams, ModeArray)
lalsim.SimInspiralWaveformParamsInsertPhenomXPHMTwistPhenomHM(lalparams, 1)
lalsim.SimInspiralWaveformParamsInsertPhenomXPHMMBandVersion(lalparams, 0)
lalsim.SimInspiralWaveformParamsInsertPhenomXPHMThresholdMband(lalparams, 0.0)
lalsim.SimInspiralWaveformParamsInsertPhenomXPrecVersion(lalparams, 223)

lal_hp_xphm, lal_hc_xphm = lalsim.SimIMRPhenomXPHM(injection_parameters['mass_1_SI'][0],                       
                                               injection_parameters['mass_2_SI'][0],                    
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
                                               1/duration,                       #/**< Sampling frequency (Hz). To use non-uniform frequency grid, set deltaF <= 0. */
                                               reference_frequency,                      #/**< Reference frequency (Hz) */
                                               lalparams                  #/**< LAL Dictionary struct */
                                               )

###### jax code
tGPS = 3600
#model = IMRPhenomXPHM.IMRPhenomXPHM(apply_fcut = True, reference_frequency=reference_frequency)

run_jim_xphm = True

if run_jim_xphm:
    frequency_array = jnp.arange(minimum_frequency, maximum_frequency, 1/duration)

    ripple_hp_xphm, ripple_hc_xphm = IMRPhenomXPHM.generate_xphm(injection_parameters['mass_1'][0],
                                           injection_parameters['mass_2'][0],
                                            injection_parameters['chi1x'][0],
                                            injection_parameters['chi1y'][0],
                                            injection_parameters['chi1z'][0],
                                            injection_parameters['chi2x'][0],
                                            injection_parameters['chi2y'][0],
                                            injection_parameters['chi2z'][0],
                                            injection_parameters['distance'][0],
                                            injection_parameters['iota'][0],
                                            injection_parameters['Phicoal'][0],
                                            frequency_array,
                                            reference_frequency)

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

    
exit()