import lalsimulation as lalsim
import jax.numpy as jnp
import numpy as np
from ripplegw.constants import C, MSUN
import lal
import matplotlib.pyplot as plt
from ripplegw.waveforms import IMRPhenomXPHM
import bilby
from utils import GPSt_to_LMST

injection_parameters = {}
injection_parameters['m1'] = np.array([80.0])
injection_parameters['m2'] = np.array([14.0])

injection_parameters['m1_SI'] = injection_parameters['m1'] * MSUN
injection_parameters['m2_SI'] = injection_parameters['m2'] * MSUN


injection_parameters['Mc'] = bilby.gw.conversion.component_masses_to_chirp_mass(injection_parameters['m1'], 
                                                                                injection_parameters['m2'])

injection_parameters['distance'] = np.array([0.4]) # In GPc

injection_parameters['distance_SI'] = np.array([0.4 * 3.0856775814913673e25])
injection_parameters['theta'] = np.array([0.])

injection_parameters['phi'] = np.array([0.])

injection_parameters['iota'] = np.array([0.5])

injection_parameters['psi'] = np.array([0.])

injection_parameters['eta'] = injection_parameters['m1'] * injection_parameters['m2'] / (injection_parameters['m1'] + injection_parameters['m2'])**2

injection_parameters['Phicoal'] = np.array([1.2])

injection_parameters['chi1x'] = np.array([.1])
injection_parameters['chi1y'] = np.array([.2])
injection_parameters['chi1z'] = np.array([.3])

injection_parameters['chi2x'] = np.array([.3])
injection_parameters['chi2y'] = np.array([.2])
injection_parameters['chi2z'] = np.array([.1])

injection_parameters['phiRef'] = np.array([40.0])


minimum_frequency = 20
maximum_frequency = 1024
duration = 8.
df = 1/duration
reference_frequency = 40

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
if run_jim:

    model = IMRPhenomXPHM.IMRPhenomXPHM(fRef = reference_frequency)

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
    
    lal_f = np.arange(0., maximum_frequency, 1./duration)
    plot_xphm_hp = lal_hp_xphm.data.data[:-1]

    # Compute amplitude and phase
    ripple_amp = np.abs(hp_xphm)
    ripple_phase = np.unwrap(np.angle(hp_xphm))
    lal_amp = np.abs(plot_xphm_hp)
    lal_phase = np.unwrap(np.angle(plot_xphm_hp))

    #diff = ripple_phase - lal_phase[int(duration*f_min):]
    N = int(minimum_frequency * duration)
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
    fig.savefig('xphm22.pdf')

    

    # Convert all parameters to JAX arrays to avoid type mixing issues

    hlm = model.hphc(f = f,
                         Mc = injection_parameters['Mc'],
                         eta = injection_parameters['eta'],
                         dL = injection_parameters['distance'],
                         theta = injection_parameters['theta'],
                         phi = injection_parameters['phi'],
                         iota = injection_parameters['iota'],
                         tcoal = np.array([GPSt_to_LMST(tGPS, lat=0.,   long=0.)]),
                         Phicoal = injection_parameters['Phicoal'],
                         chi1x = injection_parameters['chi1x'],
                         chi1y = injection_parameters['chi1y'],
                         chi1z = injection_parameters['chi1z'],
                         chi2x = injection_parameters['chi2x'],
                         chi2y = injection_parameters['chi2y'],
                         chi2z = injection_parameters['chi2z'],
                         )

# plot hlm ################

# Define modes to plot: (ell, emmprime, hlm_column_index)
modes_to_plot = [
    (2, 1, 0),
    (2, 2, 1),
    (3, 2, 2),
    (3, 3, 3),
    (4, 4, 4)
]

# Create a figure with subplots for each mode
fig, axes = plt.subplots(len(modes_to_plot), 3, figsize=(14, 3*len(modes_to_plot)))
plot_diff = False
for i, (ell, emmprime, col_idx) in enumerate(modes_to_plot):
    # Load lalsim data
    filename = f'htildelm_{ell}{emmprime}.dat'
    try:
        lalsim_data = np.loadtxt(filename)
        lalsim_freq = lalsim_data[:, 0]
        
        lalsim_amp = lalsim_data[:, 1]
        lalsim_phase = np.unwrap(lalsim_data[:, 2])

        lalsim_real = lalsim_amp * np.cos(lalsim_phase)
        #lalsim_imag = lalsim_data[:, 2]
        #lalsim_complex = lalsim_real + 1j * lalsim_imag

        # Plot amplitude
        axes[i, 0].plot(f, np.abs(hlm[:, col_idx]), label=f'ripple ({ell},{emmprime})', linewidth=2)
        axes[i, 0].plot(lalsim_freq, lalsim_amp, label=f'lalsim ({ell},{emmprime})', linestyle='--', linewidth=2)
        # Add rb_h for (2,2) mode
        N = int(minimum_frequency*duration)
        if plot_diff:
            diff = abs(np.abs(hlm[:, col_idx]) - lalsim_amp[N:-1])
            axes[i, 0].plot(f, diff, label = "difference")
        axes[i, 0].set_yscale('log')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].set_xlabel('Frequency (Hz)')
        axes[i, 0].legend()
        axes[i, 0].grid(True)
        axes[i, 0].set_title(f'Mode ({ell},{emmprime}) - Amplitude')

        # Plot phase
        axes[i, 1].plot(f, np.unwrap(np.angle(hlm[:, col_idx])), label=f'ripple ({ell},{emmprime})', linewidth=2)
        axes[i, 1].plot(lalsim_freq, lalsim_phase, label=f'lalsim ({ell},{emmprime})', linestyle='--', linewidth=2)
        # Add rb_h for (2,2) mode
        if plot_diff:
            diff = abs(np.unwrap(np.angle(hlm[:, col_idx])) - lalsim_phase[N:-1])
            axes[i, 1].plot(f, diff, label = "difference")
        axes[i, 1].set_ylabel('Phase (rad)')
        axes[i, 1].set_xlabel('Frequency (Hz)')
        axes[i, 1].legend()
        axes[i, 1].grid(True)
        #axes[i, 1].set_yscale('log')
        axes[i, 1].set_title(f'Mode ({ell},{emmprime}) - Phase')

        axes[i, 2].plot(f, np.real(hlm[:, col_idx]), label = 'ripple')
        axes[i, 2].plot(lalsim_freq, lalsim_real, label = 'lalsim', ls = '--')
        axes[i, 2].set_xlim(15, 90)
        axes[i, 2].legend()

    except FileNotFoundError:
        print(f"Warning: {filename} not found, skipping mode ({ell},{emmprime})")
        axes[i, 0].text(0.5, 0.5, f'{filename} not found', ha='center', va='center')
        axes[i, 1].text(0.5, 0.5, f'{filename} not found', ha='center', va='center')

plt.tight_layout()
fig.savefig('modes_all.pdf')

exit()

