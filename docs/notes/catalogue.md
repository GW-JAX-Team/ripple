# Waveform Catalogue

This catalogue lists the waveform models available in ripple and the physical assumptions that distinguish them.
The model names below are the names used by ripple.

## Terminology

- **Aligned-spin** models assume that both component spins are parallel or antiparallel to the orbital angular momentum.
- **Precessing-spin** models allow spin directions that cause the orbital plane to precess.
- **Higher-order modes** include radiation beyond the dominant quadrupole mode; they can be important for unequal-mass binaries and for systems viewed away from face-on.
- **Tidal effects** describe the deformation of neutron stars by their companions.

## Compact-binary coalescences

These models describe radiation from two compact objects in a quasi-circular orbit.
All compact-binary-coalescence waveforms in ripple are provided in the frequency domain at the moment.

### TaylorF2

`TaylorF2` is a post-Newtonian, aligned-spin waveform with tidal effects.
It models the inspiral only and does not include merger or ringdown.

### Earlier IMRPhenom models

| Waveform | Spin treatment | Higher-order modes | Tidal effects |
| --- | --- | --- | --- |
| `IMRPhenomD` | Aligned-spin | No | No |
| `IMRPhenomHM` | Aligned-spin | Yes | No |
| `IMRPhenomPv2` | Precessing | No | No |
| `IMRPhenomD_NRTidalv2` | Aligned-spin | No | NRTidalv2 |
| `IMRPhenomPv2_NRTidalv2` | Precessing | No | NRTidalv2 |
| `IMRPhenomNSBH` | Aligned-spin | No | NRTidal phase, NS only |

### IMRPhenomX family

The IMRPhenomX models are the newer IMRPhenom models.

| Waveform | Spin treatment | Higher-order modes | Tidal effects |
| --- | --- | --- | --- |
| `IMRPhenomXAS` | Aligned-spin | No | No |
| `IMRPhenomXHM` | Aligned-spin | Yes | No |
| `IMRPhenomXP` | Precessing | No | No |
| `IMRPhenomXPHM` | Precessing | Yes | No |
| `IMRPhenomXAS_NRTidalv3` | Aligned-spin | No | NRTidalv3 |
| `IMRPhenomXP_NRTidalv3` | Precessing | No | NRTidalv3 |

### Validity ranges

ripple ports the LAL implementations but does not enforce their parameter checks.
Outside these ranges the models extrapolate silently.
The table lists what LAL rejects and where it warns about, or documents, the edge of the calibration, with $q = m_1/m_2 \ge 1$.

| Waveform | LAL rejects | Outside the calibration |
| --- | --- | --- |
| `TaylorF2` | — | No range checks |
| `IMRPhenomD` | $\lvert\chi_i\rvert > 1$ | $q > 18$ |
| `IMRPhenomHM` | $\lvert\chi_i\rvert > 1$ | $q > 20$ |
| `IMRPhenomPv2` | $\lvert\vec\chi_i\rvert > 1$ | $q > 18$; strongly anti-aligned spins, where the precession opening angle is no longer tracked |
| `IMRPhenomD_NRTidalv2` | $\lvert\chi_i\rvert > 1$, $\Lambda_i < 0$ | $q > 18$; tides beyond $q = 1.5$ or $\kappa_2^T \notin [40, 5000]$ |
| `IMRPhenomPv2_NRTidalv2` | $\lvert\vec\chi_i\rvert > 1$, $\Lambda_i < 0$ | As `IMRPhenomPv2`, with the tides of `IMRPhenomD_NRTidalv2`; $q < 1.5$ with both $\Lambda_i > 2000$ |
| `IMRPhenomNSBH` | $m_\mathrm{BH} < m_\mathrm{NS}$, $q > 100$, $\Lambda_\mathrm{NS} > 5000$, $m_\mathrm{NS} > 3\,M_\odot$; a negative inspiral frequency derivative | $m_\mathrm{NS} < 1\,M_\odot$, non-zero NS spin |
| `IMRPhenomXAS`, `IMRPhenomXHM` | $q > 1000$, $\lvert\chi_i\rvert > 1$ | $q > 20$, $\lvert\chi_i\rvert > 0.99$ |
| `IMRPhenomXP`, `IMRPhenomXPHM` | $q > 1000$, $\lvert\vec\chi_i\rvert > 1$ | $q > 20$, $\lvert\vec\chi_i\rvert > 0.99$ |
| `IMRPhenomXAS_NRTidalv3`, `IMRPhenomXP_NRTidalv3` | As the BBH model, and $\Lambda_i < 0$ | As the BBH model; tides beyond $q = 2$ or $\kappa_2^T \notin [40, 5000]$ |

LAL applies the `IMRPhenomNSBH` limit $m_\mathrm{NS} \le 3\,M_\odot$ to the detector-frame mass, so it also rejects ordinary systems at redshift $z \gtrsim 1.15$; ripple's model has no such limit.

## Bursts

`SineGaussian` is a short-duration burst model that describes a sine-Gaussian waveform in the time domain.

## Continuous waves

These models describe long-lived, nearly periodic signals from pulsars.

| Waveform | Source | Timing treatment |
| --- | --- | --- |
| `ExactPulsarSignal` | Isolated pulsar | Geometric barycentric corrections |
| `PulsarSignal` | Isolated pulsar | Full barycentric corrections |
| `BinaryPulsarSignal` | Pulsar in a binary | Full barycentric corrections and orbital modulation |
