# Waveform Catalogue

This catalogue lists the waveform models available in ripple and the physical assumptions that distinguish them. The model names below are the names used by ripple.

## Terminology

- **Aligned-spin** models assume that both component spins are parallel or antiparallel to the orbital angular momentum.
- **Precessing-spin** models allow spin directions that cause the orbital plane to precess.
- **Higher-order modes** include radiation beyond the dominant quadrupole mode; they can be important for unequal-mass binaries and for systems viewed away from face-on.
- **Tidal effects** describe the deformation of neutron stars by their companions.

## Compact-binary coalescences

These models describe radiation from two compact objects in a quasi-circular orbit. All compact-binary-coalescence waveforms in ripple are provided in the frequency domain at the moment.

### TaylorF2

`TaylorF2` is a post-Newtonian, aligned-spin waveform with tidal effects. It models the inspiral only and does not include merger or ringdown.

### Earlier IMRPhenom models

| Model | Spin treatment | Higher-order modes | Tidal effects |
| --- | --- | --- | --- |
| `IMRPhenomD` | Aligned-spin | No | No |
| `IMRPhenomHM` | Aligned-spin | Yes | No |
| `IMRPhenomPv2` | Precessing | No | No |
| `IMRPhenomD_NRTidalv2` | Aligned-spin | No | NRTidalv2 |

### IMRPhenomX family

The IMRPhenomX models are the newer IMRPhenom models.

| Model | Spin treatment | Higher-order modes | Tidal effects |
| --- | --- | --- | --- |
| `IMRPhenomXAS` | Aligned-spin | No | No |
| `IMRPhenomXHM` | Aligned-spin | Yes | No |
| `IMRPhenomXP` | Precessing | No | No |
| `IMRPhenomXPHM` | Precessing | Yes | No |
| `IMRPhenomXAS_NRTidalv3` | Aligned-spin | No | NRTidalv3 |

## Bursts

`SineGaussian` is a short-duration burst model that describes a sine-Gaussian waveform in the time domain.

## Continuous waves

These models describe long-lived, nearly periodic signals from pulsars.

| Model | Source | Binary orbit | Timing treatment |
| --- | --- | --- | --- |
| `ExactPulsarSignal` | Isolated pulsar | No | Geometric barycentric corrections |
| `PulsarSignal` | Isolated pulsar | No | Full barycentric corrections |
| `BinaryPulsarSignal` | Pulsar in a binary | Yes | Full barycentric corrections and orbital modulation |
