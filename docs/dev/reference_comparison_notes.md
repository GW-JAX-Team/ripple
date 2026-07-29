# Reference-Comparison Notes

This unlisted page records implementation details behind reference-specific limits.
It intentionally excludes run outputs and historical calibration data.
For the enforcing thresholds, see [Reference Comparisons and Limits](reference_comparisons_and_limits.md).

## Frequency-domain comparisons

The frequency-domain overlap is a raw same-grid comparison at identical inputs.
It is not maximised over time or phase.

### IMRPhenomPv2

LAL estimates the coalescence-time correction from the derivative of a natural-cubic spline through a small phase grid around ringdown.
ripple computes the corresponding derivative directly with JAX autodiff.
The resulting time correction can differ near the merger-ringdown feature, appearing as a linear phase ramp rather than an amplitude error.
At zero in-plane spin, both implementations remain continuous.
`IMRPhenomPv2` is nevertheless not required to reduce to `IMRPhenomD`: its LAL convention swaps the component masses and spins, changing their assignment in asymmetric phase terms.

### IMRPhenomXHM

The phase of the `(3, 2)` mode is sensitive to spheroidal-to-spherical mixing near ringdown.
The intermediate phase fit is constrained by the phase derivative at the transition, so small differences there can shift the fitted phase.
Investigate a discrepancy in this region as a mode-mixing or phase-derivative issue rather than an amplitude discrepancy.

### IMRPhenomXP and IMRPhenomXPHM

The MSA precession correction is ill-conditioned near angular-momentum resonances, where terms involving `d0 + d2 + d4` suffer cancellation.
Small float64 differences in spin-evolution coefficients can then be amplified in the precession angles, particularly at high inclination.
`IMRPhenomXPHM` also inherits the `(3, 2)` mixing sensitivity through its XHM co-precessing seed.
The LAL reference explicitly requests `PhenomXPrecVersion=222`, so an MSA-initialization failure is surfaced rather than silently selecting a different precession approximation.

## Continuous-wave comparisons

`ExactPulsarSignal` uses direct LALPulsar building blocks because `CWMakeFakeData` always includes the Einstein and Shapiro delays that the model intentionally omits.
`CWMakeFakeData` instead provides the end-to-end comparison for the full pulsar models, including detector response, barycentering, and orbital modulation where applicable.
That reference path interpolates propagation delays and detector response.
A delay error produces a phase error that grows with frequency, so its normalized mismatch has a frequency-squared bound.
Detector-response interpolation has a separate amplitude effect, which is covered by the relative-norm bound.
The bounds account for those reference approximations; ripple should not reproduce them.
