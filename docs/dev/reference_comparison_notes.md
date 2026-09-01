# Reference-Comparison Notes

This unlisted page records implementation details behind reference-specific limits.
It intentionally excludes run outputs and historical calibration data.
For the enforcing thresholds, see [Reference Comparisons and Limits](reference_comparisons_and_limits.md).

## Frequency-domain comparisons

The frequency-domain overlap is a raw same-grid comparison at identical inputs.
It is not maximised over time or phase.

### LAL multibanding is disabled on the reference side

LAL's `PhenomXHMThresholdMband` defaults to `1e-3`: rather than evaluating the model on every frequency bin, LAL evaluates a coarse grid and interpolates.
That is a speed optimisation, not part of the waveform model, and it costs roughly `1e-3` in relative amplitude and `1e-4` rad in phase, concentrated in the ringdown where the coarse grid is least adequate.
Left on it puts a floor of about `1e-10` on every `IMRPhenomXHM` and `IMRPhenomXPHM` mode's overlap loss, regardless of how correct ripple is, and that floor was for a time mistaken for a defect in ripple's higher modes.
Both the mode-summed backend and the per-mode test therefore set the threshold to zero.

`IMRPhenomXPHM` needs the `PhenomXHMThresholdMband` flag specifically.
Its own `PhenomXPHMThresholdMband` and `PhenomXPHMMBandVersion` flags govern only the Euler-angle grid, while the co-precessing modes are generated through the XHM path and read the XHM threshold.
`IMRPhenomHM` has no multibanding, which is why it always sat at round-off.

### IMRPhenomPv2

LAL estimates the coalescence-time correction from the derivative of a natural-cubic spline through a small phase grid around ringdown.
ripple computes the corresponding derivative directly with JAX autodiff.
The resulting time correction can differ near the merger-ringdown feature, appearing as a linear phase ramp rather than an amplitude error.
At zero in-plane spin, both implementations remain continuous.
`IMRPhenomPv2` is nevertheless not required to reduce to `IMRPhenomD`: its LAL convention swaps the component masses and spins, changing their assignment in asymmetric phase terms.

### IMRPhenomXHM

With multibanding disabled, every mode except `(3, 2)` agrees with LAL at round-off.
`(3, 2)` is the only remaining source of error, so a per-mode discrepancy anywhere else should be treated as a regression rather than an expected limit.

The phase of the `(3, 2)` mode is sensitive to spheroidal-to-spherical mixing near ringdown.
The intermediate phase fit is constrained by the phase derivative at the transition, so small differences there can shift the fitted phase.
Investigate a discrepancy in this region as a mode-mixing or phase-derivative issue rather than an amplitude discrepancy.

Two conventions in LAL's higher-mode amplitude are easy to miss and both showed up as small isolated errors:

- LAL clamps a negative reconstructed amplitude to `FALSE_ZERO = 1e-15` at the end of every `IMRPhenomXHM_Amplitude_*` function.
  The `(2, 1)` amplitude has a genuine minimum in the late inspiral where the collocation polynomial can undershoot; without the clamp those bins get a spurious `pi` phase flip instead of passing through zero.
- LAL truncates every mode at `fCut = 0.3` in geometric frequency.
  ripple's `IMRPhenomXAS` amplitude has always applied this, but the higher-mode path did not, which showed up in the `(4, 4)` mode of heavy binaries — the only case where `Mf = 0.3` falls inside the analysis band while that mode's ringdown has not yet decayed away.

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
