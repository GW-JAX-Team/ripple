# Reference-Comparison Notes

This unlisted page records implementation details behind reference-specific limits.
It intentionally excludes run outputs and historical calibration data.
For the enforcing thresholds, see [Reference Comparisons and Limits](reference_comparisons_and_limits.md).

## Frequency-domain comparisons

The frequency-domain overlap is a raw same-grid comparison at identical inputs.
It is not maximised over time or phase.

### Time origin and `tc`

LAL's frequency-domain models take no coalescence time.
Each places a model-specific feature at `t = 0`, and the choice differs between models: `IMRPhenomD` uses the exact phase derivative at the peak-amplitude frequency, the `IMRPhenomX` family a fitted peak-time shift (`linb`), the `IMRPhenomPv2` models a spline estimate at the ringdown or merger frequency, and `IMRPhenomNSBH` the merger-ringdown phase derivative at the requested upper frequency `f_max`.
ripple takes `f_max` from the model's `f_max` argument, defaulting to LAL's own default `Mf = 0.2`; the LAL comparison sets it to the grid's upper edge.
ripple reproduces each model's time origin, except for the two `IMRPhenomPv2` models described below.
The `tc` entry of the `gen_*` parameter arrays multiplies the strain by $e^{-2\pi i f\,t_c}$, which delays the signal by `tc` under LAL's Fourier convention $\tilde h(f) = \int h(t)\,e^{-2\pi i f t}\,dt$; the waveform classes fix `tc = 0`.

### IMRPhenomPv2

Both LAL and ripple place the ringdown frequency `f_RD` at `t = 0` using $t_0 = \frac{1}{2\pi}\,\frac{d\Phi}{df}$ at `f_RD`.
LAL estimates the derivative from a natural cubic spline through 10 phase samples on `[0.8, 1.2] f_RD` (clipped at `Mf = 0.2`); ripple evaluates it exactly with JAX autodiff.
The spline derivative converges to the autodiff value as its grid is refined, so the difference is LAL's interpolation error rather than a modelling choice.
The two waveforms differ by a parameter-dependent time shift of order microseconds, which appears as a linear phase ramp and accounts for the whole overlap loss: substituting LAL's spline time shift into ripple reproduces LAL at the float64 floor.
The shift is applied identically to both polarizations after the twist-up, so it is equivalent to a shift of the coalescence time.
At zero in-plane spin, both implementations remain continuous.
`IMRPhenomPv2` is nevertheless not required to reduce to `IMRPhenomD`: its LAL convention swaps the component masses and spins, changing their assignment in asymmetric phase terms.

### IMRPhenomPv2_NRTidalv2

NRTidalv2 is applied to the co-precessing IMRPhenomD waveform before the precession twist-up, as in `LALSimIMRPhenomP.c`.
Two LAL details must be reproduced: the quadrupole-monopole terms (`dQuadMon`) that LAL feeds into the IMRPhenomD inspiral phase (shared with `IMRPhenomD_NRTidalv2`), and the NRTidalv2 merger frequency computed with the mass ratio `>= 1`.
The time origin follows `IMRPhenomPv2` at the NRTidalv2 merger frequency: ripple takes the exact derivative of the co-precessing phase, tidal terms included, where LAL uses the same 10-point spline on `[0.8, 1.2] f_merger`.
The resulting microsecond-scale time shift is the only difference from LAL.

### IMRPhenomXHM

The phase of the `(3, 2)` mode is sensitive to spheroidal-to-spherical mixing near ringdown.
The intermediate phase fit is constrained by the phase derivative at the transition, so small differences there can shift the fitted phase.
Investigate a discrepancy in this region as a mode-mixing or phase-derivative issue rather than an amplitude discrepancy.

### IMRPhenomXP and IMRPhenomXPHM

The MSA precession correction is ill-conditioned near angular-momentum resonances, where terms involving `d0 + d2 + d4` suffer cancellation.
Small float64 differences in spin-evolution coefficients can then be amplified in the precession angles, particularly at high inclination.
`IMRPhenomXPHM` also inherits the `(3, 2)` mixing sensitivity through its XHM co-precessing seed.
The LAL reference explicitly requests `PhenomXPrecVersion=222`, so an MSA-initialization failure is surfaced rather than silently selecting a different precession approximation.
The BBH prior lands near this ill-conditioning rarely, so both models agree with LAL at the float64 floor.

### IMRPhenomXP_NRTidalv3

LAL builds this approximant through its `XLALSimIMRPhenomXPHM` code path, restricted to the `(2, ±2)` modes.
Multibanding is on by default there and must be disabled for the comparison (`PhenomXPHMThresholdMband = PhenomXHMThresholdMband = 0`, see `tests/cross_validation/reference/lal.py`); otherwise LAL disagrees with its own multibanding-off output for edge-on samples.
The twist cutoff matches LAL bin-for-bin: `Mf <= (fCutDef/M_sec)*M_sec`, inclusive, with `fCutDef` in `{0.3, 0.33}`.
The one corner case is `chiEff > 0.99`, where ripple's co-precessing amplitude is separately zeroed at `Mf = 0.3` (`fM_CUT` in `IMRPhenomXAS.py`) so `Mf` in `(0.3, 0.33]` stays zero while LAL keeps it; it is unreachable within the BNS test prior.
The dominant source of overlap loss is the MSA spin-evolution cubic's near-alignment ill-conditioning, which the BNS prior's low spins hit far more often than the BBH prior does.
See [`tests/cross_validation/msa_precession_instability.md`](https://github.com/GW-JAX-Team/ripple/blob/ripple-dev/tests/cross_validation/msa_precession_instability.md) for the mechanism and why it shows up as an edge-on-amplified `hc`-only error that the SNR-weighted combined metric absorbs.

## Continuous-wave comparisons

`ExactPulsarSignal` uses direct LALPulsar building blocks because `CWMakeFakeData` always includes the Einstein and Shapiro delays that the model intentionally omits.
`CWMakeFakeData` instead provides the end-to-end comparison for the full pulsar models, including detector response, barycentering, and orbital modulation where applicable.
That reference path interpolates propagation delays and detector response.
A delay error produces a phase error that grows with frequency, so its normalized mismatch has a frequency-squared bound.
Detector-response interpolation has a separate amplitude effect, which is covered by the relative-norm bound.
The bounds account for those reference approximations; ripple should not reproduce them.
