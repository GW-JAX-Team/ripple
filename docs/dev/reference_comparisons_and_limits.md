# Reference Comparisons and Limits

This page records the external comparisons enforced for ripple waveforms.
It is the source of truth for validation methods, acceptance thresholds, and reference-specific limits.
For the complete test coverage of each waveform, start with [Test Coverage](test_coverage.md); for commands, see [Run Reference Checks](run_reference_checks.md).

Thresholds are acceptance criteria, not measurements from a particular run.
Keep them in sync with the tests that enforce them.

## Frequency-domain waveforms

Each listed waveform is compared with LAL on the same frequency grid.
The primary metric is the ET-D PSD-weighted overlap loss

$$
\mathrm{OL} = 1 - \frac{\operatorname{Re}\langle h_1 \mid h_2 \rangle}
{\bigl(\langle h_1 \mid h_1 \rangle \langle h_2 \mid h_2 \rangle\bigr)^{1/2}}.
$$

For precessing models, the value asserted against the threshold is the SNR-weighted
combination of the two polarizations' overlap loss (`combined_overlap_loss` in
`tests/helpers/metrics.py`), not the worse of the two raw per-polarization values:
judging the cross polarization in isolation is misleading near edge-on inclination,
where its own signal content vanishes and a fixed-size error is amplified into an
arbitrarily large *relative* loss despite carrying negligible detector SNR. The
per-polarization losses are still recorded (`overlap_loss_p`/`overlap_loss_c` in the
JSON results) as diagnostics. Non-precessing models also have a fixed-parameter
absolute phase check; precessing models do not, because their global phase is coupled
to the spin azimuths.
The overlap loss is a raw same-grid comparison at identical inputs; it is not maximised over time or phase.

| Waveform | Overlap-loss threshold | Phase-offset threshold |
| --- | ---: | ---: |
| TaylorF2 | 1e-15 | 1e-10 |
| IMRPhenomD | 1e-15 | 1e-11 |
| IMRPhenomD_NRTidalv2 | 1e-15 | 1e-9 |
| IMRPhenomHM | 1e-15 | 1e-11 |
| IMRPhenomPv2 | 1e-4 | — |
| IMRPhenomXAS | 1e-15 | 1e-12 |
| IMRPhenomXAS_NRTidalv3 | 1e-12 | 1e-7 |
| IMRPhenomXHM | 1e-6 | 1e-6 |
| IMRPhenomXP | 1e-6 | — |
| IMRPhenomXP_NRTidalv3 | 2e-6 | — |
| IMRPhenomXPHM | 1e-6 | — |

The overlap-loss values mirror `tests/cross_validation/tolerances.toml`.

### Reference limits

- **IMRPhenomPv2:** LAL and ripple use different procedures for the coalescence-time correction.
  The resulting comparison is dominated by a linear phase ramp rather than an amplitude discrepancy.
- **IMRPhenomXAS_NRTidalv3:** the non-machine-precision threshold remains under investigation.
- **IMRPhenomXHM:** the threshold covers sensitivity of the `(3, 2)` mode near ringdown, where spheroidal-to-spherical mixing makes the phase derivative numerically delicate.
- **IMRPhenomXP** and **IMRPhenomXPHM:** at BBH spins the MSA precession correction agrees with LAL at the float64 floor; the resonance mechanism below only matters at the rare parameter combinations the BBH prior samples close to it.
- **IMRPhenomXP_NRTidalv3:** the BNS prior sits close to the same MSA correction's conditioning problem far more often than the BBH prior does (a near-alignment degeneracy in the precession-averaged spin evolution, not a resonance), which the SNR-weighting above accounts for rather than removes. See [MSA Precession Instability](msa_precession_instability.md) for the derivation.

## Time-domain waveforms

Time-domain comparisons use aligned, directly generated samples.
Their normalized mismatch is

$$
m = 1 - \frac{h_1 \mathbin{\cdot} h_2}{\lVert h_1 \rVert\lVert h_2 \rVert}.
$$

It checks shape and phase but is unchanged by a global positive amplitude scale, so each large-scale time-domain comparison also has a relative-norm check.
No time-domain test uses an FFT to enter the frequency-domain harness.

| Waveform | Reference and scope | Acceptance threshold |
| --- | --- | --- |
| SineGaussian | LALSimulation; centered plus and cross samples | mismatch < 1e-17; relative norm error < 1e-15 |
| ExactPulsarSignal | Direct LALPulsar geometric building blocks and detector response | fixed regression: mismatch < 1e-10; delay error < 1e-9 s. Randomized sweep: mismatch < `1e-9 (f0 / 100 Hz)^2`; relative norm error < 1e-7 |
| PulsarSignal | Direct per-sample LALPulsar barycentering | mismatch < 1e-10 |
| PulsarSignal | `CWMakeFakeData` end to end | mismatch < `1e-3 (f0 / 100 Hz)^2`; relative norm error < 1e-1 |
| BinaryPulsarSignal | `XLALGenerateSpinOrbitCW` orbital source phase | mismatch < 1e-12 |
| BinaryPulsarSignal | `CWMakeFakeData` end to end | mismatch < `1e-2 (f0 / 100 Hz)^2`; relative norm error < 1e-1 |

`ExactPulsarSignal` is compared with matching geometric LAL building blocks because it intentionally omits the Einstein and Shapiro terms.
The full pulsar models are also checked through `CWMakeFakeData`, which exercises detector response, barycentering, and—where applicable—orbital modulation together.

The frequency-dependent `CWMakeFakeData` mismatch thresholds account for the reference pipeline's interpolated delay.
Its relative-norm threshold separately covers its interpolated antenna response; neither calls for matching those approximations in ripple.

## Maintaining validation

When a waveform, reference adapter, sampled domain, or comparison metric changes, update the enforcing test and this page together.
For frequency-domain models, also update `tests/cross_validation/tolerances.toml`.
Record the method and threshold; do not add run-specific outcomes here.
For implementation details behind the reference-specific limits, see the unlisted [reference-comparison notes](reference_comparison_notes.md).
