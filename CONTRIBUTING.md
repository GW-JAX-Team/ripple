# Contributing

Contributions of any kind are welcome and appreciated.
See the guidelines below.

## Expectations

ripple is developed and maintained by the GW JAX Team and community contributors.
While we try to be responsive, we don’t always get to every issue immediately.
If it has been more than a week or two, feel free to ping the maintainers on the issue.

## Did you find a bug?

**Ensure the bug was not already reported** by searching on GitHub under [Issues](https://github.com/GW-JAX-Team/ripple/issues).
If you’re unable to find an open issue addressing the problem, [open a new one](https://github.com/GW-JAX-Team/ripple/issues/new).
Be sure to include a **title and clear description**, as much relevant information as possible, and the simplest possible **code sample** demonstrating the expected behaviour that is not occurring.

## Did you write a patch that fixes a bug?

Open a new GitHub pull request with the patch.
Ensure the PR description clearly describes the problem and solution.
Include the relevant issue number if applicable.

## Do you intend to add a new feature or change an existing one?

Open a new GitHub pull request with the feature or change.
Please follow these principles:

1. New features should be able to take advantage of `jax.jit` wherever possible.
2. Modular implementation is preferred.
3. If a waveform is ported from an existing implementation, it should match the original to machine precision.
   If that is not achievable, a clear explanation of the discrepancy must be provided.
   New waveforms implemented directly in ripple are not subject to this requirement.

If you are unsure whether a feature fits, open an issue first to discuss it with the maintainers.

Adding a new waveform is the most common feature contribution — see the [Developer Guide](https://gw-jax-team.github.io/ripple/latest/dev/) for the architecture and a step-by-step walkthrough, including how to distribute a waveform as a separate plugin package instead of editing ripple directly.

## Do you intend to add an example or tutorial?

Open a new GitHub pull request with the example or tutorial.
The example should be self-contained and keep imports from other packages to a minimum.
Leave case-specific analysis details out.
