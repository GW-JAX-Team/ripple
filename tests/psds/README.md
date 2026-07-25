# Power Spectral Density (PSD) Files

This directory contains noise PSD files used for testing ripple waveforms.

## Source

All PSDs are sourced from the bilby gravitational wave inference library:

**Repository:** https://github.com/bilby-dev/bilby  
**Commit:** `0985f75c664786e21cc4f662d4f12fe181b1a536`  
**Date:** 2026-02-25  
**Source Path:** `bilby/gw/detector/noise_curves/`

## Files

### ET_D_psd.txt
Einstein Telescope D-design PSD.
Used as the default test PSD for cross-validation and benchmarking.

**Format:** Two-column ASCII text (frequency [Hz], PSD [Hz^-1])

## Reference

For more information about these noise curves, see:
- bilby documentation: https://lscsoft.docs.ligo.org/bilby/
- Original bilby PSD README: https://github.com/bilby-dev/bilby/blob/main/bilby/gw/detector/noise_curves/README.md
