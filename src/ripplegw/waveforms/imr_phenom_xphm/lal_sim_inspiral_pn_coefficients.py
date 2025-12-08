from __future__ import annotations

from ripplegw.constants import C, G


def xlal_sim_inspiral_taylor_t2_timing_0pncoeff(totalmass, eta):
    totalmass_sec = totalmass * G / C**3  # /* convert totalmass from kilograms to seconds */
    return -5.0 * totalmass_sec / (256.0 * eta)


def xlal_sim_inspiral_taylor_t2_timing_2pncoeff(eta):
    return 7.43 / 2.52 + 11.0 / 3.0 * eta


def xlal_sim_inspiral_taylor_t2_timing_4pncoeff(eta):
    return 30.58673 / 5.08032 + 54.29 / 5.04 * eta + 61.7 / 7.2 * eta * eta
