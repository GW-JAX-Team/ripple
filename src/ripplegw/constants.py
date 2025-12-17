"""
Various constants, all in SI units.
"""

EulerGamma = 0.577215664901532860606512090082402431

MSUN = 1.9884098706980507e30  # kg (LAL_MSUN_SI)
"""Solar mass"""

MRSUN = 1476.6250380501247  # m (LAL_MRSUN_SI)
"""Geometrized nominal solar mass, m"""

G = 6.6743e-11  # m^3 / kg / s^2 (LAL_G_SI)
"""Newton's gravitational constant"""

C = 299792458.0  # m / s
"""Speed of light"""

"""Pi"""
PI = 3.141592653589793238462643383279502884

TWO_PI = 6.283185307179586476925286766559005768

gt = G * MSUN / (C**3.0)
"""
G MSUN / C^3 in seconds
"""

m_per_Mpc = 3.085677581491367278913937957796471611e22
"""
Meters per Mpc.
"""

clightGpc = C / 3.0856778570831e22
"""
Speed of light in vacuum (:math:`c`), in gigaparsecs per second
"""
