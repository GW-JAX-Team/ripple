"""Ground-based detector geocentric locations.

Values are the geocentric Cartesian vertex coordinates (in metres) of the
LAL ``LALDetector`` definitions (lal/lib/tools/LALDetectors.h). Only the
detector *position* is needed for the barycentered phase of a continuous-wave
signal; the arm orientation / response tensor is not required when returning
plus/cross polarizations (it enters only the antenna-pattern projection).
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Detector:
    """A detector's geocentric vertex location.

    Attributes:
        name (str): Short detector name (e.g. ``"H1"``).
        location (tuple[float, float, float]): Geocentric Cartesian
            coordinates of the vertex in metres.
    """

    name: str
    location: tuple[float, float, float]


# Geocentric vertex locations in metres, from LALDetectors.h.
DETECTORS: dict[str, Detector] = {
    "H1": Detector("H1", (-2.16141492636e06, -3.83469517889e06, 4.60035022664e06)),
    "L1": Detector("L1", (-7.42760447238e04, -5.49628371971e06, 3.22425701744e06)),
    "V1": Detector("V1", (4.54637409900e06, 8.42989697626e05, 4.37857696241e06)),
}


def get_detector(name: str) -> Detector:
    """Look up a detector by name.

    Args:
        name (str): Detector name (e.g. ``"H1"``, ``"L1"``, ``"V1"``).

    Returns:
        Detector: The matching detector definition.

    Raises:
        KeyError: If ``name`` is not a known detector.
    """
    try:
        return DETECTORS[name]
    except KeyError as exc:
        raise KeyError(
            f"Unknown detector '{name}'. Known detectors: {sorted(DETECTORS)}"
        ) from exc
