"""JAX implementation of LALSimIMRPhenomXHMQNM."""

from __future__ import annotations

from ripplegw.typing import Array


def evaluate_QNMfit_fring21(final_dimless_spin: float | Array) -> float | Array:
    """
    Docstring for evaluate_QNMfit_fring21

    :param final_dimless_spin: Description
    :type final_dimless_spin: float | Array
    :return: Description
    :rtype: float | Array
    """

    x2 = final_dimless_spin * final_dimless_spin
    x3 = x2 * final_dimless_spin
    x4 = x2 * x2
    x5 = x3 * x2

    return_val = (
        0.059471695665734674
        - 0.07585416297991414 * final_dimless_spin
        + 0.021967909664591865 * x2
        - 0.0018964744613388146 * x3
        + 0.001164879406179587 * x4
        - 0.0003387374454044957 * x5
    ) / (1 - 1.4437415542456158 * final_dimless_spin + 0.49246920313191234 * x2)
    return return_val
