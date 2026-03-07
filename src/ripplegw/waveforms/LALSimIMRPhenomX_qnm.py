def evaluate_QNMfit_fring22(finalDimlessSpin):
    """
    Evaluate the QNM fit for the ringdown frequency of the (2,2) mode.

    This is a rational polynomial fit for the ringdown frequency as a function
    of the final dimensionless spin.

    Parameters
    ----------
    finalDimlessSpin : float or jax.numpy.ndarray
        Final dimensionless spin parameter (must satisfy |finalDimlessSpin| <= 1.0)

    Returns
    -------
    float or jax.numpy.ndarray
        Fitted ringdown frequency for the (2,2) mode

    Notes
    -----
    The function raises a warning if |finalDimlessSpin| > 1.0, but continues
    evaluation in the JAX implementation (for differentiability).

    Original C implementation from LALSimIMRPhenomX.
    """

    # Compute powers of finalDimlessSpin
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    x7 = x4 * x3

    # Numerator coefficients
    numerator = (
        0.05947169566573468
        - 0.14989771215394762 * finalDimlessSpin
        + 0.09535606290986028 * x2
        + 0.02260924869042963 * x3
        - 0.02501704155363241 * x4
        - 0.005852438240997211 * x5
        + 0.0027489038393367993 * x6
        + 0.0005821983163192694 * x7
    )

    # Denominator coefficients
    denominator = (
        1
        - 2.8570126619966296 * finalDimlessSpin
        + 2.373335413978394 * x2
        - 0.6036964688511505 * x4
        + 0.0873798215084077 * x6
    )

    return_val = numerator / denominator

    return return_val


def evaluate_QNMfit_fdamp22(finalDimlessSpin):
    """
    Evaluate the QNM fit for the damping frequency of the (2,2) mode.

    This is a rational polynomial fit for the damping frequency as a function
    of the final dimensionless spin.

    Parameters
    ----------
    finalDimlessSpin : float or jax.numpy.ndarray
        Final dimensionless spin parameter (must satisfy |finalDimlessSpin| <= 1.0)

    Returns
    -------
    float or jax.numpy.ndarray
        Fitted damping frequency for the (2,2) mode

    Notes
    -----
    The function raises a warning if |finalDimlessSpin| > 1.0, but continues
    evaluation in the JAX implementation (for differentiability).

    Original C implementation from LALSimIMRPhenomX.
    """

    # Compute powers of finalDimlessSpin
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    # Numerator coefficients
    numerator = (
        0.014158792290965177
        - 0.036989395871554566 * finalDimlessSpin
        + 0.026822526296575368 * x2
        + 0.0008490933750566702 * x3
        - 0.004843996907020524 * x4
        - 0.00014745235759327472 * x5
        + 0.0001504546201236794 * x6
    )

    # Denominator coefficients
    denominator = (
        1
        - 2.5900842798681376 * finalDimlessSpin
        + 1.8952576220623967 * x2
        - 0.31416610693042507 * x4
        + 0.009002719412204133 * x6
    )

    return_val = numerator / denominator

    return return_val
