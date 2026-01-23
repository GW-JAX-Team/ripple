import jax.numpy as jnp

def evaluate_QNMfit_fring21(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring21
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    # (In JAX, we can't raise errors in jit-compiled code)
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    
    numerator = (0.059471695665734674 - 0.07585416297991414*finalDimlessSpin + 
                 0.021967909664591865*x2 - 0.0018964744613388146*x3 + 
                 0.001164879406179587*x4 - 0.0003387374454044957*x5)
    
    denominator = (1 - 1.4437415542456158*finalDimlessSpin + 0.49246920313191234*x2)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)

def evaluate_QNMfit_fring33(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring33
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    
    numerator = (0.09540436245212061 - 0.22799517865876945*finalDimlessSpin + 
                 0.13402916709362475*x2 + 0.03343753057911253*x3 - 
                 0.030848060170259615*x4 - 0.006756504382964637*x5 + 
                 0.0027301732074159835*x6)
    
    denominator = (1 - 2.7265947806178334*finalDimlessSpin + 2.144070539525238*x2 - 
                   0.4706873667569393*x4 + 0.05321818246993958*x6)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)

def evaluate_QNMfit_fring32(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring32
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    
    numerator = (0.09540436245212061 - 0.13628306966373951*finalDimlessSpin + 
                 0.030099881830507727*x2 - 0.000673589757007597*x3 + 
                 0.0118277880067919*x4 + 0.0020533816327907334*x5 - 
                 0.0015206141948469621*x6)
    
    denominator = (1 - 1.6531854335715193*finalDimlessSpin + 0.5634705514193629*x2 + 
                   0.12256204148002939*x4 - 0.027297817699401976*x6)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)

def evaluate_QNMfit_fring44(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fring44
    
    Args:
        finalDimlessSpin: Final dimensionless spin (float)
        
    Returns:
        float: QNM frequency fit result
    """
    
    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0
    
    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3
    
    numerator = (0.1287821193485683 - 0.21224284094693793*finalDimlessSpin + 
                 0.0710926778043916*x2 + 0.015487322972031054*x3 - 
                 0.002795401084713644*x4 + 0.000045483523029172406*x5 + 
                 0.00034775290179000503*x6)
    
    denominator = (1 - 1.9931645124693607*finalDimlessSpin + 1.0593147376898773*x2 - 
                   0.06378640753152783*x4)
    
    return_val = numerator / denominator
    
    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_fdamp21(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fdamp21 (damping frequency for (2,1) mode)

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: QNM damping frequency fit result
    """

    # Check bounds - return NaN for invalid input
    # (In JAX, we can't raise errors in jit-compiled code)
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2

    numerator = (2.0696914454467294 - 3.1358071947583093*finalDimlessSpin +
                 0.14456081596393977*x2 + 1.2194717985037946*x3 -
                 0.2947372598589144*x4 + 0.002943057145913646*x5)

    denominator = (146.1779212636481 - 219.81790388304876*finalDimlessSpin +
                   17.7141194900164*x2 + 75.90115083917898*x3 -
                   18.975287709794745*x4)

    return_val = numerator / denominator

    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_fdamp33(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fdamp33 (damping frequency for (3,3) mode)

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: QNM damping frequency fit result
    """

    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2

    numerator = (0.014754148319335946 - 0.03124423610028678*finalDimlessSpin +
                 0.017192623913708124*x2 + 0.001034954865629645*x3 -
                 0.0015925124814622795*x4 - 0.0001414350555699256*x5)

    denominator = (1.0 - 2.0963684630756894*finalDimlessSpin +
                   1.196809702382645*x2 - 0.09874113387889819*x4)

    return_val = numerator / denominator

    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_fdamp32(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fdamp32 (damping frequency for (3,2) mode)

    NOTE: This function is not defined in the original LALSimulation C code.
    For now, this returns the same as fdamp33 as a placeholder approximation.

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: QNM damping frequency fit result
    """
    # Placeholder: use fdamp33 as approximation since fdamp32 is not defined in C code
    return evaluate_QNMfit_fdamp33(finalDimlessSpin)


def evaluate_QNMfit_fdamp44(finalDimlessSpin: float) -> float:
    """
    Evaluate QNM fit for fdamp44 (damping frequency for (4,4) mode)

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: QNM damping frequency fit result
    """

    # Check bounds - return NaN for invalid input
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    numerator = (0.014986847152355699 - 0.01722587715950451*finalDimlessSpin -
                 0.0016734788189065538*x2 + 0.0002837322846047305*x3 +
                 0.002510528746148588*x4 + 0.00031983835498725354*x5 +
                 0.000812185411753066*x6)

    denominator = (1.0 - 1.1350205970682399*finalDimlessSpin -
                   0.0500827971270845*x2 + 0.13983808071522857*x4 +
                   0.051876225199833995*x6)

    return_val = numerator / denominator

    # Return NaN if input is invalid, otherwise return the computed value
    return jnp.where(valid_input, return_val, jnp.nan)


# Mixing coefficient functions for (3,2) mode
# These are used to transform spheroidal-harmonic ringdown ansatz to spherical-harmonic

def evaluate_QNMfit_re_l2m2lp2(finalDimlessSpin: float) -> float:
    """
    Evaluate real part of QNM mixing coefficient for l=2, m=2, lp=2

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: Real part of mixing coefficient
    """
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    numerator = (1 - 2.2956993576253635*finalDimlessSpin + 1.461988775298876*x2 +
                 0.0043296365593147035*x3 - 0.1695667458204109*x4 -
                 0.0006267849034466508*x5)

    denominator = (1 - 2.2956977727459043*finalDimlessSpin + 1.4646339137818438*x2 -
                   0.16843226886562457*x4 - 0.00007150540890128118*x6)

    return_val = numerator / denominator
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_im_l2m2lp2(finalDimlessSpin: float) -> float:
    """
    Evaluate imaginary part of QNM mixing coefficient for l=2, m=2, lp=2

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: Imaginary part of mixing coefficient
    """
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    numerator = (finalDimlessSpin * (0.3826673013161342 - 0.47531267226013896*finalDimlessSpin -
                 0.05898102880105067*x2 + 0.0724525431346487*x3 +
                 0.054714637311702986*x4 + 0.024544862718252784*x5))

    denominator = (-38.70835035062785 + 69.82140084545878*finalDimlessSpin -
                   27.99036444363243*x2 - 4.152310472191899*x4 + 1.*x6)

    return_val = numerator / denominator
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_re_l3m2lp2(finalDimlessSpin: float) -> float:
    """
    Evaluate real part of QNM mixing coefficient for l=3, m=2, lp=2

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: Real part of mixing coefficient
    """
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2

    numerator = (finalDimlessSpin * (0.47513455283841244 - 0.9016636384605536*finalDimlessSpin +
                 0.3844811236426182*x2 + 0.0855565148647794*x3 -
                 0.03620067426672167*x4 - 0.006557249133752502*x5))

    denominator = (-6.76894063440646 + 15.170831931186493*finalDimlessSpin -
                   9.406169787571082*x2 + 1.*x4)

    return_val = numerator / denominator
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_im_l3m2lp2(finalDimlessSpin: float) -> float:
    """
    Evaluate imaginary part of QNM mixing coefficient for l=3, m=2, lp=2

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: Imaginary part of mixing coefficient
    """
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    numerator = (finalDimlessSpin * (-2.8704762147145533 + 4.436434016918535*finalDimlessSpin -
                 1.0115343326360486*x2 - 0.08965314412106505*x3 -
                 0.4236810894599512*x4 - 0.041787576033810676*x5))

    denominator = (-171.80908957903395 + 272.362882450877*finalDimlessSpin -
                   76.68544453077854*x2 - 25.14197656531123*x4 + 1.*x6)

    return_val = numerator / denominator
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_re_l2m2lp3(finalDimlessSpin: float) -> float:
    """
    Evaluate real part of QNM mixing coefficient for l=2, m=2, lp=3

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: Real part of mixing coefficient
    """
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    numerator = (finalDimlessSpin * (18.522563276099167 - 37.978140351289014*finalDimlessSpin +
                 19.030390708998894*x2 + 3.0355668591803386*x3 -
                 2.210028290847915*x4 - 0.37117112862247975*x5))

    denominator = (164.52480238697507 - 377.9093045285145*finalDimlessSpin +
                   243.3353695550844*x2 - 30.79738566181734*x4 + 1.*x6)

    return_val = numerator / denominator
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_im_l2m2lp3(finalDimlessSpin: float) -> float:
    """
    Evaluate imaginary part of QNM mixing coefficient for l=2, m=2, lp=3

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: Imaginary part of mixing coefficient
    """
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    numerator = (finalDimlessSpin * (-49.7688437256778 + 120.43773704442333*finalDimlessSpin -
                 82.95323455645332*x2 + 1.721453011852496*x3 +
                 11.540237244397877*x4 - 0.9819458637589314*x5))

    denominator = (2858.5790831181725 - 6305.619505422591*finalDimlessSpin +
                   3825.6742092829054*x2 - 377.7822297815406*x4 + 1.*x6)

    return_val = numerator / denominator
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_re_l3m2lp3(finalDimlessSpin: float) -> float:
    """
    Evaluate real part of QNM mixing coefficient for l=3, m=2, lp=3

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: Real part of mixing coefficient
    """
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    numerator = (1 - 2.107852425643677*finalDimlessSpin + 1.1906393634562715*x2 +
                 0.02244848864087732*x3 - 0.09593447799423722*x4 -
                 0.0021343381708933025*x5 - 0.005319515989331159*x6)

    denominator = (1 - 2.1078515887706324*finalDimlessSpin + 1.2043484690080966*x2 -
                   0.08910191596778137*x4 - 0.005471749827809503*x6)

    return_val = numerator / denominator
    return jnp.where(valid_input, return_val, jnp.nan)


def evaluate_QNMfit_im_l3m2lp3(finalDimlessSpin: float) -> float:
    """
    Evaluate imaginary part of QNM mixing coefficient for l=3, m=2, lp=3

    Args:
        finalDimlessSpin: Final dimensionless spin (float)

    Returns:
        float: Imaginary part of mixing coefficient
    """
    valid_input = jnp.abs(finalDimlessSpin) <= 1.0

    x2 = finalDimlessSpin * finalDimlessSpin
    x3 = x2 * finalDimlessSpin
    x4 = x2 * x2
    x5 = x3 * x2
    x6 = x3 * x3

    numerator = (finalDimlessSpin * (12.45701482868677 - 29.398484595717147*finalDimlessSpin +
                 18.26221675782779*x2 + 1.9308599142669403*x3 -
                 3.159763242921214*x4 - 0.0910871567367674*x5))

    denominator = (345.52914639836257 - 815.4349339779621*finalDimlessSpin +
                   538.3888932415709*x2 - 69.3840921447381*x4 + 1.*x6)

    return_val = numerator / denominator
    return jnp.where(valid_input, return_val, jnp.nan)
