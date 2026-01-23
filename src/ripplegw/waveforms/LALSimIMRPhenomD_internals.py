
import jax.numpy as jnp
"""
def DPhiMRD(double f, IMRPhenomDPhaseCoefficients *p, double Rholm, double Taulm) {
  return ( p->alpha1 + p->alpha2/pow_2_of(f) + p->alpha3/pow(f,0.25)+ p->alpha4/(p->fDM * Taulm * (1 + pow_2_of(f - p->alpha5 * p->fRD)/(pow_2_of(p->fDM * Taulm * Rholm)))) ) * p->etaInv;
}
"""
def DPhiMRD(f, alpha1, alpha2, alpha3, alpha4, alpha5, fRD, eta, fDM, Taulm, Rholm):


    term1 = alpha1
    term2 = alpha2 / jnp.power(f, 2)
    term3 = alpha3 / jnp.power(f, 0.25)
    term4_numerator = alpha4 
    term4_denom = fDM * Taulm * (1 + jnp.power(f-alpha5*fRD , 2) / jnp.power(fDM * Taulm * Rholm, 2))
    term4 = term4_numerator / term4_denom
    return (term1 + term2 + term3 + term4) / eta