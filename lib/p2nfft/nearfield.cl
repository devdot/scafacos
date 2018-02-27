
#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include "FCSCommon.h"
#include "constants.h"


typedef void HERE_COMES_THE_CODE;


/* callback functions for near field computations */
static inline fcs_float
ifcs_p2nfft_compute_near_potential_periodic_erfc(
    __global const void *param, fcs_float dist
    )
{
  fcs_float alpha = *((__global fcs_float *) param);
  fcs_float adist = alpha * dist;
  return (FCS_CONST(1.0) - fcs_erf(adist)) / dist; /* use erf instead of erfc to fix ICC performance problems */
}

static inline fcs_float
ifcs_p2nfft_compute_near_field_periodic_erfc(
    __global const void *param, fcs_float dist
    )
{
  fcs_float alpha = *((__global fcs_float *) param);
  fcs_float inv_dist = FCS_CONST(1.0)/dist;
  fcs_float adist = alpha * dist;
  fcs_float erfc_part_ri = (FCS_CONST(1.0) - fcs_erf(adist)) * inv_dist; /* use erf instead of erfc to fix ICC performance problems */
  return -(erfc_part_ri + FCS_CONST(2.0)*alpha*FCS_P2NFFT_1_SQRTPI*fcs_exp(-adist*adist)) * inv_dist;
}

static inline void
ifcs_p2nfft_compute_near_periodic_erfc(
    __global const void *param, fcs_float dist,
    fcs_float *f, fcs_float *p
    )
{
  fcs_float alpha = *((__global fcs_float *) param);
  fcs_float inv_dist = FCS_CONST(1.0)/dist;
  fcs_float adist = alpha * dist;
  fcs_float erfc_part_ri = (FCS_CONST(1.0) - fcs_erf(adist)) * inv_dist; /* use erf instead of erfc to fix ICC performance problems */
  *p = erfc_part_ri;
  *f = -(erfc_part_ri + FCS_CONST(2.0)*alpha*FCS_P2NFFT_1_SQRTPI*fcs_exp(-adist*adist)) * inv_dist;
}

static inline fcs_float
ifcs_p2nfft_approx_erfc(
    fcs_float adist
    )
{
  /* approximate \f$ \exp(d^2) \mathrm{erfc}(d)\f$ by applying formula 7.1.26 from:
     Abramowitz/Stegun: Handbook of Mathematical Functions, Dover (9. ed.), chapter 7.
     Error <= 1.5e-7 */
  fcs_float t = FCS_CONST(1.0) / (FCS_CONST(1.0) + FCS_CONST(0.3275911) * adist);
  return fcs_exp(-adist*adist) * 
    (t * (FCS_CONST(0.254829592) + 
          t * (FCS_CONST(-0.284496736) + 
               t * (FCS_CONST(1.421413741) + 
                    t * (FCS_CONST(-1.453152027) + 
                         t * FCS_CONST(1.061405429))))));
} 

static inline fcs_float
ifcs_p2nfft_compute_near_potential_periodic_approx_erfc(
    __global const void *param, fcs_float dist
    )
{
  fcs_float alpha = *((__global fcs_float *) param);
  fcs_float adist = alpha * dist;
  return ifcs_p2nfft_approx_erfc(adist) / dist;
}

static inline fcs_float
ifcs_p2nfft_compute_near_field_periodic_approx_erfc(
    __global const void *param, fcs_float dist
    )
{
  fcs_float alpha = *((__global fcs_float *) param);
  fcs_float inv_dist = FCS_CONST(1.0)/dist;
  fcs_float adist = alpha * dist;
  fcs_float erfc_part_ri = ifcs_p2nfft_approx_erfc(adist) *inv_dist;
  return -(erfc_part_ri + FCS_CONST(2.0)*alpha*FCS_P2NFFT_1_SQRTPI*fcs_exp(-adist*adist)) * inv_dist;
}

static inline void
ifcs_p2nfft_compute_near_periodic_approx_erfc(
    __global const void *param, fcs_float dist,
    fcs_float *f, fcs_float *p
    )
{
  fcs_float alpha = *((__global fcs_float *) param);
  fcs_float inv_dist = FCS_CONST(1.0)/dist;
  fcs_float adist = alpha * dist;
  fcs_float erfc_part_ri = ifcs_p2nfft_approx_erfc(adist) *inv_dist;

  *p = erfc_part_ri;
  *f = -(erfc_part_ri + FCS_CONST(2.0)*alpha*FCS_P2NFFT_1_SQRTPI*fcs_exp(-adist*adist)) * inv_dist;
}
