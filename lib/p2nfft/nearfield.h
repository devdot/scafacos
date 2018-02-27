/*
 * Copyright (C) 2011-2013 Michael Pippig
 *
 * This file is part of ScaFaCoS.
 * 
 * ScaFaCoS is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * ScaFaCoS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *	
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef _P2NFFT_NEARFIELD_H
#define _P2NFFT_NEARFIELD_H

#ifdef HAVE_CONFIG_H
#include <config.h>
#endif

#include <math.h>
#include "FCSCommon.h"
#include "constants.h"
#include "regularization.h"
#include "interpolation.h"


typedef struct {
  fcs_int interpolation_order;      /* interpolation order */
  fcs_int interpolation_num_nodes;  /* number of sampled points */
  fcs_float *near_interpolation_table_potential; /* nearfield potential values */
  fcs_float *near_interpolation_table_force;     /* nearfield force values */
  fcs_float one_over_r_cut;
} ifcs_p2nfft_near_params;


fcs_float ifcs_p2nfft_compute_self_potential(
    const void* param);
fcs_float ifcs_p2nfft_compute_near_potential(
    const void* param, fcs_float dist);
fcs_float ifcs_p2nfft_compute_near_field(
    const void* param, fcs_float dist);
void ifcs_p2nfft_compute_near(
    const void* param, fcs_float dist, 
    fcs_float *potential, fcs_float *field);


/* callback functions for near field computations */
static inline fcs_float
ifcs_p2nfft_compute_near_potential_periodic_erfc(
    const void *param, fcs_float dist
    )
{
  fcs_float alpha = *((fcs_float *) param);
  fcs_float adist = alpha * dist;
  return (FCS_CONST(1.0) - fcs_erf(adist)) / dist; /* use erf instead of erfc to fix ICC performance problems */
}

static inline fcs_float
ifcs_p2nfft_compute_near_field_periodic_erfc(
    const void *param, fcs_float dist
    )
{
  fcs_float alpha = *((fcs_float *) param);
  fcs_float inv_dist = FCS_CONST(1.0)/dist;
  fcs_float adist = alpha * dist;
  fcs_float erfc_part_ri = (FCS_CONST(1.0) - fcs_erf(adist)) * inv_dist; /* use erf instead of erfc to fix ICC performance problems */
  return -(erfc_part_ri + FCS_CONST(2.0)*alpha*FCS_P2NFFT_1_SQRTPI*fcs_exp(-adist*adist)) * inv_dist;
}


static inline void
ifcs_p2nfft_compute_near_periodic_erfc(
    const void *param, fcs_float dist,
    fcs_float *f, fcs_float *p
    )
{
  fcs_float alpha = *((fcs_float *) param);
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
    const void *param, fcs_float dist
    )
{
  fcs_float alpha = *((fcs_float *) param);
  fcs_float adist = alpha * dist;
  return ifcs_p2nfft_approx_erfc(adist) / dist;
}

static inline fcs_float
ifcs_p2nfft_compute_near_field_periodic_approx_erfc(
    const void *param, fcs_float dist
    )
{
  fcs_float alpha = *((fcs_float *) param);
  fcs_float inv_dist = FCS_CONST(1.0)/dist;
  fcs_float adist = alpha * dist;
  fcs_float erfc_part_ri = ifcs_p2nfft_approx_erfc(adist) *inv_dist;
  return -(erfc_part_ri + FCS_CONST(2.0)*alpha*FCS_P2NFFT_1_SQRTPI*fcs_exp(-adist*adist)) * inv_dist;
}

static inline void
ifcs_p2nfft_compute_near_periodic_approx_erfc(
    const void *param, fcs_float dist,
    fcs_float *f, fcs_float *p
    )
{
  fcs_float alpha = *((fcs_float *) param);
  fcs_float inv_dist = FCS_CONST(1.0)/dist;
  fcs_float adist = alpha * dist;
  fcs_float erfc_part_ri = ifcs_p2nfft_approx_erfc(adist) *inv_dist;

  *p = erfc_part_ri;
  *f = -(erfc_part_ri + FCS_CONST(2.0)*alpha*FCS_P2NFFT_1_SQRTPI*fcs_exp(-adist*adist)) * inv_dist;
}


static inline fcs_float
ifcs_p2nfft_compute_near_potential_interpolation(
    const void *param, fcs_float dist
    )
{
  ifcs_p2nfft_near_params *d = (ifcs_p2nfft_near_params*) param;

  return 1.0/dist - ifcs_p2nfft_interpolation_near(
      dist, d->one_over_r_cut,
      d->interpolation_order, d->interpolation_num_nodes,
      d->near_interpolation_table_potential);
}

static inline fcs_float
ifcs_p2nfft_compute_near_field_interpolation(
    const void *param, fcs_float dist
    )
{
  ifcs_p2nfft_near_params *d = (ifcs_p2nfft_near_params*) param;
  fcs_float inv_dist = 1.0/dist;

  return -inv_dist*inv_dist - ifcs_p2nfft_interpolation_near(
        dist, d->one_over_r_cut,
        d->interpolation_order, d->interpolation_num_nodes,
        d->near_interpolation_table_force);
}

static inline void
ifcs_p2nfft_compute_near_interpolation(
    const void *param, fcs_float dist,
    fcs_float *f, fcs_float *p
    )
{
  ifcs_p2nfft_near_params *d = (ifcs_p2nfft_near_params*) param;
  fcs_float inv_dist = 1.0/dist;

  *p = inv_dist - ifcs_p2nfft_interpolation_near(
      dist, d->one_over_r_cut,
      d->interpolation_order, d->interpolation_num_nodes,
      d->near_interpolation_table_potential);
  *f = -inv_dist*inv_dist - ifcs_p2nfft_interpolation_near(
        dist, d->one_over_r_cut,
        d->interpolation_order, d->interpolation_num_nodes,
        d->near_interpolation_table_force);
}

/* define one inline near field evaluation function for each interpolation order */
#define FCS_P2NFFT_NEAR_INTPOL_FUNC(_suffix_) \
  static inline void \
  ifcs_p2nfft_compute_near_interpolation_ ## _suffix_( \
      const void *param, fcs_float dist, \
      fcs_float *f, fcs_float *p \
      ) \
  { \
    ifcs_p2nfft_near_params *d = (ifcs_p2nfft_near_params*) param; \
    fcs_float inv_dist = 1.0/dist; \
    *p = inv_dist - ifcs_p2nfft_intpol_even_ ## _suffix_( \
        dist, d->near_interpolation_table_potential, \
        d->interpolation_num_nodes, d->one_over_r_cut); \
    *f = -inv_dist*inv_dist - ifcs_p2nfft_intpol_even_ ## _suffix_( \
        dist, d->near_interpolation_table_force, \
        d->interpolation_num_nodes, d->one_over_r_cut); \
  }

FCS_P2NFFT_NEAR_INTPOL_FUNC(const)
FCS_P2NFFT_NEAR_INTPOL_FUNC(lin)
FCS_P2NFFT_NEAR_INTPOL_FUNC(quad)
FCS_P2NFFT_NEAR_INTPOL_FUNC(cub)

#endif
