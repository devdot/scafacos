/*
 * Copyright (C) 2011-2013 Michael Pippig
 * Copyright (C) 2012 Alexander Köwitsch
 * Copyright (C) 2011 Sebastian Banert
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

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>


#include "run.h"
#include "types.h"
#include "utils.h"
#include "nearfield.h"
#include "interpolation.h"
#include <common/near/near.h>
//#include "constants.h"

#define FCS_P2NFFT_DISABLE_PNFFT_INFO 1
#define CREATE_GHOSTS_SEPARATE 0

/* callback functions for performing a whole loop of near field computations (using ifcs_p2nfft_compute_near_...) */
static FCS_NEAR_LOOP_FP(ifcs_p2nfft_compute_near_periodic_erfc_loop, ifcs_p2nfft_compute_near_periodic_erfc)
static FCS_NEAR_LOOP_FP(ifcs_p2nfft_compute_near_periodic_approx_erfc_loop, ifcs_p2nfft_compute_near_periodic_approx_erfc)
// static FCS_NEAR_LOOP_FP(ifcs_p2nfft_compute_near_interpolation_loop, ifcs_p2nfft_compute_near_interpolation)
static FCS_NEAR_LOOP_FP(ifcs_p2nfft_compute_near_interpolation_const_loop, ifcs_p2nfft_compute_near_interpolation_const)
static FCS_NEAR_LOOP_FP(ifcs_p2nfft_compute_near_interpolation_lin_loop, ifcs_p2nfft_compute_near_interpolation_lin)
static FCS_NEAR_LOOP_FP(ifcs_p2nfft_compute_near_interpolation_quad_loop, ifcs_p2nfft_compute_near_interpolation_quad)
static FCS_NEAR_LOOP_FP(ifcs_p2nfft_compute_near_interpolation_cub_loop, ifcs_p2nfft_compute_near_interpolation_cub)

static void convolution(
    const INT *local_N, const fcs_pnfft_complex *regkern_hat,
    fcs_pnfft_complex *f_hat);

#define SPROD3(_u_, _v_) ( (_u_)[0] * (_v_)[0] + (_u_)[1] * (_v_)[1] + (_u_)[2] * (_v_)[2] ) 
#define XYZ2TRI(_d_, _x_, _ib_) ( SPROD3((_ib_) + 3*(_d_), (_x_)) )
/* compute d-th component of A^T * v */
#define At_TIMES_VEC(_A_, _v_, _d_) ( (_v_)[0] * (_A_)[_d_] + (_v_)[1] * (_A_)[_d_ + 3] + (_v_)[2] * (_A_)[_d_ + 6] )


static fcs_int box_not_large_enough(
    fcs_int npart, const fcs_float *pos_with_offset, const fcs_float *box_base, const fcs_float *ibox, const fcs_int *periodicity
    )
{
  fcs_float pos[3];
  for(fcs_int j=0; j<npart; j++)
  {
    pos[0] = pos_with_offset[3*j + 0] - box_base[0];
    pos[1] = pos_with_offset[3*j + 1] - box_base[1];
    pos[2] = pos_with_offset[3*j + 2] - box_base[2];

    if( (periodicity[0] == 0) && (XYZ2TRI(0, pos, ibox) < 0.0) ) return 1;
    if( (periodicity[0] == 0) && (XYZ2TRI(0, pos, ibox) > 1.0) ) return 1;

    if( (periodicity[1] == 0) && (XYZ2TRI(1, pos, ibox) < 0.0) ) return 1;
    if( (periodicity[1] == 0) && (XYZ2TRI(1, pos, ibox) > 1.0) ) return 1;

    if( (periodicity[2] == 0) && (XYZ2TRI(2, pos, ibox) < 0.0) ) return 1;
    if( (periodicity[2] == 0) && (XYZ2TRI(2, pos, ibox) > 1.0) ) return 1;
  }

  return 0;
}

const char ifcs_p2nfft_near_compute_source[] = {
#include "nearfield.cl_str.h"
};

FCSResult ifcs_p2nfft_run(
    void *rd, fcs_int local_num_particles, fcs_int max_local_num_particles,
    fcs_float *positions, fcs_float *charges,
    fcs_float *potential, fcs_float *field
    )
{
  const char* fnc_name = "ifcs_p2nfft_run";
  ifcs_p2nfft_data_struct *d = (ifcs_p2nfft_data_struct*) rd;
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  C csum;
  C csum_global;
  fcs_float rsum;
  fcs_float rsum_global;
#endif

#if FCS_ENABLE_INFO || FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  int myrank;
  MPI_Comm_rank(d->cart_comm_3d, &myrank);
#endif

  FCS_P2NFFT_INIT_TIMING(d->cart_comm_3d);


  /* handle particles, that left the box [0,L] */
  /* for non-periodic boundary conditions: user must increase the box */
  if(box_not_large_enough(local_num_particles, positions, d->box_base, d->box_inv, d->periodicity))
    return fcs_result_create(FCS_ERROR_WRONG_ARGUMENT, fnc_name, "Box size does not fit. Some particles left the box.");

  /* TODO: implement additional scaling of particles to ensure x \in [0,L)
   * Idea: use allreduce to get min and max coordinates, adapt scaling of particles for every time step */

  /* Start forw sort timing */
  FCS_P2NFFT_START_TIMING(d->cart_comm_3d);
  
  /* Compute the near field */
  fcs_near_t near;

  fcs_int sorted_num_particles, ghost_num_particles;
  fcs_float *sorted_positions, *ghost_positions;
  fcs_float *sorted_charges, *ghost_charges;
  fcs_gridsort_index_t *sorted_indices, *ghost_indices;
  fcs_gridsort_t gridsort;

  fcs_gridsort_create(&gridsort);
  
  fcs_gridsort_set_system(&gridsort, d->box_base, d->ebox_a, d->ebox_b, d->ebox_c, d->periodicity);

  fcs_gridsort_set_bounds(&gridsort, d->lower_border, d->upper_border);

  fcs_gridsort_set_particles(&gridsort, local_num_particles, max_local_num_particles, positions, charges);

  fcs_gridsort_set_max_particle_move(&gridsort, d->max_particle_move);

  fcs_gridsort_set_cache(&gridsort, &d->gridsort_cache);

#if CREATE_GHOSTS_SEPARATE
  fcs_gridsort_sort_forward(&gridsort, 0, d->cart_comm_3d);
  FCS_P2NFFT_FINISH_TIMING(d->cart_comm_3d, "Forward grid sort");

  /* Start near sort timing */
  FCS_P2NFFT_START_TIMING(d->cart_comm_3d);
  if (d->short_range_flag) fcs_gridsort_create_ghosts(&gridsort, d->r_cut, d->cart_comm_3d);
#else
  fcs_gridsort_sort_forward(&gridsort, (d->short_range_flag ? d->r_cut: 0.0), d->cart_comm_3d);
#endif

  fcs_gridsort_separate_ghosts(&gridsort);

  fcs_gridsort_get_real_particles(&gridsort, &sorted_num_particles, &sorted_positions, &sorted_charges, &sorted_indices);
  fcs_gridsort_get_ghost_particles(&gridsort, &ghost_num_particles, &ghost_positions, &ghost_charges, &ghost_indices);

  /* Finish forw sort timing */
#if CREATE_GHOSTS_SEPARATE
  FCS_P2NFFT_FINISH_TIMING(d->cart_comm_3d, "Forward near sort");
#else
  FCS_P2NFFT_FINISH_TIMING(d->cart_comm_3d, "Forward grid and near sort");
#endif

  /* Handle particles, that left the box [0,L] */
  /* For periodic boundary conditions: just fold them back.
   * We change sorted_positions (and not positions), since we are allowed to overwrite them. */
  fcs_wrap_positions(sorted_num_particles, sorted_positions, d->box_a, d->box_b, d->box_c, d->box_base, d->periodicity);

  /* Start near field timing */
  FCS_P2NFFT_START_TIMING(d->cart_comm_3d);

/*  printf("%d: input number = %" FCS_LMOD_INT "d, sorted number = %" FCS_LMOD_INT "d, ghost number = %" FCS_LMOD_INT "d\n",
    myrank, local_num_particles, sorted_num_particles, ghost_num_particles);*/

  /* additional switchs to turn off computation of field / potential */
  if(d->flags & FCS_P2NFFT_IGNORE_FIELD)     field     = NULL;
  if(d->flags & FCS_P2NFFT_IGNORE_POTENTIAL) potential = NULL;

  fcs_int compute_field     = (field != NULL);
  fcs_int compute_potential = (potential != NULL);

  fcs_float *sorted_field     = (compute_field)     ? malloc(sizeof(fcs_float)*3*sorted_num_particles) : NULL;
  fcs_float *sorted_potential = (compute_potential) ? malloc(sizeof(fcs_float)*sorted_num_particles) : NULL;

  /* Initialize all the potential */
  if(compute_potential)
    for (fcs_int j = 0; j < sorted_num_particles; ++j)
      sorted_potential[j] = 0;
  
  /* Initialize all the forces */
  if(compute_field)
    for (fcs_int j = 0; j < 3 * sorted_num_particles; ++j)
      sorted_field[j] = 0;

  ifcs_p2nfft_near_params near_params;
  fcs_float *near_sorted_field = sorted_field;
  fcs_float *near_sorted_potential = sorted_potential;

  if(d->short_range_flag){
    fcs_near_create(&near);
  
    void *compute_param = NULL;

    if(d->interpolation_order >= 0){
      switch(d->interpolation_order){
        case 0: fcs_near_set_loop(&near, ifcs_p2nfft_compute_near_interpolation_const_loop); break;
        case 1: fcs_near_set_loop(&near, ifcs_p2nfft_compute_near_interpolation_lin_loop); break;
        case 2: fcs_near_set_loop(&near, ifcs_p2nfft_compute_near_interpolation_quad_loop); break;
        case 3: fcs_near_set_loop(&near, ifcs_p2nfft_compute_near_interpolation_cub_loop); break;
        default: return fcs_result_create(FCS_ERROR_WRONG_ARGUMENT, fnc_name,"P2NFFT interpolation order is too large.");
      } 
      near_params.interpolation_order = d->interpolation_order;
      near_params.interpolation_num_nodes = d->near_interpolation_num_nodes;
      near_params.near_interpolation_table_potential = d->near_interpolation_table_potential;
      near_params.near_interpolation_table_force = d->near_interpolation_table_force;
      near_params.one_over_r_cut = d->one_over_r_cut;

      compute_param = &near_params;
    } else if(d->reg_kernel == FCS_P2NFFT_REG_KERNEL_EWALD) {
      if(d->interpolation_order == -1) {
        fcs_near_set_loop(&near, ifcs_p2nfft_compute_near_periodic_erfc_loop);
        fcs_near_set_field_potential_source(&near, ifcs_p2nfft_near_compute_source, "ifcs_p2nfft_compute_near_periodic_erfc");
        fcs_near_set_compute_param_size(&near, sizeof(fcs_float));
      } else {
        fcs_near_set_loop(&near, ifcs_p2nfft_compute_near_periodic_approx_erfc_loop);
        fcs_near_set_field_potential_source(&near, ifcs_p2nfft_near_compute_source, "ifcs_p2nfft_compute_near_periodic_approx_erfc");
        fcs_near_set_compute_param_size(&near, sizeof(fcs_float));
      }
      compute_param = &d->alpha;
    } else {
      fcs_near_set_field(&near, ifcs_p2nfft_compute_near_field);
      fcs_near_set_potential(&near, ifcs_p2nfft_compute_near_potential);
      compute_param = rd;
    }

    // fcs_int *periodicity = NULL; /* sorter uses periodicity of the communicator */
    fcs_near_set_system(&near, d->box_base, d->box_a, d->box_b, d->box_c, d->periodicity);
  
    if (d->async_near)
    {
      near_sorted_field     = (compute_field)     ? malloc(sizeof(fcs_float)*3*sorted_num_particles) : NULL;
      near_sorted_potential = (compute_potential) ? malloc(sizeof(fcs_float)*sorted_num_particles) : NULL;

      if(compute_potential)
        for (fcs_int j = 0; j < sorted_num_particles; ++j)
          near_sorted_potential[j] = 0;

      if(compute_field)
        for (fcs_int j = 0; j < 3 * sorted_num_particles; ++j)
          near_sorted_field[j] = 0;
    }

    fcs_near_set_particles(&near, sorted_num_particles, sorted_num_particles, sorted_positions, sorted_charges, sorted_indices,
      near_sorted_field, near_sorted_potential);
  
    fcs_near_set_ghosts(&near, ghost_num_particles, ghost_positions, ghost_charges, ghost_indices);

    if (d->async_near)
    {
      fcs_near_compute_prepare(&near, d->r_cut, compute_param, d->cart_comm_3d);

      fcs_near_compute_start(&near);
  
    } else
    {
      fcs_near_compute(&near, d->r_cut, compute_param, d->cart_comm_3d);

      fcs_near_destroy(&near);
    }
  }

  /* Finish near field timing */
//  tm_timer += MPI_Wtime();
//  printf("P2NFFT_TIMING: rank = %d, Near field computation takes %e s\n", myrank, tm_timer);
  FCS_P2NFFT_FINISH_TIMING(d->cart_comm_3d, "Near field computation");
  
  /* Checksum: global sum of nearfield energy */
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  fcs_float near_energy = 0.0;
  fcs_float near_global;
  if(compute_potential)
    for(fcs_int j = 0; j < sorted_num_particles; ++j)
      near_energy += 0.5 * sorted_charges[j] * sorted_potential[j];
  MPI_Reduce(&near_energy, &near_global, 1, FCS_MPI_FLOAT, MPI_SUM, 0, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: near field energy: %" FCS_LMOD_FLOAT "f\n", near_global);
#endif

  /* Checksum: fields resulting from nearfield interactions */
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  for(fcs_int t=0; t<3; t++){
    rsum = 0.0;
    if(compute_field)
      for(fcs_int j = 0; (j < sorted_num_particles); ++j)
        rsum += fabs(sorted_field[3*j+t]);
    MPI_Reduce(&rsum, &rsum_global, 1, FCS_MPI_FLOAT, MPI_SUM, 0, d->cart_comm_3d);
    if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: near field %" FCS_LMOD_INT "d. component: %" FCS_LMOD_FLOAT "f\n", t, rsum_global);
  }
  
  if(compute_field)
    if (myrank == 0) fprintf(stderr, "E_NEAR(0) = %" FCS_LMOD_FLOAT "e\n", sorted_field[0]);
#endif
      
  /* Reinit PNFFT corresponding to number of sorted nodes */
  unsigned pnfft_malloc_flags = PNFFT_MALLOC_X | PNFFT_MALLOC_F;
  if(compute_field)
    pnfft_malloc_flags |= PNFFT_MALLOC_GRAD_F;

  FCS_PNFFT(init_nodes)(d->pnfft, sorted_num_particles,
      pnfft_malloc_flags,
      PNFFT_FREE_X|   PNFFT_FREE_F|   PNFFT_FREE_GRAD_F);

  fcs_pnfft_complex *f_hat, *f, *grad_f;
  fcs_float *x;

  f_hat  = FCS_PNFFT(get_f_hat)(d->pnfft);
  f      = FCS_PNFFT(get_f)(d->pnfft);
  grad_f = FCS_PNFFT(get_grad_f)(d->pnfft);
  x      = FCS_PNFFT(get_x)(d->pnfft);

// #if FCS_ENABLE_INFO
//   fcs_float min[3], max[3], gmin[3], gmax[3];
// 
//   /* initialize */
//   for(fcs_int t=0; t<3; t++)
//     min[t] = (sorted_num_particles >0) ? sorted_positions[t] : 1e16;
//   for(fcs_int t=0; t<3; t++)
//     max[t] = (sorted_num_particles >0) ? sorted_positions[t] : -1e16;
// 
//   for (fcs_int j = 1; j < sorted_num_particles; ++j){
//     for(fcs_int t=0; t<3; t++){
//       if(sorted_positions[3*j+t] < min[t])
// 	min[t] = sorted_positions[3*j+t];
//       if(sorted_positions[3*j+t] > max[t])
// 	max[t] = sorted_positions[3*j+t];
//     }
//   }
//       
//   MPI_Reduce(&min, &gmin, 3, FCS_MPI_FLOAT, MPI_MIN, 0, d->cart_comm_3d);
//   MPI_Reduce(&max, &gmax, 3, FCS_MPI_FLOAT, MPI_MAX, 0, d->cart_comm_3d);
//   if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: min range of particles: (%e, %e, %e)\n", gmin[0], gmin[1], gmin[2]);
//   if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: max range of particles: (%e, %e, %e)\n", gmax[0], gmax[1], gmax[2]);
//   fprintf(stderr, "myrank = %d, sorted_num_particles = %d\n", myrank, sorted_num_particles);
// #endif
  
  /* Set NFFT nodes within [-0.5,0.5]^3 */
  for (fcs_int j = 0; j < sorted_num_particles; ++j)
  {
    fcs_float pos[3];

    pos[0] = sorted_positions[3*j + 0] - d->box_base[0];
    pos[1] = sorted_positions[3*j + 1] - d->box_base[1];
    pos[2] = sorted_positions[3*j + 2] - d->box_base[2];

    x[3 * j + 0] = ( XYZ2TRI(0, pos, d->box_inv) - 0.5 ) / d->box_expand[0];
    x[3 * j + 1] = ( XYZ2TRI(1, pos, d->box_inv) - 0.5 ) / d->box_expand[1];
    x[3 * j + 2] = ( XYZ2TRI(2, pos, d->box_inv) - 0.5 ) / d->box_expand[2];
  }
    
  /* Set NFFT values */
  for (fcs_int j = 0; j < sorted_num_particles; ++j) f[j] = sorted_charges[j];

// #if FCS_ENABLE_INFO
//   fcs_float min[3], max[3], gmin[3], gmax[3];
// 
//   /* initialize */
//   for(fcs_int t=0; t<3; t++)
//     min[t] = (sorted_num_particles >0) ? x[t] : 1e16;
//   for(fcs_int t=0; t<3; t++)
//     max[t] = (sorted_num_particles >0) ? x[t] : -1e16;
// 
//   for (fcs_int j = 1; j < sorted_num_particles; ++j){
//     for(fcs_int t=0; t<3; t++){
//       if(x[3*j+t] < min[t])
// 	min[t] = x[3*j+t];
//       if(x[3*j+t] > max[t])
// 	max[t] = x[3*j+t];
//     }
//   }
//       
//   MPI_Reduce(&min, &gmin, 3, FCS_MPI_FLOAT, MPI_MIN, 0, d->cart_comm_3d);
//   MPI_Reduce(&max, &gmax, 3, FCS_MPI_FLOAT, MPI_MAX, 0, d->cart_comm_3d);
//   if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: min range of particles: (%e, %e, %e)\n", gmin[0], gmin[1], gmin[2]);
//   if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: max range of particles: (%e, %e, %e)\n", gmax[0], gmax[1], gmax[2]);
//   fprintf(stderr, "myrank = %d, sorted_num_particles = %d\n", myrank, sorted_num_particles);
// #endif

  FCS_P2NFFT_START_TIMING(d->cart_comm_3d);
  FCS_PNFFT(precompute_psi)(d->pnfft);
  FCS_P2NFFT_FINISH_TIMING(d->cart_comm_3d, "pnfft_precompute_psi");

  /* Reset pnfft timer (delete timings from fcs_init and fcs_tune) */  
#if FCS_ENABLE_INFO && !FCS_P2NFFT_DISABLE_PNFFT_INFO
  FCS_PNFFT(reset_timer)(d->pnfft);
#endif

  /* Checksum: Input of adjoint NFFT */
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  rsum = 0.0;
  for (fcs_int j = 0; j < 3*sorted_num_particles; ++j)
    rsum += fabs(x[j]);
  MPI_Reduce(&rsum, &rsum_global, 1, MPI_DOUBLE, MPI_SUM, 0, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: checksum of x: %" FCS_LMOD_FLOAT "e\n", rsum_global);

  csum = 0.0;
  for (fcs_int j = 0; j < sorted_num_particles; ++j)
    csum += fabs(creal(f[j])) + _Complex_I * fabs(cimag(f[j]));
  MPI_Reduce(&csum, &csum_global, 2, MPI_DOUBLE, MPI_SUM, 0, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: checksum of NFFT^H input: %e + I* %e\n", creal(csum_global), cimag(csum_global));
#endif

  /* Start far field timing */
  FCS_P2NFFT_START_TIMING(d->cart_comm_3d);

  /* Perform adjoint NFFT */
  if(!d->pnfft_direct)
    FCS_PNFFT(adj)(d->pnfft);
  else
    FCS_PNFFT(direct_adj)(d->pnfft);

  /* Checksum: Output of adjoint NFFT */  
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  csum = 0.0;
  for(fcs_int k = 0; k < d->local_N[0]*d->local_N[1]*d->local_N[2]; ++k)
     csum += fabs(creal(f_hat[k])) + _Complex_I * fabs(cimag(f_hat[k]));
  MPI_Reduce(&csum, &csum_global, 2, MPI_DOUBLE, MPI_SUM, 0, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: checksum of Fourier coefficients before convolution: %e + I* %e\n", creal(csum_global), cimag(csum_global));
#endif

  /* Checksum: Fourier coefficients of regkernel  */  
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  csum = 0.0;
  for(fcs_int k = 0; k < d->local_N[0]*d->local_N[1]*d->local_N[2]; ++k)
     csum += fabs(creal(d->regkern_hat[k])) + _Complex_I * fabs(cimag(d->regkern_hat[k]));
  MPI_Reduce(&csum, &csum_global, 2, MPI_DOUBLE, MPI_SUM, 0, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: checksum of Regkernel Fourier coefficients: %e + I* %e\n", creal(csum_global), cimag(csum_global));
#endif

  /* Multiply with the analytically given Fourier coefficients */
  convolution(d->local_N, d->regkern_hat,
      f_hat);

  /* Checksum: Input of NFFT */
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  csum = 0.0;
  for(fcs_int k = 0; k < d->local_N[0]*d->local_N[1]*d->local_N[2]; ++k)
     csum += fabs(creal(f_hat[k])) + _Complex_I * fabs(cimag(f_hat[k]));
  MPI_Reduce(&csum, &csum_global, 2, MPI_DOUBLE, MPI_SUM, 0, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: checksum of Fourier coefficients after convolution: %e + I* %e\n", creal(csum_global), cimag(csum_global));
#endif
    
  /* Perform NFFT */
  if(!d->pnfft_direct)
    FCS_PNFFT(trafo)(d->pnfft);
  else
    FCS_PNFFT(direct_trafo)(d->pnfft);

  /* Copy the results to the output vector and rescale with L^{-T} */
  if(compute_potential){
    for (fcs_int j = 0; j < sorted_num_particles; ++j){
      sorted_potential[j] += fcs_creal(f[j]);
    }
  }

  if(compute_field){
    for (fcs_int j = 0; j < sorted_num_particles; ++j){
      sorted_field[3 * j + 0] -= fcs_creal( At_TIMES_VEC(d->ebox_inv, grad_f + 3*j, 0) );
      sorted_field[3 * j + 1] -= fcs_creal( At_TIMES_VEC(d->ebox_inv, grad_f + 3*j, 1) );
      sorted_field[3 * j + 2] -= fcs_creal( At_TIMES_VEC(d->ebox_inv, grad_f + 3*j, 2) );
    }
  }

  /* Checksum: fields resulting from farfield interactions */
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  for(fcs_int t=0; t<3; t++){
    rsum = 0.0;
    if(compute_field)
      for(fcs_int j = 0; (j < sorted_num_particles); ++j)
        rsum += fabs(sorted_field[3*j+t]);
    MPI_Reduce(&rsum, &rsum_global, 1, FCS_MPI_FLOAT, MPI_SUM, 0, d->cart_comm_3d);
    if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: checksum near plus far field %" FCS_LMOD_INT "d. component: %" FCS_LMOD_FLOAT "f\n", t, rsum_global);
  }

  if(compute_field){
    if (myrank == 0) fprintf(stderr, "E_FAR(0) = %" FCS_LMOD_FLOAT "e\n", -fcs_creal( At_TIMES_VEC(d->ebox_inv, grad_f + 3*0, 0) ));
    if (myrank == 0) fprintf(stderr, "E_NEAR_FAR(0) = %" FCS_LMOD_FLOAT "e\n", sorted_field[0]);
  }
#endif

  /* Finish far field timing */
  FCS_P2NFFT_FINISH_TIMING(d->cart_comm_3d, "Far field computation");

#if FCS_ENABLE_TIMING_PNFFT
  /* Print pnfft timer */
  FCS_P2NFFT_START_TIMING(d->cart_comm_3d);
  FCS_PNFFT(print_average_timer_adv)(d->pnfft, d->cart_comm_3d);
  FCS_P2NFFT_FINISH_TIMING(d->cart_comm_3d, "Printing of PNFFT timings");
#endif

  /* Checksum: global sum of farfield energy */
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  fcs_float far_energy = 0.0;
  fcs_float far_global;

  for(fcs_int j = 0; j < sorted_num_particles; ++j)
    far_energy += 0.5 * sorted_charges[j] * f[j];

  MPI_Reduce(&far_energy, &far_global, 1, FCS_MPI_FLOAT, MPI_SUM, 0, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: far field energy: %" FCS_LMOD_FLOAT "f\n", far_global);
#endif
  
  /* Start self interaction timing */
  FCS_P2NFFT_START_TIMING(d->cart_comm_3d);

  /* Calculate self-interactions */
  if(compute_potential)
    for (fcs_int j = 0; j < sorted_num_particles; ++j)
      sorted_potential[j] -= sorted_charges[j] * ifcs_p2nfft_compute_self_potential(rd);

  /* Finish self interaction timing */
  FCS_P2NFFT_FINISH_TIMING(d->cart_comm_3d, "self interaction calculation");

  if(d->short_range_flag && d->async_near){
    fcs_near_compute_join(&near);
    fcs_near_compute_finish(&near);

    if(compute_potential){
      for (fcs_int j = 0; j < sorted_num_particles; ++j){
        sorted_potential[j] += near_sorted_potential[j];
      }
      free(near_sorted_potential);
    }

    if(compute_field){
      for (fcs_int j = 0; j < sorted_num_particles; ++j){
        sorted_field[3 * j + 0] += near_sorted_field[3 * j + 0];
        sorted_field[3 * j + 1] += near_sorted_field[3 * j + 1];
        sorted_field[3 * j + 2] += near_sorted_field[3 * j + 2];
      }
      free(near_sorted_field);
    }

    fcs_near_destroy(&near);
  }

  /* Calculate virial if needed */
  if(d->virial != NULL){
    if ((d->num_periodic_dims == 3) && (d->reg_kernel == FCS_P2NFFT_REG_KERNEL_EWALD)) {
      fcs_float total_energy = 0.0;
      fcs_float total_global;
      if(compute_potential)
        for(fcs_int j = 0; j < sorted_num_particles; ++j)
          total_energy += 0.5 * sorted_charges[j] * sorted_potential[j];

      MPI_Allreduce(&total_energy, &total_global, 1, FCS_MPI_FLOAT, MPI_SUM, d->cart_comm_3d);

      /* Approximate virial in 3d-periodic case:
       * Fill the main diagonal with one third of the total energy */      
      for(fcs_int t=0; t<9; t++)
        d->virial[t] = 0.0;
      d->virial[0] = d->virial[4] = d->virial[8] = total_global/3.0;
    } 
    else if ((d->num_periodic_dims == 0) && (d->reg_kernel == FCS_P2NFFT_REG_KERNEL_OTHER)) {
      fcs_float local_virial[9];

      for(fcs_int t=0; t<9; t++)
        local_virial[t] = 0.0;
      
      if(compute_field)
        for(fcs_int j = 0; j < sorted_num_particles; ++j)
          for(fcs_int t0=0; t0<3; t0++)
            for(fcs_int t1=0; t1<3; t1++)
              local_virial[t0*3+t1] += sorted_charges[j] * sorted_field[3*j+t0] * sorted_positions[3*j+t1];
      
      MPI_Allreduce(local_virial, d->virial, 9, FCS_MPI_FLOAT, MPI_SUM, d->cart_comm_3d);
    }
    else {
      return fcs_result_create(FCS_ERROR_WRONG_ARGUMENT, fnc_name, "Virial computation is currently not available for mixed boundary conditions.");
    }

  }

  /* Checksum: global sum of self energy */
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  fcs_float self_energy = 0.0;
  fcs_float self_global;
  for(fcs_int j = 0; j < sorted_num_particles; ++j)
    self_energy -= 0.5 * sorted_charges[j] * sorted_charges[j] * ifcs_p2nfft_compute_self_potential(rd);

  MPI_Reduce(&self_energy, &self_global, 1, FCS_MPI_FLOAT, MPI_SUM, 0, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: self energy: %" FCS_LMOD_FLOAT "f\n", self_global);
#endif
      
  /* Checksum: global sum of total energy */
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  fcs_float total_energy = 0.0;
  fcs_float total_global;
  if(compute_potential)
    for(fcs_int j = 0; j < sorted_num_particles; ++j)
      total_energy += 0.5 * sorted_charges[j] * sorted_potential[j];

  MPI_Reduce(&total_energy, &total_global, 1, FCS_MPI_FLOAT, MPI_SUM, 0, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: total energy: %" FCS_LMOD_FLOAT "f\n", total_global);
#endif

  /* Try: calculate total dipol moment */
#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  fcs_float total_dipol_local[3] = {0.0, 0.0, 0.0};
  fcs_float total_dipol_global[3];
  for(fcs_int j = 0; j < sorted_num_particles; ++j)
    for(fcs_int t=0; t<3; ++t)
//       total_dipol_local[t] += sorted_charges[j] * sorted_positions[3*j+t];
      total_dipol_local[t] += sorted_charges[j] * (sorted_positions[3*j+t] - d->box_l[t]);

  MPI_Allreduce(&total_dipol_local, &total_dipol_global, 3, FCS_MPI_FLOAT, MPI_SUM, d->cart_comm_3d);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: total dipol: (%" FCS_LMOD_FLOAT "e, %" FCS_LMOD_FLOAT "e, %" FCS_LMOD_FLOAT "e)\n", total_dipol_global[0], total_dipol_global[1], total_dipol_global[2]);

  for(fcs_int t=0; t<3; ++t)
//     total_dipol_global[t] *= 4.0*PNFFT_PI/3.0 / (d->box_l[0]*d->box_l[1]*d->box_l[2]);
//     total_dipol_global[t] *= PNFFT_PI / (d->box_l[0]*d->box_l[1]*d->box_l[2]);
    total_dipol_global[t] *= PNFFT_PI/3.0 *PNFFT_PI / (d->box_l[0]*d->box_l[1]*d->box_l[2]);
//     total_dipol_global[t] *= 2.0*PNFFT_PI / (d->box_l[0]*d->box_l[1]*d->box_l[2]);
  if (myrank == 0) fprintf(stderr, "P2NFFT_DEBUG: scaled total dipol: (%" FCS_LMOD_FLOAT "e, %" FCS_LMOD_FLOAT "e, %" FCS_LMOD_FLOAT "e)\n", total_dipol_global[0], total_dipol_global[1], total_dipol_global[2]);
#endif
      
  /* Start back sort timing */
  FCS_P2NFFT_START_TIMING(d->cart_comm_3d);

  fcs_gridsort_set_sorted_results(&gridsort, sorted_num_particles, sorted_field, sorted_potential);
  fcs_gridsort_set_results(&gridsort, max_local_num_particles, field, potential);

  fcs_int resort;

  if (d->resort) resort = fcs_gridsort_prepare_resort(&gridsort, d->cart_comm_3d);
  else resort = 0;

  /* Backsort data into user given ordering (if resort is disabled) */
  if (!resort) fcs_gridsort_sort_backward(&gridsort, d->cart_comm_3d);

  fcs_gridsort_resort_destroy(&d->gridsort_resort);

  if (resort) fcs_gridsort_resort_create(&d->gridsort_resort, &gridsort, d->cart_comm_3d);
  
  d->local_num_particles = local_num_particles;

  if (sorted_field) free(sorted_field);
  if (sorted_potential) free(sorted_potential);

  fcs_gridsort_free(&gridsort);

  fcs_gridsort_destroy(&gridsort);

  /* Finish back sort timing */
  FCS_P2NFFT_FINISH_TIMING(d->cart_comm_3d, "Backward sort");

#if FCS_ENABLE_DEBUG || FCS_P2NFFT_DEBUG
  /* print first value of fields */
  if(compute_field){
    if (myrank == 0) printf("P2NFFT_INFO: E(0) = %e\n", creal(field[0]));
    if (myrank == 0) printf("P2NFFT_INFO: dE(0) = %e\n", creal(field[0])+1.619834832399799e-06);
  }
#endif

  return NULL;
}

static void convolution(
    const INT *local_N, const fcs_pnfft_complex *regkern_hat,
    fcs_pnfft_complex *f_hat
    )
{  
  INT m=0;

  for (INT k0 = 0; k0 < local_N[0]; ++k0)
    for (INT k1 = 0; k1 < local_N[1]; ++k1)
      for (INT k2 = 0; k2 < local_N[2]; ++k2, ++m){
//         fprintf(stderr, "f_hat[%td, %td, %td] = %e + I * %e, regkern_hat[%td, %td, %td] = %e I * %e\n", k0, k1, k2, creal(f_hat[m]), cimag(f_hat[m]), k0, k1, k2, creal(regkern_hat[m]), cimag(regkern_hat[m]));
//         fprintf(stderr, "regkern_hat[%td, %td, %td] = %e I * %e\n", k0, k1, k2, creal(regkern_hat[m]), cimag(regkern_hat[m]));
        f_hat[m] *= regkern_hat[m];
      }
}


void ifcs_p2nfft_set_max_particle_move(void *rd, fcs_float max_particle_move)
{
  ifcs_p2nfft_data_struct *d = (ifcs_p2nfft_data_struct*) rd;

  d->max_particle_move = max_particle_move;
}

void ifcs_p2nfft_set_resort(void *rd, fcs_int resort)
{
  ifcs_p2nfft_data_struct *d = (ifcs_p2nfft_data_struct*) rd;

  d->resort = resort;
}

void ifcs_p2nfft_get_resort(void *rd, fcs_int *resort)
{
  ifcs_p2nfft_data_struct *d = (ifcs_p2nfft_data_struct*) rd;

  *resort = d->resort;
}

void ifcs_p2nfft_get_resort_availability(void *rd, fcs_int *availability)
{
  ifcs_p2nfft_data_struct *d = (ifcs_p2nfft_data_struct*) rd;

  *availability = fcs_gridsort_resort_is_available(d->gridsort_resort);
}

void ifcs_p2nfft_get_resort_particles(void *rd, fcs_int *resort_particles)
{
  ifcs_p2nfft_data_struct *d = (ifcs_p2nfft_data_struct*) rd;

  if (d->gridsort_resort == FCS_GRIDSORT_RESORT_NULL)
  {
    *resort_particles = d->local_num_particles;
    return;
  }
  
  *resort_particles = fcs_gridsort_resort_get_sorted_particles(d->gridsort_resort);
}

void ifcs_p2nfft_resort_ints(void *rd, fcs_int *src, fcs_int *dst, fcs_int n, MPI_Comm comm)
{
  ifcs_p2nfft_data_struct *d = (ifcs_p2nfft_data_struct*) rd;

  if (d->gridsort_resort == FCS_GRIDSORT_RESORT_NULL) return;
  
  fcs_gridsort_resort_ints(d->gridsort_resort, src, dst, n, comm);
}

void ifcs_p2nfft_resort_floats(void *rd, fcs_float *src, fcs_float *dst, fcs_int n, MPI_Comm comm)
{
  ifcs_p2nfft_data_struct *d = (ifcs_p2nfft_data_struct*) rd;

  if (d->gridsort_resort == FCS_GRIDSORT_RESORT_NULL) return;
  
  fcs_gridsort_resort_floats(d->gridsort_resort, src, dst, n, comm);
}

void ifcs_p2nfft_resort_bytes(void *rd, void *src, void *dst, fcs_int n, MPI_Comm comm)
{
  ifcs_p2nfft_data_struct *d = (ifcs_p2nfft_data_struct*) rd;

  if (d->gridsort_resort == FCS_GRIDSORT_RESORT_NULL) return;
  
  fcs_gridsort_resort_bytes(d->gridsort_resort, src, dst, n, comm);
}
