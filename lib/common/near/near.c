/*
  Copyright (C) 2011, 2012, 2013 Olaf Lenz, Rene Halver, Michael Hofmann
  Copyright (C) 2017 Dirk Leichsenring
  Copyright (C) 2018 Michael Hofmann
  
  This file is part of ScaFaCoS.
  
  ScaFaCoS is free software: you can redistribute it and/or modify
  it under the terms of the GNU Lesser Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  ScaFaCoS is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU Lesser Public License for more details.
  
  You should have received a copy of the GNU Lesser Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include "near_common.h"

static fcs_int real_neighbours[] = {
   1,  0,  0,
  -1,  1,  0,
   0,  1,  0,
   1,  1,  0,
  -1, -1,  1,
   0, -1,  1,
   1, -1,  1,
  -1,  0,  1,
   0,  0,  1,
   1,  0,  1,
  -1,  1,  1,
   0,  1,  1,
   1,  1,  1, };
static fcs_int nreal_neighbours = sizeof(real_neighbours) / 3 / sizeof(fcs_int);

#if FCS_NEAR_OCL

static fcs_int real_neighbours_full[] = {
  -1, -1, -1,
   0, -1, -1,
   1, -1, -1,
  -1,  0, -1,
   0,  0, -1,
   1,  0, -1,
  -1,  1, -1,
   0,  1, -1,
   1,  1, -1,
  -1, -1,  0,
   0, -1,  0,
   1, -1,  0,
  -1,  0,  0,
   1,  0,  0,
  -1,  1,  0,
   0,  1,  0,
   1,  1,  0,
  -1, -1,  1,
   0, -1,  1,
   1, -1,  1,
  -1,  0,  1,
   0,  0,  1,
   1,  0,  1,
  -1,  1,  1,
   0,  1,  1,
   1,  1,  1, };
static fcs_int nreal_neighbours_full = sizeof(real_neighbours_full) / 3 / sizeof(fcs_int);

#endif /* FCS_NEAR_OCL */

static fcs_int ghost_neighbours[] = {
  -1, -1, -1,
   0, -1, -1,
   1, -1, -1,
  -1,  0, -1,
   0,  0, -1,
   1,  0, -1,
  -1,  1, -1,
   0,  1, -1,
   1,  1, -1,
  -1, -1,  0,
   0, -1,  0,
   1, -1,  0,
  -1,  0,  0,
   0,  0,  0,
   1,  0,  0,
  -1,  1,  0,
   0,  1,  0,
   1,  1,  0,
  -1, -1,  1,
   0, -1,  1,
   1, -1,  1,
  -1,  0,  1,
   0,  0,  1,
   1,  0,  1,
  -1,  1,  1,
   0,  1,  1,
   1,  1,  1, };
static fcs_int nghost_neighbours = sizeof(ghost_neighbours) / 3 / sizeof(fcs_int);

#define max_nboxes  27

#define box_fmt       "%lld,%lld,%lld (%lld)"
#define box_val(_b_)  BOX_GET_X(_b_, 0), BOX_GET_X(_b_, 1), BOX_GET_X(_b_, 2), (_b_)

#define idx_fmt       "%d,%d,%d,%d,%d,%d (%lld)"
#define idx_val(_x_)  (int) GRIDSORT_PERIODIC_GET(_x_, 0), \
                      (int) GRIDSORT_PERIODIC_GET(_x_, 1), \
                      (int) GRIDSORT_PERIODIC_GET(_x_, 2), \
                      (int) GRIDSORT_PERIODIC_GET(_x_, 3), \
                      (int) GRIDSORT_PERIODIC_GET(_x_, 4), \
                      (int) GRIDSORT_PERIODIC_GET(_x_, 5), \
                      (_x_)


#ifdef BOX_SFC
static int int2sfc[] = {
  0x000000, 0x000001, 0x000008, 0x000009, 0x000040, 0x000041, 0x000048, 0x000049, 0x000200, 0x000201, 0x000208, 0x000209, 0x000240, 0x000241, 0x000248, 0x000249,
  0x001000, 0x001001, 0x001008, 0x001009, 0x001040, 0x001041, 0x001048, 0x001049, 0x001200, 0x001201, 0x001208, 0x001209, 0x001240, 0x001241, 0x001248, 0x001249,
  0x008000, 0x008001, 0x008008, 0x008009, 0x008040, 0x008041, 0x008048, 0x008049, 0x008200, 0x008201, 0x008208, 0x008209, 0x008240, 0x008241, 0x008248, 0x008249,
  0x009000, 0x009001, 0x009008, 0x009009, 0x009040, 0x009041, 0x009048, 0x009049, 0x009200, 0x009201, 0x009208, 0x009209, 0x009240, 0x009241, 0x009248, 0x009249,
  0x040000, 0x040001, 0x040008, 0x040009, 0x040040, 0x040041, 0x040048, 0x040049, 0x040200, 0x040201, 0x040208, 0x040209, 0x040240, 0x040241, 0x040248, 0x040249,
  0x041000, 0x041001, 0x041008, 0x041009, 0x041040, 0x041041, 0x041048, 0x041049, 0x041200, 0x041201, 0x041208, 0x041209, 0x041240, 0x041241, 0x041248, 0x041249,
  0x048000, 0x048001, 0x048008, 0x048009, 0x048040, 0x048041, 0x048048, 0x048049, 0x048200, 0x048201, 0x048208, 0x048209, 0x048240, 0x048241, 0x048248, 0x048249,
  0x049000, 0x049001, 0x049008, 0x049009, 0x049040, 0x049041, 0x049048, 0x049049, 0x049200, 0x049201, 0x049208, 0x049209, 0x049240, 0x049241, 0x049248, 0x049249,
  0x200000, 0x200001, 0x200008, 0x200009, 0x200040, 0x200041, 0x200048, 0x200049, 0x200200, 0x200201, 0x200208, 0x200209, 0x200240, 0x200241, 0x200248, 0x200249,
  0x201000, 0x201001, 0x201008, 0x201009, 0x201040, 0x201041, 0x201048, 0x201049, 0x201200, 0x201201, 0x201208, 0x201209, 0x201240, 0x201241, 0x201248, 0x201249,
  0x208000, 0x208001, 0x208008, 0x208009, 0x208040, 0x208041, 0x208048, 0x208049, 0x208200, 0x208201, 0x208208, 0x208209, 0x208240, 0x208241, 0x208248, 0x208249,
  0x209000, 0x209001, 0x209008, 0x209009, 0x209040, 0x209041, 0x209048, 0x209049, 0x209200, 0x209201, 0x209208, 0x209209, 0x209240, 0x209241, 0x209248, 0x209249,
  0x240000, 0x240001, 0x240008, 0x240009, 0x240040, 0x240041, 0x240048, 0x240049, 0x240200, 0x240201, 0x240208, 0x240209, 0x240240, 0x240241, 0x240248, 0x240249,
  0x241000, 0x241001, 0x241008, 0x241009, 0x241040, 0x241041, 0x241048, 0x241049, 0x241200, 0x241201, 0x241208, 0x241209, 0x241240, 0x241241, 0x241248, 0x241249,
  0x248000, 0x248001, 0x248008, 0x248009, 0x248040, 0x248041, 0x248048, 0x248049, 0x248200, 0x248201, 0x248208, 0x248209, 0x248240, 0x248241, 0x248248, 0x248249,
  0x249000, 0x249001, 0x249008, 0x249009, 0x249040, 0x249041, 0x249048, 0x249049, 0x249200, 0x249201, 0x249208, 0x249209, 0x249240, 0x249241, 0x249248, 0x249249,
};

#define INT2SFC(_v_)  ((box_t) int2sfc[_v_])


static box_t sfc_BOX_GET_X(box_t b, int x)
{
  box_t i = BOX_BITS;
  box_t v = 0;


  b >>= x;

  while(i > 0)
  {
    --i;
    v |= ((b >> (3 * i)) & BOX_CONST(1)) << i;
  }

  return v;
}


static box_t sfc_BOX_SET(box_t v0, box_t v1, box_t v2)
{
  box_t b = 0;


  b |= (INT2SFC((v0 >> (0 * 8)) & 0xFF) << (0 + (0 * 24))) | (INT2SFC((v1 >> (0 * 8)) & 0xFF) << (1 + (0 * 24))) | (INT2SFC((v2 >> (0 * 8)) & 0xFF) << (2 + (0 * 24)));
  b |= (INT2SFC((v0 >> (1 * 8)) & 0xFF) << (0 + (1 * 24))) | (INT2SFC((v1 >> (1 * 8)) & 0xFF) << (1 + (1 * 24))) | (INT2SFC((v2 >> (1 * 8)) & 0xFF) << (2 + (1 * 24)));
  b |= (INT2SFC((v0 >> (2 * 8)) & 0x1F) << (0 + (2 * 24))) | (INT2SFC((v1 >> (2 * 8)) & 0x1F) << (1 + (2 * 24))) | (INT2SFC((v2 >> (2 * 8)) & 0x1F) << (2 + (2 * 24)));

  return b;
}
#endif


void fcs_near_param_create(fcs_near_param_t *near_param)
{
#if FCS_NEAR_OCL
  near_param->ocl = 0;
  near_param->ocl_conf[0] = '\0';
#endif /* FCS_NEAR_OCL */
#if FCS_NEAR_OCL_SORT
  near_param->ocl_sort = 0;
  near_param->ocl_sort_algo = 0;
#endif /* FCS_NEAR_OCL_SORT */
}


void fcs_near_param_destroy(fcs_near_param_t *near_param)
{

}


void fcs_near_param_set_param(fcs_near_param_t *near_param, fcs_near_param_t *param)
{
  *near_param = *param;
}


#if HAVE_OPENCL

fcs_int fcs_near_param_set_ocl(fcs_near_param_t *near_param, fcs_int ocl)
{
#if FCS_NEAR_OCL
  near_param->ocl = ocl;

  return 0;
#else /* FCS_NEAR_OCL */
  return 1;
#endif /* FCS_NEAR_OCL */
}


fcs_int fcs_near_param_set_ocl_conf(fcs_near_param_t *near_param, const char *ocl_conf)
{
#if FCS_NEAR_OCL
  strncpy(near_param->ocl_conf, ocl_conf, FCS_NEAR_PARAM_OCL_CONF_SIZE);

  return 0;
#else /* FCS_NEAR_OCL */
  return 1;
#endif /* FCS_NEAR_OCL */
}


fcs_int fcs_near_param_set_ocl_sort(fcs_near_param_t *near_param, fcs_int ocl_sort)
{
#if FCS_NEAR_OCL_SORT
  near_param->ocl_sort = ocl_sort;

  return 0;
#else /* FCS_NEAR_OCL_SORT */
  return 1;
#endif /* FCS_NEAR_OCL_SORT */
}

fcs_int fcs_near_param_set_ocl_sort_algo(fcs_near_param_t *near_param, fcs_int ocl_sort_algo)
{
#if FCS_NEAR_OCL_SORT
  near_param->ocl_sort_algo = ocl_sort_algo;

  return 0;
#else /* FCS_NEAR_OCL_SORT */
  return 1;
#endif /* FCS_NEAR_OCL_SORT */
}

#endif /* HAVE_OPENCL */


void fcs_near_create(fcs_near_t *near)
{
  near->compute_field = NULL;
  near->compute_potential = NULL;
  near->compute_field_potential = NULL;

  near->compute_field_3diff = NULL;
  near->compute_potential_3diff = NULL;
  near->compute_field_potential_3diff = NULL;

  near->compute_loop = NULL;

  near->compute_field_potential_source = NULL;
  near->compute_field_potential_function = NULL;

  near->compute_param_size = 0;

  near->box_base[0] = near->box_base[1] = near->box_base[2] = 0;
  near->box_a[0] = near->box_a[1] = near->box_a[2] = 0;
  near->box_b[0] = near->box_b[1] = near->box_b[2] = 0;
  near->box_c[0] = near->box_c[1] = near->box_c[2] = 0;
  near->periodicity[0] = near->periodicity[1] = near->periodicity[2] = -1;

  near->nparticles = near->max_nparticles = 0;
  near->positions = NULL;
  near->charges = NULL;
  near->indices = NULL;
  near->field = NULL;
  near->potentials = NULL;

  near->nghosts = 0;
  near->ghost_positions = NULL;
  near->ghost_charges = NULL;
  near->ghost_indices = NULL;

  near->max_particle_move = -1;

  near->resort = 0;
  near->gridsort_resort = FCS_GRIDSORT_RESORT_NULL;

  fcs_near_param_create(&near->near_param);

  near->context = NULL;
}


void fcs_near_destroy(fcs_near_t *near)
{
  near->compute_field = NULL;
  near->compute_potential = NULL;
  near->compute_field_potential = NULL;

  near->compute_field_3diff = NULL;
  near->compute_potential_3diff = NULL;
  near->compute_field_potential_3diff = NULL;

  near->compute_loop = NULL;

  if (near->compute_field_potential_source) free(near->compute_field_potential_source);
  near->compute_field_potential_source = NULL;
  if (near->compute_field_potential_function) free(near->compute_field_potential_function);
  near->compute_field_potential_function = NULL;

  near->compute_param_size = 0;

  near->box_base[0] = near->box_base[1] = near->box_base[2] = 0;
  near->box_a[0] = near->box_a[1] = near->box_a[2] = 0;
  near->box_b[0] = near->box_b[1] = near->box_b[2] = 0;
  near->box_c[0] = near->box_c[1] = near->box_c[2] = 0;
  near->periodicity[0] = near->periodicity[1] = near->periodicity[2] = -1;

  near->nparticles = near->max_nparticles = 0;
  near->positions = NULL;
  near->charges = NULL;
  near->indices = NULL;
  near->field = NULL;
  near->potentials = NULL;

  near->nghosts = 0;
  near->ghost_positions = NULL;
  near->ghost_charges = NULL;
  near->ghost_indices = NULL;

  fcs_gridsort_resort_destroy(&near->gridsort_resort);

  fcs_near_param_destroy(&near->near_param);

  near->context = NULL;
}


void fcs_near_set_param(fcs_near_t *near, fcs_near_param_t *near_param)
{
  fcs_near_param_set_param(&near->near_param, near_param);
}


void fcs_near_set_potential(fcs_near_t *near, fcs_near_potential_f compute_potential)
{
  near->compute_potential = compute_potential;
}


void fcs_near_set_field(fcs_near_t *near, fcs_near_field_f compute_field)
{
  near->compute_field = compute_field;
}


void fcs_near_set_field_potential(fcs_near_t *near, fcs_near_field_potential_f compute_field_potential)
{
  near->compute_field_potential = compute_field_potential;
}


void fcs_near_set_potential_3diff(fcs_near_t *near, fcs_near_potential_3diff_f compute_potential_3diff)
{
  near->compute_potential_3diff = compute_potential_3diff;
}


void fcs_near_set_field_3diff(fcs_near_t *near, fcs_near_field_3diff_f compute_field_3diff)
{
  near->compute_field_3diff = compute_field_3diff;
}


void fcs_near_set_field_potential_3diff(fcs_near_t *near, fcs_near_field_potential_3diff_f compute_field_potential_3diff)
{
  near->compute_field_potential_3diff = compute_field_potential_3diff;
}


void fcs_near_set_loop(fcs_near_t *near, fcs_near_loop_f compute_loop)
{
  near->compute_loop = compute_loop;
}


void fcs_near_set_compute_param_size(fcs_near_t *near, fcs_int compute_param_size)
{
  near->compute_param_size = compute_param_size;
}


void fcs_near_set_field_potential_source(fcs_near_t *near, const char *compute_field_potential_source, const char *compute_field_potential_function)
{
  if (near->compute_field_potential_source) free(near->compute_field_potential_source);
  near->compute_field_potential_source = (compute_field_potential_source)?strdup(compute_field_potential_source):NULL;

  if (near->compute_field_potential_function) free(near->compute_field_potential_function);
  near->compute_field_potential_function = (compute_field_potential_function)?strdup(compute_field_potential_function):NULL;
}


void fcs_near_set_system(fcs_near_t *near, const fcs_float *box_base, const fcs_float *box_a, const fcs_float *box_b, const fcs_float *box_c, const fcs_int *periodicity)
{
  fcs_int i;


  for (i = 0; i < 3; ++i)
  {
    near->box_base[i] = box_base[i];
    near->box_a[i] = box_a[i];
    near->box_b[i] = box_b[i];
    near->box_c[i] = box_c[i];

    if (periodicity) near->periodicity[i] = periodicity[i];
  }
}


void fcs_near_set_particles(fcs_near_t *near, fcs_int nparticles, fcs_int max_nparticles, fcs_float *positions, fcs_float *charges, fcs_gridsort_index_t *indices, fcs_float *field, fcs_float *potentials)
{
  near->nparticles = nparticles;
  near->max_nparticles = max_nparticles;
  near->positions = positions;
  near->charges = charges;
  near->indices = indices;
  near->field = field;
  near->potentials = potentials;
}


void fcs_near_set_ghosts(fcs_near_t *near, fcs_int nghosts, fcs_float *positions, fcs_float *charges, fcs_gridsort_index_t *indices)
{
  near->nghosts = nghosts;
  near->ghost_positions = positions;
  near->ghost_charges = charges;
  near->ghost_indices = indices;
}


void fcs_near_set_max_particle_move(fcs_near_t *near, fcs_float max_particle_move)
{
  near->max_particle_move = max_particle_move;
}


void fcs_near_set_resort(fcs_near_t *near, fcs_int resort)
{
  near->resort = resort;
}


#if PRINT_PARTICLES
static void print_particles(fcs_int n, fcs_float *xyz, int size, int rank, MPI_Comm comm)
{
  const int root = 0;
  fcs_int max_n, i, j;
  fcs_float *in_xyz;
  MPI_Status status;
  int in_count;


  MPI_Reduce(&n, &max_n, 1, FCS_MPI_INT, MPI_MAX, root, comm);

  in_xyz = malloc(max_n * 3 * sizeof(fcs_float));

  if (rank == root)
  {
    for (i = 0; i < size; ++i)
    {
      if (i == root) MPI_Sendrecv(xyz, 3 * n, FCS_MPI_FLOAT, root, 0, in_xyz, 3 * max_n, FCS_MPI_FLOAT, root, 0, comm, &status);
      else MPI_Recv(in_xyz, 3 * max_n, FCS_MPI_FLOAT, i, 0, comm, &status);
      MPI_Get_count(&status, FCS_MPI_FLOAT, &in_count);

      in_count /= 3;

      for (j = 0; j < in_count; ++j) printf("%" FCS_LMOD_INT "d  %" FCS_LMOD_FLOAT "f  %" FCS_LMOD_FLOAT "f  %" FCS_LMOD_FLOAT "f\n", i, in_xyz[j * 3 + 0], in_xyz[j * 3 + 1], in_xyz[j * 3 + 2]);
    }

  } else
  {
    MPI_Send(xyz, 3 * n, FCS_MPI_FLOAT, root, 0, comm);
  }

  free(in_xyz);
}
#endif


static void create_boxes(fcs_int nlocal, box_t *boxes, fcs_float *positions, fcs_gridsort_index_t *indices, fcs_float *box_base, fcs_float *box_a, fcs_float *box_b, fcs_float *box_c, fcs_int *periodicity, fcs_float cutoff)
{
  fcs_int i;
  fcs_int with_periodic;
  fcs_float icutoff, base[3];


  with_periodic = (periodicity[0] || periodicity[1] || periodicity[2]);

  icutoff = 1.0 / cutoff;

  base[0] = box_base[0] - 2 * cutoff;
  base[1] = box_base[1] - 2 * cutoff;
  base[2] = box_base[2] - 2 * cutoff;

  if (with_periodic) fcs_gridsort_unfold_periodic_particles(nlocal, indices, positions, box_a, box_b, box_c);

  for (i = 0; i < nlocal; ++i)
  {
    boxes[i] = BOX_SET((int) ((positions[3 * i + 0] - base[0]) * icutoff), (int) ((positions[3 * i + 1] - base[1]) * icutoff), (int) ((positions[3 * i + 2] - base[2]) * icutoff));
  }

  /* sentinel with max. box number */
  boxes[nlocal] = BOX_SET(BOX_MASK, BOX_MASK, BOX_MASK);
}


#if PRINT_BOX_STATS

static void print_box_stats(fcs_int nlocal, fcs_float *positions, fcs_float *box_base, fcs_float *box_a, fcs_float *box_b, fcs_float *box_c, fcs_float cutoff)
{
  fcs_float icutoff;
  fcs_int i, dim_boxes[3], nboxes, *box_stats, b[3];


  icutoff = 1.0 / cutoff;

  dim_boxes[0] = (int) ((box_a[0] - box_base[0]) * icutoff) + 1;
  dim_boxes[1] = (int) ((box_b[1] - box_base[1]) * icutoff) + 1;
  dim_boxes[2] = (int) ((box_c[2] - box_base[2]) * icutoff) + 1;

  nboxes = dim_boxes[0] * dim_boxes[1] * dim_boxes[2];
  box_stats = malloc(nboxes * sizeof(fcs_int));

  for (i = 0; i < nboxes; ++i) box_stats[i] = 0;

  for (i = 0; i < nlocal; ++i)
  {
    b[0] = (int) ((positions[3 * i + 0] - box_base[0]) * icutoff);
    b[1] = (int) ((positions[3 * i + 1] - box_base[1]) * icutoff);
    b[2] = (int) ((positions[3 * i + 2] - box_base[2]) * icutoff);

    b[0] = z_minmax(0, b[0], dim_boxes[0] - 1);
    b[1] = z_minmax(0, b[1], dim_boxes[1] - 1);
    b[2] = z_minmax(0, b[2], dim_boxes[2] - 1);

    ++box_stats[(b[2] * dim_boxes[1] + b[1]) * dim_boxes[0] + b[0]];
  }

  fcs_int nboxes_empty = 0;
  fcs_int nppb_min = nlocal;
  fcs_int nppb_max = 0;
  fcs_float nppb_avg = (fcs_float) nlocal / nboxes;
  fcs_float nppb_std = 0;
  for (i = 0; i < nboxes; ++i)
  {
    if (box_stats[i] == 0) ++nboxes_empty;

    nppb_min = z_min(box_stats[i], nppb_min);
    nppb_max = z_max(box_stats[i], nppb_max);
    nppb_std += (box_stats[i] - nppb_avg) * (box_stats[i] - nppb_avg);
  }
  nppb_std = fcs_sqrt(nppb_std / nboxes);

  printf("box statistics:\n");
  printf("  particles: %" FCS_LMOD_INT "d\n", nlocal);
  printf("  boxes: %" FCS_LMOD_INT "d x %" FCS_LMOD_INT "d x %" FCS_LMOD_INT "d = %" FCS_LMOD_INT "d\n", dim_boxes[0], dim_boxes[1], dim_boxes[2], nboxes);
  printf("  empty boxes: %" FCS_LMOD_INT "d = %f%%\n", nboxes_empty, 100.0 * nboxes_empty / nboxes);
  printf("  particles per box:\n");
  printf("    minimum: %" FCS_LMOD_INT "d\n", nppb_min);
  printf("    maximum: %" FCS_LMOD_INT "d\n", nppb_max);
  printf("    average: %" FCS_LMOD_FLOAT "f\n", nppb_avg);
  printf("    standard deviation: %" FCS_LMOD_FLOAT "f\n", nppb_std);
}

#endif


static void sort_into_boxes(fcs_int nlocal, box_t *boxes, fcs_float *positions, fcs_float *charges, fcs_gridsort_index_t *indices, fcs_float *field, fcs_float *potentials)
{
  fcs_near_fp_elements_t s0, sx0;
  fcs_near_f__elements_t s1, sx1;
  fcs_near__p_elements_t s2, sx2;
  fcs_near____elements_t s3, sx3;


  if (field && potentials)
  {
    fcs_near_fp_elem_set_size(&s0, nlocal);
    fcs_near_fp_elem_set_max_size(&s0, nlocal);
    fcs_near_fp_elem_set_keys(&s0, boxes);
    fcs_near_fp_elem_set_data(&s0, positions, charges, indices, field, potentials);

    fcs_near_fp_elements_alloc(&sx0, s0.size, SLCM_ALL);

    fcs_near_fp_sort_radix(&s0, &sx0, -1, -1, -1);

    fcs_near_fp_elements_free(&sx0);
  }

  if (field && potentials == NULL)
  {
    fcs_near_f__elem_set_size(&s1, nlocal);
    fcs_near_f__elem_set_max_size(&s1, nlocal);
    fcs_near_f__elem_set_keys(&s1, boxes);
    fcs_near_f__elem_set_data(&s1, positions, charges, indices, field);

    fcs_near_f__elements_alloc(&sx1, s1.size, SLCM_ALL);

    fcs_near_f__sort_radix(&s1, &sx1, -1, -1, -1);

    fcs_near_f__elements_free(&sx1);
  }

  if (field == NULL && potentials)
  {
    fcs_near__p_elem_set_size(&s2, nlocal);
    fcs_near__p_elem_set_max_size(&s2, nlocal);
    fcs_near__p_elem_set_keys(&s2, boxes);
    fcs_near__p_elem_set_data(&s2, positions, charges, indices, potentials);

    fcs_near__p_elements_alloc(&sx2, s2.size, SLCM_ALL);

    fcs_near__p_sort_radix(&s2, &sx2, -1, -1, -1);

    fcs_near__p_elements_free(&sx2);
  }

  if (field == NULL && potentials == NULL)
  {
    fcs_near____elem_set_size(&s3, nlocal);
    fcs_near____elem_set_max_size(&s3, nlocal);
    fcs_near____elem_set_keys(&s3, boxes);
    fcs_near____elem_set_data(&s3, positions, charges, indices);

    fcs_near____elements_alloc(&sx3, s3.size, SLCM_ALL);

    fcs_near____sort_radix(&s3, &sx3, -1, -1, -1);

    fcs_near____elements_free(&sx3);
  }
}


#ifdef BOX_SKIP_FORMAT
static void make_boxes_skip_format(fcs_int nlocal, box_t *boxes)
{
  fcs_int i, j, bs, h;
  box_t b;


  bs = 0;
  b = -1;

  for (i = 0; i < nlocal; ++i)
  {
    if (b == boxes[i]) continue;

    h = (i - bs) / 2;
    for (j = 0; j < h; ++j)
    {
      boxes[i - 1 - j] = -(i - bs - 1 - j);
      boxes[bs + 1 + j] = -(i - bs - 1 - j);
    }

    bs = i;
    b = boxes[i];
  }

  h = (i - bs) / 2;
  for (j = 0; j < h; ++j)
  {
    boxes[i - 1 - j] = -(i - bs - 1 - j);
    boxes[bs + 1 + j] = -(i - bs - 1 - j);
  }
}
#endif


#if PRINT_PARTICLES
static void print_boxes(fcs_int nlocal, box_t *boxes)
{
  fcs_int i;


  for (i = 0; i < nlocal; ++i)
  {
    printf("%5" FCS_LMOD_INT "d: " box_fmt "\n", i, box_val(boxes[i]));
  }
}
#endif


static void find_box(box_t *boxes, fcs_int max, box_t box, fcs_int low, fcs_int *start, fcs_int *size)
{
/*  printf("find_box: " box_fmt " @ %" FCS_LMOD_INT "d\n", box_val(box), low);*/

#ifdef BOX_SKIP_FORMAT
# define BOX_NEXT(_b_, _l_)      (((_b_)[(_l_) + 1] >= 0)?((_l_) + 1):((_l_) - (_b_)[(_l_) + 1] + 1))
# define BOX_PREV(_b_, _l_)      (((_b_)[(_l_) - 1] >= 0)?((_l_) - 1):((_l_) + (_b_)[(_l_) - 1] - 1))
# define BOX_NUM_BACK(_b_, _l_)  (((_b_)[_l_] >= 0)?((_b_)[_l_]):((_b_)[(_l_) + (_b_)[_l_]]))
#else
# define BOX_NEXT(_b_, _l_)      ((_l_) + 1)
# define BOX_PREV(_b_, _l_)      ((_l_) - 1)
# define BOX_NUM_BACK(_b_, _l_)  (_b_)[_l_]
#endif

  /* backward search for start  */
  while (low > 0 && BOX_NUM_BACK(boxes, low - 1) >= box) low = BOX_PREV(boxes, low);

  /* forward search for start  */
  while (low < max && boxes[low] < box) low = BOX_NEXT(boxes, low);

  *start = low;

  /* search for end */
  while (low < max && boxes[low] == box) low = BOX_NEXT(boxes, low);

  *size = low - *start;

/*  printf("  -> %" FCS_LMOD_INT "d / %" FCS_LMOD_INT "d\n", *start, *size);*/
}


static void find_neighbours(fcs_int nneighbours, fcs_int *neighbours, box_t *boxes, fcs_int max, box_t box, fcs_int *lasts, fcs_int *starts, fcs_int *sizes)
{
  fcs_int i;
  box_t nbox;


  for (i = 0; i < nneighbours; ++i)
  {
    nbox = BOX_ADD(box, neighbours[3 * i + 0], neighbours[3 * i + 1], neighbours[3 * i + 2]);

    find_box(boxes, max, nbox, lasts[i], &starts[i], &sizes[i]);

/*    printf("  neighbour %" FCS_LMOD_INT "d: " box_fmt " -> %" FCS_LMOD_INT "d,%" FCS_LMOD_INT "d,%" FCS_LMOD_INT "d\n", i, box_val(nbox), lasts[i], starts[i], sizes[i]);*/
  }
}


#if FCS_NEAR_OCL

static fcs_int count_boxes(fcs_int nparticles, box_t *boxes)
{
  box_t current_box;
  fcs_int current_last, current_start, current_size;
  fcs_int nboxes = 0;

  current_last = 0;
  do {
    current_box = boxes[current_last];
    find_box(boxes, nparticles, current_box, current_last, &current_start, &current_size);
    ++nboxes;
    current_last = current_start + current_size;

  } while (current_last < nparticles);

  return nboxes;
}


static const char *fcs_ocl_near_cl_compute_config =
#if POTENTIAL_CONST1
  "#define POTENTIAL_CONST1  1\n"
#else
  "#define POTENTIAL_CONST1  0\n"
#endif
#if COMPUTE
  "#define COMPUTE  1\n"
#else
  "#define COMPUTE  0\n"
#endif
#if COMPUTE_BOX
  "#define COMPUTE_BOX  1\n"
#else
  "#define COMPUTE_BOX  0\n"
#endif
#if COMPUTE_REAL_NEIGHBOURS
  "#define COMPUTE_REAL_NEIGHBOURS  1\n"
#else
  "#define COMPUTE_REAL_NEIGHBOURS  0\n"
#endif
#if COMPUTE_GHOST_NEIGHBOURS
  "#define COMPUTE_GHOST_NEIGHBOURS  1\n"
#else
  "#define COMPUTE_GHOST_NEIGHBOURS  0\n"
#endif
#if COMPUTE_VERBOSE
  "#define COMPUTE_VERBOSE  1\n"
#else
  "#define COMPUTE_VERBOSE  0\n"
#endif
  ;

static const char *fcs_ocl_near_cl_compute_coulomb = "fcs_ocl_near_coulomb_field_potential";

static const char *fcs_ocl_near_cl_compute =
#include "near.cl_str.h"
  ;

static fcs_int fcs_ocl_near_init(fcs_ocl_context_t *ocl, fcs_int nunits, fcs_ocl_unit_t *units, const char *nfp_source, const char *nfp_function, int comm_rank)
{
  cl_device_id device_id;
  fcs_ocl_get_device(&units[0], comm_rank, &device_id);

#if FCS_NEAR_OCL_CPU_CUS
  // calculate how to partition the device
  // get total compute units
  cl_uint cus;
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cus), &cus, NULL));

  // check for boundries
  if(FCS_NEAR_OCL_CPU_CUS > cus) {
    // fail
    printf("error: trying to use %d CUs, only %d available\n", FCS_NEAR_OCL_CPU_CUS, cus);
    abort();
  }

  // make list with desired CU count for property parameter
  const cl_device_partition_property properties[] = {CL_DEVICE_PARTITION_BY_COUNTS, FCS_NEAR_OCL_CPU_CUS, CL_DEVICE_PARTITION_BY_COUNTS_LIST_END};

  cl_device_id sub_device_id;
  CL_CHECK(clCreateSubDevices(device_id, properties, 1, &sub_device_id, NULL));

  device_id = sub_device_id;

  INFO_CMD(printf(INFO_PRINT_PREFIX "only use %d of %d CUs\n", FCS_NEAR_OCL_CPU_CUS, cus););
#endif // FCS_NEAR_OCL_CPU_CUS

#if FCS_NEAR_OCL_SORT
  // save device id for later usage
  ocl->device_id = device_id;

  // query local memory size
  clGetDeviceInfo(device_id, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(ocl->local_memory), &ocl->local_memory, NULL);
  clGetDeviceInfo(device_id, CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(ocl->global_memory), &ocl->global_memory, NULL);

  // query base address
  CL_CHECK(clGetDeviceInfo(device_id, CL_DEVICE_MEM_BASE_ADDR_ALIGN , sizeof(ocl->base_addr_align), &ocl->base_addr_align, NULL));

  // set values to initial settings
  ocl->buffers_on_device = 0;
  ocl->buffers_on_device_ghost = 0;
  ocl->is_ghosts = 0;
#endif

  cl_int ret;

  ocl->context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);
  if (ret != CL_SUCCESS) return 1;

  const char *sources[] = {
    fcs_ocl_cl_config,
    fcs_ocl_cl,
    fcs_ocl_math_cl,
    fcs_ocl_near_cl_compute_config,
    "#define _nfp_  ", fcs_ocl_near_cl_compute_coulomb, "\n",
    "",
    fcs_ocl_near_cl_compute
  };
  const cl_uint nsources = sizeof(sources) / sizeof(sources[0]);

  if (nfp_source && nfp_function)
  {
    sources[5] = nfp_function;
    sources[7] = nfp_source;
  }

  ocl->program = clCreateProgramWithSource(ocl->context, nsources, sources, NULL, &ret);
  if (ret != CL_SUCCESS) return 1;  // FIXME: CL_CHECK_ERR?

  ret = clBuildProgram(ocl->program, 1, &device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS)
  {
    size_t length;
    char buffer[32*1024];
    clGetProgramBuildInfo(ocl->program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
    printf("clGetProgramBuildInfo: %.*s\n", (int) length, buffer);
    return 1;
  }

  ocl->compute_kernel = clCreateKernel(ocl->program, "fcs_ocl_near_compute", &ret);
  if (ret != CL_SUCCESS) return 1;
#ifdef DO_TIMING
  ocl->command_queue = clCreateCommandQueue(ocl->context, device_id, CL_QUEUE_PROFILING_ENABLE , &ret);
#else
  ocl->command_queue = clCreateCommandQueue(ocl->context, device_id, 0, &ret);
#endif
  if (ret != CL_SUCCESS) return 1;

  return 0;
}


static fcs_int fcs_ocl_near_release(fcs_ocl_context_t *ocl)
{
  cl_int ret;

  ret = clReleaseCommandQueue(ocl->command_queue);
  if (ret != CL_SUCCESS) return 1;

  ret = clReleaseKernel(ocl->compute_kernel);
  if (ret != CL_SUCCESS) return 1;

  ret = clReleaseProgram(ocl->program);
  if (ret != CL_SUCCESS) return 1;

  ret = clReleaseContext(ocl->context);
  if (ret != CL_SUCCESS) return 1;

  return 0;
}


static fcs_int fcs_ocl_compute_near_prepare(fcs_ocl_context_t *ocl, fcs_float cutoff, const void *compute_param, fcs_int compute_param_size,
  fcs_int nparticles, fcs_float *positions, fcs_float *charges, fcs_float *field, fcs_float *potentials,
  fcs_int nghosts, fcs_float *gpositions, fcs_float *gcharges)
{
#if FCS_NEAR_OCL_WAIT_WRITE
  int nwrite_events = 0;
  cl_event write_events[4];
# define WRITE_EVENT  &write_events[nwrite_events++]
#else /* FCS_NEAR_OCL_WAIT_WRITE */
# define WRITE_EVENT  NULL
#endif /* FCS_NEAR_OCL_WAIT_WRITE */

  if (compute_param_size > 0)
  {
    ocl->mem_param  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, compute_param_size, (void *) compute_param, &_err));

    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_param, CL_FALSE, 0, compute_param_size, compute_param, 0, NULL, NULL));

  } else ocl->mem_param = NULL;

#if FCS_NEAR_OCL_SORT_KEEP_BUFFERS
  // make sure the buffers are on the device
  if(ocl->buffers_on_device != 1)
#endif // FCS_NEAR_OCL_SORT_KEEP_BUFFERS
  {
    // but now write to device
    ocl->mem_positions  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, nparticles * 3 * sizeof(fcs_float), positions, &_err));
    ocl->mem_charges    = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, nparticles * sizeof(fcs_float), charges, &_err));
    ocl->mem_field      = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, nparticles * 3 * sizeof(fcs_float), field, &_err));
    ocl->mem_potentials = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, nparticles * sizeof(fcs_float), potentials, &_err));

    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_positions,  CL_FALSE, 0, nparticles * 3 * sizeof(fcs_float), positions, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_charges,    CL_FALSE, 0, nparticles * sizeof(fcs_float), charges, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_field,      CL_FALSE, 0, nparticles * 3 * sizeof(fcs_float), field, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_potentials, CL_FALSE, 0, nparticles * sizeof(fcs_float), potentials, 0, NULL, NULL));
  }

  ocl->mem_box_info = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, ocl->nboxes * 6 * sizeof(fcs_int), ocl->box_info, &_err));
  CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_box_info, CL_FALSE, 0, ocl->nboxes * 6 * sizeof(fcs_int), ocl->box_info, 0, NULL, WRITE_EVENT));

  if (ocl->nreal_neighbour_boxes > 0)
  {
    ocl->mem_real_neighbour_boxes = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, ocl->nreal_neighbour_boxes * 2 * sizeof(fcs_int), ocl->real_neighbour_boxes, &_err));
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_real_neighbour_boxes, CL_FALSE, 0, ocl->nreal_neighbour_boxes * 2 * sizeof(fcs_int), ocl->real_neighbour_boxes, 0, NULL, WRITE_EVENT));

  } else ocl->mem_real_neighbour_boxes = NULL;

  if (nghosts > 0)
  {
#if FCS_NEAR_OCL_SORT_KEEP_BUFFERS
    if(ocl->buffers_on_device_ghost != 1)
#endif // FCS_NEAR_OCL_SORT_KEEP_BUFFERS
    {
      // write ghost buffers to device
      ocl->mem_gpositions = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, nghosts * 3 * sizeof(fcs_float), gpositions, &_err));
      ocl->mem_gcharges   = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, nghosts * sizeof(fcs_float), gcharges, &_err));

      CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_gpositions, CL_FALSE, 0, nghosts * 3 * sizeof(fcs_float), gpositions, 0, NULL, NULL));
      CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_gcharges, CL_FALSE, 0, nghosts * sizeof(fcs_float), gcharges, 0, NULL, WRITE_EVENT));
    }
    if (ocl->nghost_neighbour_boxes > 0)
    {
      ocl->mem_ghost_neighbour_boxes = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY | CL_MEM_USE_HOST_PTR, ocl->nghost_neighbour_boxes * 2 * sizeof(fcs_int), ocl->ghost_neighbour_boxes, &_err));
      CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_ghost_neighbour_boxes, CL_FALSE, 0, ocl->nghost_neighbour_boxes * 2 * sizeof(fcs_int), ocl->ghost_neighbour_boxes, 0, NULL, WRITE_EVENT));

    } else ocl->mem_ghost_neighbour_boxes = NULL;
  }

#if FCS_NEAR_OCL_WAIT_WRITE
  CL_CHECK(clWaitForEvents(nwrite_events, write_events));
  while (nwrite_events > 0) CL_CHECK(clReleaseEvent(write_events[--nwrite_events]));
#endif /* FCS_NEAR_OCL_WAIT_WRITE */

  CL_CHECK(clSetKernelArg(ocl->compute_kernel, 0, sizeof(cutoff), &cutoff));
  CL_CHECK(clSetKernelArg(ocl->compute_kernel, 1, sizeof(cl_mem), &ocl->mem_param));
  CL_CHECK(clSetKernelArg(ocl->compute_kernel, 2, sizeof(cl_mem), &ocl->mem_positions));
  CL_CHECK(clSetKernelArg(ocl->compute_kernel, 3, sizeof(cl_mem), &ocl->mem_charges));
  CL_CHECK(clSetKernelArg(ocl->compute_kernel, 4, sizeof(cl_mem), &ocl->mem_field));
  CL_CHECK(clSetKernelArg(ocl->compute_kernel, 5, sizeof(cl_mem), &ocl->mem_potentials));
  CL_CHECK(clSetKernelArg(ocl->compute_kernel, 6, sizeof(cl_mem), &ocl->mem_box_info));
  CL_CHECK(clSetKernelArg(ocl->compute_kernel, 7, sizeof(cl_mem), &ocl->mem_real_neighbour_boxes));
  if (nghosts > 0)
  {
    CL_CHECK(clSetKernelArg(ocl->compute_kernel, 8, sizeof(cl_mem), &ocl->mem_gpositions));
    CL_CHECK(clSetKernelArg(ocl->compute_kernel, 9, sizeof(cl_mem), &ocl->mem_gcharges));
    CL_CHECK(clSetKernelArg(ocl->compute_kernel, 10, sizeof(cl_mem), &ocl->mem_ghost_neighbour_boxes));

  } else
  {
    CL_CHECK(clSetKernelArg(ocl->compute_kernel, 8, sizeof(cl_mem), NULL));
    CL_CHECK(clSetKernelArg(ocl->compute_kernel, 9, sizeof(cl_mem), NULL));
    CL_CHECK(clSetKernelArg(ocl->compute_kernel, 10, sizeof(cl_mem), NULL));
  }

  return 0;
}

static fcs_int fcs_ocl_compute_near_start(fcs_ocl_context_t *ocl)
{
  size_t global_work_size[1] = { ocl->nboxes };

  CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->compute_kernel, 1, NULL, global_work_size, NULL, 0, NULL, &ocl->kernel_completion));

  return 0;
}

static fcs_int fcs_ocl_compute_near_join(fcs_ocl_context_t *ocl)
{
  CL_CHECK(clWaitForEvents(1, &ocl->kernel_completion));

  CL_CHECK(clReleaseEvent(ocl->kernel_completion));

  return 0;
}

static fcs_int fcs_ocl_compute_near_release(fcs_ocl_context_t *ocl, fcs_int nparticles, fcs_float *field, fcs_float *potentials, fcs_int nghosts)
{
  CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_field, CL_FALSE, 0, nparticles * 3 * sizeof(fcs_float), field, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_potentials, CL_TRUE, 0, nparticles * sizeof(fcs_float), potentials, 0, NULL, NULL));

#if 0
  CL_CHECK(clFlush(ocl->command_queue));
  CL_CHECK(clFinish(ocl->command_queue));
#endif

  if (ocl->mem_param) CL_CHECK(clReleaseMemObject(ocl->mem_param));

  CL_CHECK(clReleaseMemObject(ocl->mem_positions));
  CL_CHECK(clReleaseMemObject(ocl->mem_charges));
  CL_CHECK(clReleaseMemObject(ocl->mem_field));
  CL_CHECK(clReleaseMemObject(ocl->mem_potentials));

  CL_CHECK(clReleaseMemObject(ocl->mem_box_info));
  if (ocl->mem_real_neighbour_boxes) CL_CHECK(clReleaseMemObject(ocl->mem_real_neighbour_boxes));

  if (nghosts > 0)
  {
    CL_CHECK(clReleaseMemObject(ocl->mem_gpositions));
    CL_CHECK(clReleaseMemObject(ocl->mem_gcharges));

    if (ocl->mem_ghost_neighbour_boxes) CL_CHECK(clReleaseMemObject(ocl->mem_ghost_neighbour_boxes));
  }

  return 0;
}

#endif /* FCS_NEAR_OCL */


static void compute_near(fcs_float *positions0, fcs_float *charges0, fcs_float *field0, fcs_float *potentials0, fcs_int start0, fcs_int size0,
                         fcs_float *positions1, fcs_float *charges1, fcs_int start1, fcs_int size1, fcs_float cutoff, fcs_near_t *near, const void *near_param)
{
  FCS_NEAR_LOOP_HEAD();


  if (near->compute_loop)
  {
    near->compute_loop(positions0, charges0, field0, potentials0, start0, size0, positions1, charges1, start1, size1, cutoff, near_param);
    return;
  }

/*  printf("compute: %" FCS_LMOD_INT "d / %" FCS_LMOD_INT "d vs. %" FCS_LMOD_INT "d / %" FCS_LMOD_INT "d\n", start0, size0, start1, size1);*/

  if (near->compute_field_potential)
  {
    if (field0 && potentials0)
      FCS_NEAR_LOOP_BODY_FP(near->compute_field_potential);

    if (field0 && potentials0 == NULL)
      FCS_NEAR_LOOP_BODY_FP_F(near->compute_field_potential);

    if (field0 == NULL && potentials0)
      FCS_NEAR_LOOP_BODY_FP_P(near->compute_field_potential);

  } else if (near->compute_field_potential_3diff)
  {
    if (field0 && potentials0)
      FCS_NEAR_LOOP_BODY_3DIFF_FP(near->compute_field_potential_3diff);

    if (field0 && potentials0 == NULL)
      FCS_NEAR_LOOP_BODY_3DIFF_FP_F(near->compute_field_potential_3diff);

    if (field0 == NULL && potentials0)
      FCS_NEAR_LOOP_BODY_3DIFF_FP_P(near->compute_field_potential_3diff);

  } else
  {
    if ((field0 && near->compute_field) && (potentials0 && near->compute_potential))
      FCS_NEAR_LOOP_BODY_F_P(near->compute_field, near->compute_potential);

    if ((field0 && near->compute_field) && (potentials0 == NULL || near->compute_potential == NULL))
      FCS_NEAR_LOOP_BODY_F(near->compute_field);

    if ((field0 == NULL || near->compute_field == NULL) && (potentials0 && near->compute_potential))
      FCS_NEAR_LOOP_BODY_P(near->compute_potential);

    if ((field0 && near->compute_field_3diff) && (potentials0 && near->compute_potential_3diff))
      FCS_NEAR_LOOP_BODY_3DIFF_F_P(near->compute_field_3diff, near->compute_potential_3diff);

    if ((field0 && near->compute_field_3diff) && (potentials0 == NULL || near->compute_potential_3diff == NULL))
      FCS_NEAR_LOOP_BODY_3DIFF_F(near->compute_field_3diff);

    if ((field0 == NULL || near->compute_field_3diff == NULL) && (potentials0 && near->compute_potential_3diff))
      FCS_NEAR_LOOP_BODY_3DIFF_P(near->compute_potential_3diff);
  }
}



static fcs_int near_compute_init(fcs_near_t *near, fcs_float cutoff, const void *compute_param, MPI_Comm comm)
{
  int cart_dims[3], cart_periods[3], cart_coords[3], topo_status;

  near->context = malloc(sizeof(fcs_near_compute_context_t));

  near->context->cutoff = cutoff;
  near->context->compute_param = compute_param;
  near->context->comm = comm;

  near->context->running = 0;

#ifdef DO_TIMING
  double *t = near->context->t;

  fcs_int i;
  for (i = 0; i < sizeof(near->context->t) / sizeof(near->context->t[0]); ++i) t[i] = 0;
#endif

  TIMING_SYNC(near->context->comm); TIMING_START(t[0]);

  MPI_Comm_size(near->context->comm, &near->context->comm_size);
  MPI_Comm_rank(near->context->comm, &near->context->comm_rank);

  if (cutoff <= 0) return 1;

  MPI_Topo_test(near->context->comm, &topo_status);

  if (near->periodicity[0] < 0 || near->periodicity[1] < 0 || near->periodicity[2] < 0)
  {
    if (topo_status == MPI_CART)
    {
      MPI_Cart_get(comm, 3, cart_dims, cart_periods, cart_coords);
      near->context->periodicity[0] = cart_periods[0];
      near->context->periodicity[1] = cart_periods[1];
      near->context->periodicity[2] = cart_periods[2];

    } else return -1;

  } else
  {
    near->context->periodicity[0] = near->periodicity[0];
    near->context->periodicity[1] = near->periodicity[1];
    near->context->periodicity[2] = near->periodicity[2];
  }

  if ((near->compute_field_potential && near->compute_field_potential_3diff) || ((near->compute_field || near->compute_potential) && (near->compute_field_3diff || near->compute_potential_3diff)))
    return -2;

  INFO_CMD(
    if (near->context->comm_rank == 0)
    {
      printf(INFO_PRINT_PREFIX "near settings:\n");
      printf(INFO_PRINT_PREFIX "  box: [%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f]: [%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f] x [%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f] x [%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f]\n",
        near->box_base[0], near->box_base[1], near->box_base[2],
        near->box_a[0], near->box_a[1], near->box_a[2],
        near->box_b[0], near->box_b[1], near->box_b[2],
        near->box_c[0], near->box_c[1], near->box_c[2]);
      printf(INFO_PRINT_PREFIX "  periodicity: [%" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d]\n", near->context->periodicity[0], near->context->periodicity[1], near->context->periodicity[2]);
      printf(INFO_PRINT_PREFIX "  cutoff: %" FCS_LMOD_FLOAT "f\n", cutoff);
#if FCS_NEAR_OCL
      printf(INFO_PRINT_PREFIX "  ocl: %" FCS_LMOD_INT "d\n", near->near_param.ocl);
      printf(INFO_PRINT_PREFIX "  ocl_conf: '%s'\n", near->near_param.ocl_conf);
#endif  /* FCS_NEAR_OCL */
#if FCS_NEAR_OCL_SORT
      printf(INFO_PRINT_PREFIX "  ocl_sort: %" FCS_LMOD_INT "d\n", near->near_param.ocl_sort);
      printf(INFO_PRINT_PREFIX "  ocl_sort_algo: %" FCS_LMOD_INT "d\n", near->near_param.ocl_sort_algo);
#endif /* FCS_NEAR_OCL_SORT */
    }
  );

#if FCS_NEAR_OCL
  if (near->near_param.ocl || near->near_param.ocl_sort)
  {
#define MAX_UNITS  4
    fcs_int nunits = MAX_UNITS;
    fcs_ocl_unit_t units[MAX_UNITS];

    fcs_ocl_parse_conf(near->near_param.ocl_conf, &nunits, units);

    INFO_CMD(
      if (near->context->comm_rank == 0)
      {
        printf(INFO_PRINT_PREFIX "  ocl units: %" FCS_LMOD_INT "d\n", nunits);

        fcs_int j;
        for (j = 0; j < nunits; ++j)
        {
          printf(INFO_PRINT_PREFIX "    %" FCS_LMOD_INT "d: %" FCS_LMOD_INT "d | '%s' | '%s' | %" FCS_LMOD_INT "d\n", j, units[j].platform_index, units[j].platform_suffix, units[j].device_type, units[j].device_index);
        }
      }
    );

    fcs_ocl_near_init(&near->context->ocl, nunits, units, near->compute_field_potential_source, near->compute_field_potential_function, near->context->comm_rank);
  }
#endif /* FCS_NEAR_OCL */

  INFO_CMD(
    if (near->context->comm_rank == 0)
    {
      printf(INFO_PRINT_PREFIX "near:\n");
      printf(INFO_PRINT_PREFIX "  particles: %" FCS_LMOD_INT "d\n", near->nparticles);
      printf(INFO_PRINT_PREFIX "  ghosts: %" FCS_LMOD_INT "d\n", near->nghosts);
    }
  );

  near->context->real_boxes = malloc((near->nparticles + 1) * sizeof(box_t)); /* + 1 for a sentinel */
  if (near->nghosts > 0) near->context->ghost_boxes = malloc((near->nghosts + 1) * sizeof(box_t)); /* + 1 for a sentinel */
  else near->context->ghost_boxes = NULL;

  TIMING_SYNC(near->context->comm); TIMING_START(t[1]);
  create_boxes(near->nparticles, near->context->real_boxes, near->positions, near->indices, near->box_base, near->box_a, near->box_b, near->box_c, near->context->periodicity, near->context->cutoff);
  if (near->context->ghost_boxes) create_boxes(near->nghosts, near->context->ghost_boxes, near->ghost_positions, near->ghost_indices, near->box_base, near->box_a, near->box_b, near->box_c, near->context->periodicity, near->context->cutoff);
  TIMING_SYNC(near->context->comm); TIMING_STOP(t[1]);

#if PRINT_PARTICLES
  printf("real:\n");
  print_particles(near->nparticles, near->positions, near->context->comm_size, near->context->comm_rank, near->context->comm);
  if (near->context->ghost_boxes)
  {
    printf("ghost:\n");
    print_particles(near->nghosts, near->ghost_positions, near->context->comm_size, near->context->comm_rank, near->context->comm);
  }
#endif

#if PRINT_BOX_STATS
  print_box_stats(near->nparticles, near->positions, near->box_base, near->box_a, near->box_b, near->box_c, near->context->cutoff);
#endif

  TIMING_SYNC(near->context->comm); TIMING_START(t[2]);
#if FCS_NEAR_OCL_SORT
  if(near->near_param.ocl_sort)
  {
    fcs_ocl_sort(near);
  }
  else
#endif // FCS_NEAR_OCL_SORT
  {
    sort_into_boxes(near->nparticles, near->context->real_boxes, near->positions, near->charges, near->indices, near->field, near->potentials);
    if (near->context->ghost_boxes) sort_into_boxes(near->nghosts, near->context->ghost_boxes, near->ghost_positions, near->ghost_charges, near->ghost_indices, NULL, NULL);
  }
  TIMING_SYNC(near->context->comm); TIMING_STOP(t[2]);

#ifdef BOX_SKIP_FORMAT
  make_boxes_skip_format(near->nparticles, near->context->real_boxes);
  if (near->context->ghost_boxes) make_boxes_skip_format(near->nghosts, near->context->ghost_boxes);
#endif

#if PRINT_PARTICLES
  print_boxes(near->nparticles, near->context->real_boxes);
#endif

/*  for (i = 0; i < nlocal_particles; ++i)
    printf("%" FCS_LMOD_INT "d: %f,%f,%f  " box_fmt "  %lld\n", i, positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2], box_val(&boxes[3 * i]), indices[i]);*/

  return 0;
}


static fcs_int near_compute_main_start(fcs_near_t *near)
{
  fcs_int i;
  box_t current_box;
  fcs_int current_last, current_start, current_size;
  fcs_int real_lasts[max_nboxes], real_starts[max_nboxes], real_sizes[max_nboxes];
  fcs_int ghost_lasts[max_nboxes], ghost_starts[max_nboxes], ghost_sizes[max_nboxes];

#ifdef DO_TIMING
  double _t, *t = near->context->t;
#endif

  TIMING_SYNC(near->context->comm); TIMING_START(t[3]);

  current_last = 0;
  for (i = 0; i < max_nboxes; ++i) real_lasts[i] = ghost_lasts[i] = 0;

  TIMING_SYNC(near->context->comm); TIMING_START(t[4]);

#if FCS_NEAR_OCL
  if (near->near_param.ocl)
  {
    near->context->ocl.nboxes = count_boxes(near->nparticles, near->context->real_boxes);

    near->context->ocl.box_info = malloc(near->context->ocl.nboxes * 6 * sizeof(int));
    near->context->ocl.real_neighbour_boxes = malloc(near->context->ocl.nboxes * nreal_neighbours_full * 2 * sizeof(int));
    if (near->context->ghost_boxes)
      near->context->ocl.ghost_neighbour_boxes = malloc(near->context->ocl.nboxes * nghost_neighbours * 2 * sizeof(int));

    fcs_int *current_box_info = near->context->ocl.box_info;
    fcs_int current_real_neighbour_start = 0;
    fcs_int current_ghost_neighbour_start = 0;

    do
    {
      current_box = near->context->real_boxes[current_last];

      TIMING_START(_t);

      find_box(near->context->real_boxes, near->nparticles, current_box, current_last, &current_start, &current_size);

/*      printf("box: " box_fmt ", start: %" FCS_LMOD_INT "d, size: %" FCS_LMOD_INT "d\n", box_val(current_box), current_start, current_size);*/

      find_neighbours(nreal_neighbours_full, real_neighbours_full, near->context->real_boxes, near->nparticles, current_box, real_lasts, real_starts, real_sizes);
      if (near->context->ghost_boxes) find_neighbours(nghost_neighbours, ghost_neighbours, near->context->ghost_boxes, near->nghosts, current_box, ghost_lasts, ghost_starts, ghost_sizes);

      TIMING_STOP_ADD(_t, t[5]);

      TIMING_START(_t);
      current_box_info[0] = current_start;
      current_box_info[1] = current_start + current_size;

      current_box_info[2] = current_box_info[3] = current_real_neighbour_start / 2;

      for (i = 0; i < nreal_neighbours_full; ++i)
      {
/*        printf("  real-neighbour %" FCS_LMOD_INT "d: %" FCS_LMOD_INT "d / %" FCS_LMOD_INT "d\n", i, current_starts[i], current_sizes[i]);*/

        real_lasts[i] = real_starts[i] + real_sizes[i];

        if (real_sizes[i] == 0) continue;

        near->context->ocl.real_neighbour_boxes[current_real_neighbour_start + 0] = real_starts[i];
        near->context->ocl.real_neighbour_boxes[current_real_neighbour_start + 1] = real_starts[i] + real_sizes[i];

        current_real_neighbour_start += 2;
        ++current_box_info[3];
      }

      current_box_info[4] = current_box_info[5] = current_ghost_neighbour_start / 2;

      if (near->context->ghost_boxes)
      for (i = 0; i < nghost_neighbours; ++i)
      {
/*        printf("  ghost-neighbour %" FCS_LMOD_INT "d: %" FCS_LMOD_INT "d / %" FCS_LMOD_INT "d\n", i, ghost_starts[i], ghost_sizes[i]);*/

        ghost_lasts[i] = ghost_starts[i] + ghost_sizes[i];

        if (ghost_sizes[i] == 0) continue;

        near->context->ocl.ghost_neighbour_boxes[current_ghost_neighbour_start + 0] = ghost_starts[i];
        near->context->ocl.ghost_neighbour_boxes[current_ghost_neighbour_start + 1] = ghost_starts[i] + ghost_sizes[i];

        current_ghost_neighbour_start += 2;
        ++current_box_info[5];
      }
      TIMING_STOP_ADD(_t, t[6]);

      current_box_info += 6;
      current_last = current_start + current_size;

    } while (current_last < near->nparticles);

    near->context->ocl.nreal_neighbour_boxes = current_real_neighbour_start / 2;
    near->context->ocl.nghost_neighbour_boxes = current_ghost_neighbour_start / 2;

#if COMPUTE_VERBOSE
    printf("nboxes: %" FCS_LMOD_INT "d\n", near->context->ocl.nboxes);
    printf("nreal_neighbour_boxes: %" FCS_LMOD_INT "d\n", near->context->ocl.nreal_neighbour_boxes);
    printf("nghost_neighbour_boxes: %" FCS_LMOD_INT "d\n", near->context->ocl.nghost_neighbour_boxes);

    for (i = 0; i < near->context->ocl.nboxes; ++i)
    {
      printf("box_info[%" FCS_LMOD_INT "d]: %" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d\n", i,
        near->context->ocl.box_info[6 * i + 0], near->context->ocl.box_info[6 * i + 1], near->context->ocl.box_info[6 * i + 2],
        near->context->ocl.box_info[6 * i + 3], near->context->ocl.box_info[6 * i + 4], near->context->ocl.box_info[6 * i + 5]);
    }

    for (i = 0; i < near->context->ocl.nreal_neighbour_boxes; ++i)
    {
      printf("nreal_neighbour_boxes[%" FCS_LMOD_INT "d]: %" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d\n", i,
        near->context->ocl.real_neighbour_boxes[2 * i + 0], near->context->ocl.real_neighbour_boxes[2 * i + 1]);
    }

    for (i = 0; i < near->context->ocl.nghost_neighbour_boxes; ++i)
    {
      printf("nghost_neighbour_boxes[%" FCS_LMOD_INT "d]: %" FCS_LMOD_INT "d, %" FCS_LMOD_INT "d\n", i,
        near->context->ocl.ghost_neighbour_boxes[2 * i + 0], near->context->ocl.ghost_neighbour_boxes[2 * i + 1]);
    }
#endif /* COMPUTE_VERBOSE */

    TIMING_SYNC(near->context->comm); TIMING_START(t[7]);

    fcs_ocl_compute_near_prepare(&near->context->ocl, near->context->cutoff, near->context->compute_param, near->compute_param_size,
      near->nparticles, near->positions, near->charges, near->field, near->potentials,
      near->nghosts, near->ghost_positions, near->ghost_charges);

    TIMING_SYNC(near->context->comm); TIMING_STOP(t[7]);

    TIMING_SYNC(near->context->comm); TIMING_START(t[8]);

    fcs_ocl_compute_near_start(&near->context->ocl);

#if !FCS_NEAR_OCL_ASYNC

    fcs_ocl_compute_near_join(&near->context->ocl);

    TIMING_SYNC(near->context->comm); TIMING_STOP(t[8]);

    TIMING_SYNC(near->context->comm); TIMING_START(t[9]);

    fcs_ocl_compute_near_release(&near->context->ocl, near->nparticles, near->field, near->potentials, near->nghosts);

    TIMING_SYNC(near->context->comm); TIMING_STOP(t[9]);

    near->context->ocl.nboxes = 0;
    free(near->context->ocl.box_info);
    near->context->ocl.box_info = NULL;
    free(near->context->ocl.real_neighbour_boxes);
    near->context->ocl.real_neighbour_boxes = NULL;
    if (near->context->ghost_boxes)
    {
      free(near->context->ocl.ghost_neighbour_boxes);
      near->context->ocl.ghost_neighbour_boxes = NULL;
    }

#endif /* !FCS_NEAR_OCL_ASYNC */

  } else
#endif /* FCS_NEAR_OCL */
  {
    do
    {
      current_box = near->context->real_boxes[current_last];

      TIMING_START(_t);

      find_box(near->context->real_boxes, near->nparticles, current_box, current_last, &current_start, &current_size);

  /*    printf("box: " box_fmt ", start: %" FCS_LMOD_INT "d, size: %" FCS_LMOD_INT "d\n",
        box_val(current_box), current_start, current_size);*/

      find_neighbours(nreal_neighbours, real_neighbours, near->context->real_boxes, near->nparticles, current_box, real_lasts, real_starts, real_sizes);
      if (near->context->ghost_boxes) find_neighbours(nghost_neighbours, ghost_neighbours, near->context->ghost_boxes, near->nghosts, current_box, ghost_lasts, ghost_starts, ghost_sizes);

      TIMING_STOP_ADD(_t, t[5]);

      TIMING_START(_t);
      compute_near(near->positions, near->charges, near->field, near->potentials, current_start, current_size, NULL, NULL, current_start, current_size, near->context->cutoff, near, near->context->compute_param);
      for (i = 0; i < nreal_neighbours; ++i)
      {
  /*      printf("  real-neighbour %" FCS_LMOD_INT "d: %" FCS_LMOD_INT "d / %" FCS_LMOD_INT "d\n", i, current_starts[i], current_sizes[i]);*/

        real_lasts[i] = real_starts[i] + real_sizes[i];

        compute_near(near->positions, near->charges, near->field, near->potentials, current_start, current_size, NULL, NULL, real_starts[i], real_sizes[i], near->context->cutoff, near, near->context->compute_param);
      }

      if (near->context->ghost_boxes)
      for (i = 0; i < nghost_neighbours; ++i)
      {
  /*      printf("  ghost-neighbour %" FCS_LMOD_INT "d: %" FCS_LMOD_INT "d / %" FCS_LMOD_INT "d\n", i, ghost_starts[i], ghost_sizes[i]);*/

        ghost_lasts[i] = ghost_starts[i] + ghost_sizes[i];

        compute_near(near->positions, near->charges, near->field, near->potentials, current_start, current_size, near->ghost_positions, near->ghost_charges, ghost_starts[i], ghost_sizes[i], near->context->cutoff, near, near->context->compute_param);
      }
      current_last = current_start + current_size;
      TIMING_STOP_ADD(_t, t[6]);

    } while (current_last < near->nparticles);
  }

  TIMING_SYNC(near->context->comm); TIMING_STOP(t[4]);

  return 0;
}


static fcs_int near_compute_main_join(fcs_near_t *near)
{
#ifdef DO_TIMING
  double *t = near->context->t;
#endif

#if FCS_NEAR_OCL
  if (near->near_param.ocl)
  {
#if FCS_NEAR_OCL_ASYNC

    fcs_ocl_compute_near_join(&near->context->ocl);

    TIMING_SYNC(near->context->comm); TIMING_STOP(t[8]);

    TIMING_SYNC(near->context->comm); TIMING_START(t[9]);

    fcs_ocl_compute_near_release(&near->context->ocl, near->nparticles, near->field, near->potentials, near->nghosts);

    TIMING_SYNC(near->context->comm); TIMING_STOP(t[9]);

    if (near->context->ghost_boxes)
    {
      free(near->context->ocl.ghost_neighbour_boxes);
      near->context->ocl.ghost_neighbour_boxes = NULL;
    }

    near->context->ocl.nboxes = 0;
    free(near->context->ocl.box_info);
    near->context->ocl.box_info = NULL;
    free(near->context->ocl.real_neighbour_boxes);
    near->context->ocl.real_neighbour_boxes = NULL;

#endif /* FCS_NEAR_OCL_ASYNC */
  }
#endif /* FCS_NEAR_OCL */

  free(near->context->real_boxes);
  if (near->context->ghost_boxes) free(near->context->ghost_boxes);

  TIMING_SYNC(near->context->comm); TIMING_STOP(t[3]);

  return 0;
}


#if FCS_NEAR_ASYNC

static void *near_compute_main(void *arg)
{
  fcs_near_t *near = (fcs_near_t *) arg;

  near_compute_main_start(near);
  near_compute_main_join(near);

  return NULL;
}

#endif /* FCS_NEAR_ASYNC */


static fcs_int near_compute_start(fcs_near_t *near, int async)
{
  if (near->context->running) return 1;

#if FCS_NEAR_ASYNC
  near->context->async = async;

  if (near->context->async)
  {
    pthread_create(&near->context->thread, NULL, near_compute_main, near);

  } else
#endif /* FCS_NEAR_ASYNC */
  {
    near_compute_main_start(near);
  }

  near->context->running = 1;

  return 0;
}


static fcs_int near_compute_join(fcs_near_t *near)
{
  if (!near->context->running) return 1;

#if FCS_NEAR_ASYNC
  if (near->context->async)
  {
    pthread_join(near->context->thread, NULL);

  } else
#endif /* FCS_NEAR_ASYNC */
  {
    near_compute_main_join(near);
  }

  near->context->running = 0;

  return 0;
}


static fcs_int near_compute_release(fcs_near_t *near)
{
#ifdef DO_TIMING
  double *t = near->context->t;
#endif

#if FCS_NEAR_OCL
  if (near->near_param.ocl || near->near_param.ocl_sort)
  {
    fcs_ocl_near_release(&near->context->ocl);
  }
#endif /* FCS_NEAR_OCL */

  TIMING_SYNC(near->context->comm); TIMING_STOP(t[0]);

#if PRINT_PARTICLES
  printf("%d: result = %" FCS_LMOD_INT "d\n", near->context->comm_rank, near->nparticles);
  for (fcs_int i = 0; i < near->nparticles; ++i)
  {
    printf("%" FCS_LMOD_INT "d: %" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f,%" FCS_LMOD_FLOAT "f  %" FCS_LMOD_FLOAT "f\n",
      i, near->field[3 * i + 0], near->field[3 * i + 1], near->field[3 * i + 2], near->potentials[i]);
  }
#endif

  TIMING_CMD(
    if (near->context->comm_rank == 0)
      printf(TIMING_PRINT_PREFIX "fcs_near_compute: %f  %f  %f  %f  %f  %f  %f  %f  %f  %f\n", t[0], t[1], t[2], t[3], t[4], t[5], t[6], t[7], t[8], t[9]);
  );

  free(near->context);
  near->context = NULL;

  return 0;
}


fcs_int fcs_near_compute(fcs_near_t *near, fcs_float cutoff, const void *compute_param, MPI_Comm comm)
{
  fcs_int ret;

  ret = near_compute_init(near, cutoff, compute_param, comm);
  if (ret) goto exit;

  ret = near_compute_start(near, 0);
  if (ret) goto free_exit;

  ret = near_compute_join(near);
  if (ret) goto free_exit;

free_exit:
  ret = ret || near_compute_release(near);

exit:
  return ret;
}


fcs_int fcs_near_compute_prepare(fcs_near_t *near, fcs_float cutoff, const void *compute_param, MPI_Comm comm)
{
  return near_compute_init(near, cutoff, compute_param, comm);
}


fcs_int fcs_near_compute_start(fcs_near_t *near)
{
  return near_compute_start(near, 1);
}


fcs_int fcs_near_compute_join(fcs_near_t *near)
{
  return near_compute_join(near);
}


fcs_int fcs_near_compute_finish(fcs_near_t *near)
{
  return near_compute_release(near);
}


/*#define SORT_FORWARD_BOUNDS*/
/*#define CREATE_GHOSTS_SEPARATE*/
/*#define SEPARATE_GHOSTS*/
/*#define SEPARATE_ZSLICES  7*/


fcs_int fcs_near_field_solver(fcs_near_t *near,
                              fcs_float cutoff,
                              const void *compute_param,
                              MPI_Comm comm)
{
  int comm_size, comm_rank;

  fcs_int i;

  fcs_near_t near_s;

  fcs_int nlocal_s;
  fcs_float *positions_s, *charges_s;
  fcs_gridsort_index_t *indices_s;
  fcs_float *field_s;
  fcs_float *potentials_s;

  fcs_int nlocal_s_real;
  fcs_float *positions_s_real, *charges_s_real;
  fcs_gridsort_index_t *indices_s_real;

  fcs_int resort;

#ifdef SEPARATE_GHOSTS
  fcs_int nlocal_s_ghost;
  fcs_float *positions_s_ghost, *charges_s_ghost;
  fcs_gridsort_index_t *indices_s_ghost;
#endif

  fcs_gridsort_t gridsort;

#ifdef SORT_FORWARD_BOUNDS
  fcs_float lower_bounds[3], upper_bounds[3];
#endif

  fcs_int periodicity[3];
  MPI_Comm cart_comm;
  int cart_dims[3], cart_periods[3], cart_coords[3], topo_status;

#ifdef DO_TIMING
  double t[4] = { 0, 0, 0, 0 };
#endif


  if (comm == MPI_COMM_NULL)
  {
    /* ERROR */
    return -1;
  }

  TIMING_SYNC(comm); TIMING_START(t[0]);

  MPI_Comm_size(comm, &comm_size);
  MPI_Comm_rank(comm, &comm_rank);

  if (cutoff <= 0) goto exit;

  MPI_Topo_test(comm, &topo_status);

  if (near->periodicity[0] < 0 || near->periodicity[1] < 0 || near->periodicity[2] < 0)
  {
    if (topo_status == MPI_CART)
    {
      MPI_Cart_get(comm, 3, cart_dims, cart_periods, cart_coords);
      periodicity[0] = cart_periods[0];
      periodicity[1] = cart_periods[1];
      periodicity[2] = cart_periods[2];

    } else return -1;

  } else
  {
    periodicity[0] = near->periodicity[0];
    periodicity[1] = near->periodicity[1];
    periodicity[2] = near->periodicity[2];
  }

  if (topo_status != MPI_CART)
  {
    cart_dims[0] = cart_dims[1] = cart_dims[2] = 0;
    MPI_Dims_create(comm_size, 3, cart_dims);

    cart_periods[0] = periodicity[0];
    cart_periods[1] = periodicity[1];
    cart_periods[2] = periodicity[2];

    MPI_Cart_create(comm, 3, cart_dims, cart_periods, 0, &cart_comm);

  } else cart_comm = comm;

/*  printf("%d: input = %" FCS_LMOD_INT "d\n", comm_rank, nlocal_particles);
  for (int i = 0; i < nlocal_particles; ++i)
  {
    printf("  %d: %f,%f,%f  %lld\n", i, positions[3 * i + 0], positions[3 * i + 1], positions[3 * i + 2]);
  }*/

  fcs_gridsort_create(&gridsort);

  fcs_gridsort_set_system(&gridsort, near->box_base, near->box_a, near->box_b, near->box_c, periodicity);

#ifdef SORT_FORWARD_BOUNDS
  MPI_Cart_get(cart_comm, 3, cart_dims, cart_periods, cart_coords);

  lower_bounds[0] = near->box_base[0] + (fcs_float) cart_coords[0] * near->box_a[0] / (fcs_float) cart_dims[0];
  lower_bounds[1] = near->box_base[1] + (fcs_float) cart_coords[1] * near->box_b[1] / (fcs_float) cart_dims[1];
  lower_bounds[2] = near->box_base[2] + (fcs_float) cart_coords[2] * near->box_c[2] / (fcs_float) cart_dims[2];

  upper_bounds[0] = near->box_base[0] + (fcs_float) (cart_coords[0] + 1.0) * near->box_a[0] / (fcs_float) cart_dims[0];
  upper_bounds[1] = near->box_base[1] + (fcs_float) (cart_coords[1] + 1.0) * near->box_b[1] / (fcs_float) cart_dims[1];
  upper_bounds[2] = near->box_base[2] + (fcs_float) (cart_coords[2] + 1.0) * near->box_c[2] / (fcs_float) cart_dims[2];

  DEBUG_CMD(
    printf(DEBUG_PRINT_PREFIX "%d: bounds: %" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f - %" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f, %" FCS_LMOD_FLOAT "f\n",
      comm_rank, lower_bounds[0], lower_bounds[1], lower_bounds[2], upper_bounds[0], upper_bounds[1], upper_bounds[2]);
  );

  fcs_gridsort_set_bounds(&gridsort, lower_bounds, upper_bounds);
#endif

  fcs_gridsort_set_particles(&gridsort, near->nparticles, near->max_nparticles, near->positions, near->charges);

  fcs_gridsort_set_max_particle_move(&gridsort, near->max_particle_move);

  TIMING_SYNC(comm); TIMING_START(t[1]);
#ifdef CREATE_GHOSTS_SEPARATE
  fcs_gridsort_sort_forward(&gridsort, 0, cart_comm);
  fcs_gridsort_create_ghosts(&gridsort, cutoff, cart_comm);
#else
  fcs_gridsort_sort_forward(&gridsort, cutoff, cart_comm);
#endif
  TIMING_SYNC(comm); TIMING_STOP(t[1]);

  fcs_gridsort_get_sorted_particles(&gridsort, &nlocal_s, NULL, &positions_s, &charges_s, &indices_s);

#ifdef SEPARATE_GHOSTS
  fcs_gridsort_separate_ghosts(&gridsort);
  fcs_gridsort_get_ghost_particles(&gridsort, &nlocal_s_ghost, &positions_s_ghost, &charges_s_ghost, &indices_s_ghost);
#endif

#ifdef SEPARATE_ZSLICES
  fcs_int zslices_nparticles[SEPARATE_ZSLICES];
  fcs_gridsort_separate_zslices(&gridsort, SEPARATE_ZSLICES, zslices_nparticles);
#endif

  fcs_gridsort_get_real_particles(&gridsort, &nlocal_s_real, &positions_s_real, &charges_s_real, &indices_s_real);

#if PRINT_PARTICLES
  printf("%d: sorted (real) = %" FCS_LMOD_INT "d\n", comm_rank, nlocal_s_real);
  for (int i = 0; i < nlocal_s_real; ++i)
  {
    printf("  %d: %d: %f,%f,%f  %f  " idx_fmt "\n", comm_rank, i, positions_s[3 * i + 0], positions_s[3 * i + 1], positions_s[3 * i + 2], charges_s[i], idx_val(indices_s[i]));
  }

#ifdef SEPARATE_GHOSTS
  printf("%d: sorted (ghost) = %" FCS_LMOD_INT "d\n", comm_rank, nlocal_s_ghost);
  for (int i = 0; i < nlocal_s_ghost; ++i)
  {
    printf("  %d: %f,%f,%f  " idx_fmt "\n", i, positions_s[3 * (nlocal_s_real + i) + 0], positions_s[3 * (nlocal_s_real + i) + 1], positions_s[3 * (nlocal_s_real + i) + 2], idx_val(indices_s[nlocal_s_real + i]));
  }
#endif
#endif

  if (near->field) field_s = malloc(nlocal_s_real * 3 * sizeof(fcs_float));
  else field_s = NULL;
  if (near->potentials) potentials_s = malloc(nlocal_s_real * sizeof(fcs_float));
  else potentials_s = NULL;

  if (field_s && potentials_s)
  {
    for (i = 0; i < nlocal_s_real; ++i) field_s[3 * i + 0] = field_s[3 * i + 1] = field_s[3 * i + 2] = potentials_s[i] = 0;

  } else
  {
    if (field_s) for (i = 0; i < nlocal_s_real; ++i) field_s[3 * i + 0] = field_s[3 * i + 1] = field_s[3 * i + 2] = 0;
    if (potentials_s) for (i = 0; i < nlocal_s_real; ++i) potentials_s[i] = 0;
  }

  fcs_near_create(&near_s);

  fcs_near_set_param(&near_s, &near->near_param);

  fcs_near_set_field(&near_s, near->compute_field);
  fcs_near_set_potential(&near_s, near->compute_potential);
  fcs_near_set_field_potential(&near_s, near->compute_field_potential);

  fcs_near_set_field_3diff(&near_s, near->compute_field_3diff);
  fcs_near_set_potential_3diff(&near_s, near->compute_potential_3diff);
  fcs_near_set_field_potential_3diff(&near_s, near->compute_field_potential_3diff);

  fcs_near_set_loop(&near_s, near->compute_loop);

  fcs_near_set_field_potential_source(&near_s, near->compute_field_potential_source, near->compute_field_potential_function);

  if (near->periodicity[0] < 0 || near->periodicity[1] < 0 || near->periodicity[2] < 0)
    fcs_near_set_system(&near_s, near->box_base, near->box_a, near->box_b, near->box_c, NULL);
  else
    fcs_near_set_system(&near_s, near->box_base, near->box_a, near->box_b, near->box_c, near->periodicity);

  fcs_near_set_particles(&near_s, nlocal_s_real, nlocal_s_real, positions_s_real, charges_s_real, indices_s_real, field_s, potentials_s);

#ifdef SEPARATE_GHOSTS
  fcs_near_set_ghosts(&near_s, nlocal_s_ghost, positions_s_ghost, charges_s_ghost, indices_s_ghost);
#endif

  TIMING_SYNC(comm); TIMING_START(t[2]);
  fcs_near_compute(&near_s, cutoff, compute_param, cart_comm);
  TIMING_SYNC(comm); TIMING_STOP(t[2]);

  fcs_near_destroy(&near_s);

#if PRINT_PARTICLES
  printf("%d: result = %" FCS_LMOD_INT "d\n", comm_rank, nlocal_s);
  for (int i = 0; i < nlocal_s_real; ++i)
  {
    printf("%d: %f,%f,%f  %f\n", i, field_s[3 * i + 0], field_s[3 * i + 1], field_s[3 * i + 2], potentials_s[i]);
  }
#endif

  fcs_gridsort_set_sorted_results(&gridsort, nlocal_s_real, field_s, potentials_s);
  fcs_gridsort_set_results(&gridsort, near->max_nparticles, near->field, near->potentials);

  TIMING_SYNC(comm); TIMING_START(t[3]);
  if (near->resort) resort = fcs_gridsort_prepare_resort(&gridsort, comm);
  else resort = 0;

  if (!resort) fcs_gridsort_sort_backward(&gridsort, comm);

  fcs_gridsort_resort_destroy(&near->gridsort_resort);

  if (resort) fcs_gridsort_resort_create(&near->gridsort_resort, &gridsort, comm);
  TIMING_SYNC(comm); TIMING_STOP(t[3]);

  if (field_s) free(field_s);
  if (potentials_s) free(potentials_s);

  fcs_gridsort_free(&gridsort);

  fcs_gridsort_destroy(&gridsort);

  if (cart_comm != comm) MPI_Comm_free(&cart_comm);

exit:
  TIMING_SYNC(comm); TIMING_STOP(t[0]);

  TIMING_CMD(
    if (comm_rank == 0)
      printf(TIMING_PRINT_PREFIX "fcs_near_field_solver: %f  %f  %f  %f\n", t[0], t[1], t[2], t[3]);
  );

  return 0;
}


void fcs_near_resort_create(fcs_near_resort_t *near_resort, fcs_near_t *near)
{
  *near_resort = near->gridsort_resort;

  near->gridsort_resort = FCS_GRIDSORT_RESORT_NULL;
}


void fcs_near_resort_destroy(fcs_near_resort_t *near_resort)
{
  fcs_gridsort_resort_destroy(near_resort);

  *near_resort = FCS_NEAR_RESORT_NULL;
}


void fcs_near_resort_print(fcs_near_resort_t near_resort, MPI_Comm comm)
{
  fcs_gridsort_resort_print(near_resort, comm);
}


fcs_int fcs_near_resort_is_available(fcs_near_resort_t near_resort)
{
  return fcs_gridsort_resort_is_available(near_resort);
}


fcs_int fcs_near_resort_get_original_particles(fcs_near_resort_t near_resort)
{
  return fcs_gridsort_resort_get_original_particles(near_resort);
}


fcs_int fcs_near_resort_get_sorted_particles(fcs_near_resort_t near_resort)
{
  return fcs_gridsort_resort_get_sorted_particles(near_resort);
}


void fcs_near_resort_ints(fcs_near_resort_t near_resort, fcs_int *src, fcs_int *dst, fcs_int n, MPI_Comm comm)
{
  fcs_gridsort_resort_ints(near_resort, src, dst, n, comm);
}


void fcs_near_resort_floats(fcs_near_resort_t near_resort, fcs_float *src, fcs_float *dst, fcs_int n, MPI_Comm comm)
{
  fcs_gridsort_resort_floats(near_resort, src, dst, n, comm);
}


void fcs_near_resort_bytes(fcs_near_resort_t near_resort, void *src, void *dst, fcs_int n, MPI_Comm comm)
{
  fcs_gridsort_resort_bytes(near_resort, src, dst, n, comm);
}
