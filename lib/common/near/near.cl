
typedef void HERE_COMES_THE_CODE;


static void fcs_ocl_near_coulomb_field_potential(__global const void *param, fcs_float dist, fcs_float *f, fcs_float *p)
{
  *p = 1.0 / dist;
  *f = -(*p) * (*p);
}


__kernel void fcs_ocl_near_compute(fcs_float cutoff, __global void *param,
  __global fcs_float *positions, __global fcs_float *charges, __global fcs_float *field, __global fcs_float *potentials,
  __global fcs_int *box_info, __global fcs_int *real_neighbour_boxes,
  __global fcs_float *gpositions, __global fcs_float *gcharges, 
  __global fcs_int *ghost_neighbour_boxes)
{
  fcs_float r_ij, f, p, fx;
  fcs_float x0, y0, z0, dx, dy, dz;
  fcs_float fs_x, fs_y, fs_z, ps;
  fcs_int i, j, k;

  size_t b = get_global_id(0);

#if COMPUTE_VERBOSE
  printf("b: %d\n", (int) b);
#endif

#if COMPUTE
  for (i = box_info[6 * b + 0]; i < box_info[6 * b + 1]; ++i)
  {
    fs_x = fs_y = fs_z = 0;
    ps = 0;

    x0 = positions[3 * i + 0];
    y0 = positions[3 * i + 1];
    z0 = positions[3 * i + 2];

#if COMPUTE_BOX
    for (j = i + 1; j < box_info[6 * b + 1]; ++j)
    {
      dx = positions[3 * j + 0] - x0;
      dy = positions[3 * j + 1] - y0;
      dz = positions[3 * j + 2] - z0;

      r_ij = fcs_sqrt((dx * dx) + (dy * dy) + (dz * dz));

#if COMPUTE_VERBOSE
      printf("box: %d vs. %d -> %f\n", (int) i, (int) j, (float) r_ij);
#endif /* COMPUTE_VERBOSE */

      if (r_ij > cutoff) continue;

      _nfp_(param, r_ij, &f, &p);

      fx = f * charges[j] / r_ij;
      fs_x += fx * dx;
      fs_y += fx * dy;
      fs_z += fx * dz;

      fx = -f * charges[i] / r_ij;
      field[3 * j + 0] += fx * dx;
      field[3 * j + 1] += fx * dy;
      field[3 * j + 2] += fx * dz;

#if POTENTIAL_CONST1
      ps += 1;
      potentials[j] += 1;
#else /* POTENTIAL_CONST1 */
      ps += p * charges[j];
      potentials[j] += p * charges[i];
#endif /* POTENTIAL_CONST1 */
    }
#endif /* COMPUTE_BOX */

#if COMPUTE_REAL_NEIGHBOURS
    for (k = box_info[6 * b + 2]; k < box_info[6 * b + 3]; ++k)
    {
      for (j = real_neighbour_boxes[2 * k + 0]; j < real_neighbour_boxes[2 * k + 1]; ++j)
      {
        dx = positions[3 * j + 0] - x0;
        dy = positions[3 * j + 1] - y0;
        dz = positions[3 * j + 2] - z0;

        r_ij = fcs_sqrt((dx * dx) + (dy * dy) + (dz * dz));

#if COMPUTE_VERBOSE
        printf("real: %d vs. %d -> %f\n", (int) i, (int) j, (float) r_ij);
#endif /* COMPUTE_VERBOSE */

        if (r_ij > cutoff) continue;

        _nfp_(param, r_ij, &f, &p);

        fx = f * charges[j] / r_ij;
        fs_x += fx * dx;
        fs_y += fx * dy;
        fs_z += fx * dz;
#if POTENTIAL_CONST1
        ps += 1;
#else /* POTENTIAL_CONST1 */
        ps += p * charges[j];
#endif /* POTENTIAL_CONST1 */
      }
    }
#endif /* COMPUTE_REAL_NEIGHBOURS */

#if COMPUTE_GHOST_NEIGHBOURS
    for (k = box_info[6 * b + 4]; k < box_info[6 * b + 5]; ++k)
    {
      for (j = ghost_neighbour_boxes[2 * k + 0]; j < ghost_neighbour_boxes[2 * k + 1]; ++j)
      {
        dx = gpositions[3 * j + 0] - x0;
        dy = gpositions[3 * j + 1] - y0;
        dz = gpositions[3 * j + 2] - z0;

        r_ij = fcs_sqrt((dx * dx) + (dy * dy) + (dz * dz));

#if COMPUTE_VERBOSE
        printf("ghost: %d vs. %d -> %f\n", (int) i, (int) j, (float) r_ij);
#endif /* COMPUTE_VERBOSE */

        if (r_ij > cutoff) continue;

        _nfp_(param, r_ij, &f, &p);

        fx = f * gcharges[j] / r_ij;
        fs_x += fx * dx;
        fs_y += fx * dy;
        fs_z += fx * dz;
#if POTENTIAL_CONST1
        ps += 1;
#else /* POTENTIAL_CONST1 */
        ps += p * gcharges[j];
#endif /* POTENTIAL_CONST1 */
      }
    }
#endif /* COMPUTE_GHOST_NEIGHBOURS */

    field[3 * i + 0] += fs_x;
    field[3 * i + 1] += fs_y;
    field[3 * i + 2] += fs_z;

    potentials[i] += ps;
  }
#endif /* COMPUTE */
}
