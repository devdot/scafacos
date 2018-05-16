/*
  Copyright (C) 2018 Michael Hofmann, Thomas Schaller
  
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


#ifndef __NEAR_OCL_H__
#define __NEAR_OCL_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "near.h"

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#define MEM_SIZE (128)
#define MAX_SOURCE_SIZE (0xFFFFFF)

#define CL_CHECK(_expr)                                                     \
do {                                                                        \
  cl_int _err = _expr;                                                      \
  if (_err == CL_SUCCESS)                                                   \
    break;                                                                  \
  fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err);  \
  abort();                                                                  \
} while (0)


/*  Hilfsfunktion zur Ausgabe von OpenCL Fehlercodes  */
#define CL_CHECK_ERR(_expr)                                                  \
({                                                                           \
  cl_int _err = CL_INVALID_VALUE;                                            \
  typeof(_expr) _ret = _expr;                                                \
  if (_err != CL_SUCCESS) {                                                  \
    fprintf(stderr, "OpenCL Error: '%s' returned %d!\n", #_expr, (int)_err); \
    abort();                                                                 \
  }                                                                          \
  _ret;                                                                      \
})

typedef struct
{
  fcs_int nboxes;
  fcs_int *box_info, *real_neighbour_boxes, *ghost_neighbour_boxes;
  fcs_int nreal_neighbour_boxes, nghost_neighbour_boxes;

  cl_context context;
  cl_command_queue command_queue;

  cl_program program;
  cl_kernel compute_kernel;

  cl_mem mem_param;
  cl_mem mem_positions, mem_charges, mem_field, mem_potentials;
  cl_mem mem_gpositions, mem_gcharges;
  cl_mem mem_box_info, mem_real_neighbour_boxes, mem_ghost_neighbour_boxes;

  cl_event kernel_completion;

#if FCS_NEAR_OCL_SORT
  int use_index;

  cl_device_id device_id;

  cl_ulong local_memory;

  cl_mem mem_boxes;
  cl_mem mem_data;
  
  cl_mem mem_indices;

  cl_program sort_program_bitonic;
  cl_kernel sort_kernel_bitonic_global_2; 

  cl_program sort_program_hybrid;
  cl_kernel sort_kernel_bitonic_local;

  cl_program sort_program_radix;
  cl_kernel sort_kernel_radix_histogram;
  cl_kernel sort_kernel_radix_histogram_paste;
  cl_kernel sort_kernel_radix_scan;
  cl_kernel sort_kernel_radix_reorder;

  cl_kernel sort_kernel_move_data_float;
  cl_kernel sort_kernel_move_data_float_triple;
  cl_kernel sort_kernel_move_data_gridsort_index;

  cl_event sort_kernel_completion;
#endif

} fcs_ocl_context_t;


#ifdef __cplusplus
}
#endif

#endif /* __NEAR_OCL_H__ */
