/*
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

#ifndef __FCS_OCL_H__
#define __FCS_OCL_H__


#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif


extern const char *fcs_ocl_cl_config;
extern const char *fcs_ocl_cl;
extern const char *fcs_ocl_math_cl;


typedef struct
{
#define FCS_OCL_UNIT_STR_SIZE  32

  fcs_int platform_index;
  char platform_suffix[FCS_OCL_UNIT_STR_SIZE];

  char device_type[FCS_OCL_UNIT_STR_SIZE];
  fcs_int device_index;

} fcs_ocl_unit_t;


typedef struct _fcs_ocl_t
{
  void *dummy;

} fcs_ocl_t;


fcs_int fcs_ocl_init(fcs_ocl_t *ocl);
fcs_int fcs_ocl_tune(fcs_ocl_t *ocl);
fcs_int fcs_ocl_destroy(fcs_ocl_t *ocl);

fcs_int fcs_ocl_parse_conf(const char *ocl_conf, fcs_int *nunits, fcs_ocl_unit_t *units);
fcs_int fcs_ocl_get_device(fcs_ocl_unit_t *unit, int comm_rank, cl_device_id *device_id);


#endif /* __FCS_OCL_H__ */
