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


#ifdef HAVE_CONFIG_H
# include <config.h>
#endif

#include <stdio.h>
#include <string.h>

#include "z_tools.h"
#include "fcs_ocl.h"


#if defined(FCS_ENABLE_DEBUG_OCL)
# define DO_DEBUG
# define DEBUG_CMD(_cmd_)  Z_MOP(_cmd_)
#else
# define DEBUG_CMD(_cmd_)  Z_NOP()
#endif
#define DEBUG_PRINT_PREFIX  "OCL_DEBUG: "

#if defined(FCS_ENABLE_INFO_OCL)
# define DO_INFO
# define INFO_CMD(_cmd_)  Z_MOP(_cmd_)
#else
# define INFO_CMD(_cmd_)  Z_NOP()
#endif
#define INFO_PRINT_PREFIX  "OCL_INFO: "

#if defined(FCS_ENABLE_TIMING_OCL)
# define DO_TIMING
# define TIMING_CMD(_cmd_)  Z_MOP(_cmd_)
#else
# define TIMING_CMD(_cmd_)  Z_NOP()
#endif
#define TIMING_PRINT_PREFIX  "OCL_TIMING: "



const char *fcs_ocl_cl_config =
  "#undef FCS_FLOAT_IS_FLOAT\n"
  "#undef FCS_FLOAT_IS_DOUBLE\n"
  "#undef FCS_FLOAT_IS_LONG_DOUBLE\n"
#if defined(FCS_FLOAT_IS_FLOAT)
  "#define FCS_FLOAT_IS_FLOAT  1\n"
#elif defined(FCS_FLOAT_IS_DOUBLE)
  "#define FCS_FLOAT_IS_DOUBLE  1\n"
#elif defined(FCS_FLOAT_IS_LONG_DOUBLE)
# error FCS float data type long double not supported by OpenCL
#else
# error FCS float data type is unknown
#endif
  "\n"
  "#undef FCS_INT_IS_SHORT\n"
  "#undef FCS_INT_IS_INT\n"
  "#undef FCS_INT_IS_LONG\n"
  "#undef FCS_INT_IS_LONG_LONG\n"
#if defined(FCS_INT_IS_SHORT)
  "#define FCS_INT_IS_SHORT  1\n"
#elif defined(FCS_INT_IS_INT)
  "#define FCS_INT_IS_INT  1\n"
#elif defined(FCS_INT_IS_LONG)
  "#define FCS_INT_IS_LONG  1\n"
#elif defined(FCS_INT_IS_LONG_LONG)
# error FCS int data type long long not supported by OpenCL
#else
# error FCS int data type is unknown
#endif
  "\n"
  ;

const char *fcs_ocl_cl =
#include "fcs_ocl.cl_str.h"
  ;

const char *fcs_ocl_math_cl =
#include "fcs_math.cl_str.h"
  ;


fcs_int fcs_ocl_init(fcs_ocl_t *ocl)
{
  DEBUG_CMD(printf(DEBUG_PRINT_PREFIX "initializing OpenCL\n"););

  return 0;
}


fcs_int fcs_ocl_tune(fcs_ocl_t *ocl)
{
  DEBUG_CMD(printf(DEBUG_PRINT_PREFIX "tuning OpenCL\n"););

  cl_uint nplatforms;
  clGetPlatformIDs(0, NULL, &nplatforms);

  return 0;
}


fcs_int fcs_ocl_destroy(fcs_ocl_t *ocl)
{
  DEBUG_CMD(printf(DEBUG_PRINT_PREFIX "destroying OpenCL\n"););

  return 0;
}


fcs_int fcs_ocl_parse_conf(const char *ocl_conf, fcs_int *nunits, fcs_ocl_unit_t *units)
{
  char *conf = strdup(ocl_conf);

  const fcs_int max_nunits = *nunits;

  char *next_unit = conf;
  *nunits = 0;
  while (next_unit && *nunits < max_nunits)
  {
    const char *current_unit = next_unit;
    next_unit = strchr(current_unit, '/');
    if (next_unit) { *next_unit = '\0'; ++next_unit; }
    DEBUG_CMD(printf(DEBUG_PRINT_PREFIX "current unit: '%s'\n", current_unit););

    char *n = strchr(current_unit, ':');
    if (n) { *n = '\0'; ++n; }
    DEBUG_CMD(printf(DEBUG_PRINT_PREFIX "  platform: '%s'\n", current_unit););

    units[*nunits].platform_suffix[0] = '\0';
    units[*nunits].platform_index = -1;
    if (current_unit[0] != '\0')
    {
      fcs_int idx;
      if (sscanf(current_unit, "%" FCS_LMOD_INT "d", &idx) == 1) units[*nunits].platform_index = idx;
      else strncpy(units[*nunits].platform_suffix, current_unit, FCS_OCL_UNIT_STR_SIZE);
    }

    fcs_int current_platform = *nunits;

    if (!n)
    {
      DEBUG_CMD(printf(DEBUG_PRINT_PREFIX "  device: ''\n"););

      units[*nunits].device_type[0] = '\0';
      units[*nunits].device_index = 0;

      ++(*nunits);
    }

    while (n && *nunits < max_nunits)
    {
      const char *c = n;
      n = strchr(c, ':');
      if (n) { *n = '\0'; ++n; }

      DEBUG_CMD(printf(DEBUG_PRINT_PREFIX "  device: '%s'\n", c););

      if (current_platform != *nunits)
      {
        units[*nunits].platform_index = units[current_platform].platform_index;
        strcpy(units[*nunits].platform_suffix, units[current_platform].platform_suffix);
      }

      char *idx = strchr(c, '[');
      if (idx) { *idx = '\0'; ++idx; }

      strcpy(units[*nunits].device_type, c);

      if (idx)
      {
        if (strcmp(idx, "rank")) units[*nunits].device_index = -1;
        else units[*nunits].device_index = atoi(idx);

      } else units[*nunits].device_index = 0;

      ++(*nunits);
    }
  }

  free(conf);

  return 0;
}


static cl_device_type device_str2type(const char *device)
{
  cl_device_type type = CL_DEVICE_TYPE_ALL;

  if (strcmp(device, "default") == 0) type = CL_DEVICE_TYPE_DEFAULT;
  else if (strcmp(device, "cpu") == 0) type = CL_DEVICE_TYPE_CPU;
  else if (strcmp(device, "gpu") == 0) type = CL_DEVICE_TYPE_GPU;
  else if (strcmp(device, "accel") == 0) type = CL_DEVICE_TYPE_ACCELERATOR;
  else if (strcmp(device, "all") == 0) type = CL_DEVICE_TYPE_ALL;

  return type;
}


fcs_int fcs_ocl_get_device(fcs_ocl_unit_t *unit, int comm_rank, cl_device_id *device_id)
{
  cl_int ret;

#define MAX_NPLATFORMS  8
  cl_platform_id platform_ids[MAX_NPLATFORMS];
  cl_uint nplatforms = 0;
  ret = clGetPlatformIDs(MAX_NPLATFORMS, platform_ids, &nplatforms);
#undef MAX_NPLATFORMS

  cl_device_type type = device_str2type(unit->device_type);

  ret = CL_DEVICE_NOT_FOUND;
  fcs_int p = z_max(0, unit->platform_index);
  while (ret != CL_SUCCESS && p < nplatforms)
  {
    if (p == unit->platform_index || unit->platform_index < 0)
    {
#define MAX_NDEVICES  8
      cl_device_id device_ids[MAX_NDEVICES];
      cl_uint ndevices = 0;
      ret = clGetDeviceIDs(platform_ids[p], type, MAX_NDEVICES, device_ids, &ndevices);
#undef MAX_NDEVICES

      fcs_int d = unit->device_index;

      if (d < 0)
      {
        if (d == -1) d = comm_rank;
        else d = 0;
      }

      if (ret == CL_SUCCESS && d < ndevices)
      {
        *device_id = device_ids[d];

#define MAX_NAME_SIZE  256
        INFO_CMD(
          if (comm_rank == 0)
          {
            char platform_name[MAX_NAME_SIZE];
            char device_name[MAX_NAME_SIZE];
            clGetPlatformInfo(platform_ids[p], CL_PLATFORM_NAME, MAX_NAME_SIZE, platform_name, NULL);
            clGetDeviceInfo(*device_id, CL_DEVICE_NAME, MAX_NAME_SIZE, device_name, NULL);

            printf(INFO_PRINT_PREFIX "platform index: '%" FCS_LMOD_INT "d'\n", p);
            printf(INFO_PRINT_PREFIX "platform name: '%s'\n", platform_name);
            printf(INFO_PRINT_PREFIX "device index: '%" FCS_LMOD_INT "d'\n", d);
            printf(INFO_PRINT_PREFIX "device name: '%s'\n", device_name);
          }
        );
#undef MAX_NAME_SIZE

      } else
      {
        ret = CL_DEVICE_NOT_FOUND;
        if (p == unit->platform_index) p = nplatforms;
      }
    }

    ++p;
  }

  return (ret == CL_SUCCESS)?0:1;
}
