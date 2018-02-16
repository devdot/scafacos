/*
  Copyright (C) 2011, 2012, 2013, 2014 Michael Hofmann

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
#include <config.h>
#endif

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <mpi.h>

#include "common/near/near.h"
#include "fcs_common.h"


FCSResult fcs_common_set_parameter(FCS handle, fcs_bool continue_on_errors, char **current, char **next, fcs_int *matched)
{
  char *param = *current;
  char *cur = *next;

  *matched = 0;

  FCS_PARSE_DUMMY();

  return FCS_RESULT_SUCCESS;

next_param:
  *current = param;
  *next = cur;

  *matched = 1;

  return FCS_RESULT_SUCCESS;
}


FCSResult fcs_common_print_parameters(FCS handle)
{
  return FCS_RESULT_SUCCESS;
}


FCSResult fcs_near_set_ocl(fcs_near_param_t *near_param, fcs_int ocl)
{
#if HAVE_OPENCL
  if (fcs_near_param_set_ocl(near_param, ocl) == 0) return FCS_RESULT_SUCCESS;
#endif /* HAVE_OPENCL */

  return FCS_RESULT_FAILURE;
}


FCSResult fcs_near_set_ocl_conf(fcs_near_param_t *near_param, const char *ocl_conf)
{
#if HAVE_OPENCL
  if (fcs_near_param_set_ocl_conf(near_param, ocl_conf) == 0) return FCS_RESULT_SUCCESS;
#endif /* HAVE_OPENCL */

  return FCS_RESULT_FAILURE;
}


FCSResult fcs_near_set_parameter(fcs_near_param_t *near_param, fcs_bool continue_on_errors, char **current, char **next, fcs_int *matched)
{
  char *param = *current;
  char *cur = *next;

  *matched = 0;

  FCS_PARSE_DUMMY();

#define handle  near_param

  FCS_PARSE_IF_PARAM_THEN_FUNC1_GOTO_NEXT("near_ocl",      near_set_ocl,      FCS_PARSE_VAL(fcs_int));
  FCS_PARSE_IF_PARAM_THEN_FUNC1_GOTO_NEXT("near_ocl_conf", near_set_ocl_conf, FCS_PARSE_VAL(fcs_p_char_t));

#undef handle

  return FCS_RESULT_SUCCESS;

next_param:
  *current = param;
  *next = cur;

  *matched = 1;

  return FCS_RESULT_SUCCESS;
}


FCSResult fcs_near_print_parameters(fcs_near_param_t *near_param)
{
  return FCS_RESULT_SUCCESS;
}
