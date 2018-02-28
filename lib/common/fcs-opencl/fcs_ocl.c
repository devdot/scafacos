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

#define CL_USE_DEPRECATED_OPENCL_1_2_APIS
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include "fcs_ocl.h"


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
