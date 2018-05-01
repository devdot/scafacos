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

#ifndef __FCS_OCL_CL_H__
#define __FCS_OCL_CL_H__


#if defined(FCS_FLOAT_IS_FLOAT)
# define fcs_float  float
#elif defined(FCS_FLOAT_IS_DOUBLE)
# define fcs_float  double
# if defined(cl_khr_fp64)
#  if __OPENCL_VERSION__ <= CL_VERSION_1_1
#   pragma OPENCL EXTENSION cl_khr_fp64 : enable
#  endif
# else
#  error OpenCL extension cl_khr_fp64 not available
# endif
#elif defined(FCS_FLOAT_IS_LONG_DOUBLE)
# define fcs_float  long double
# error FCS float data type long double not supported by OpenCL
#else
# error FCS float data type is unknown
#endif

#if defined(FCS_INT_IS_SHORT)
# define fcs_int  short
#elif defined(FCS_INT_IS_INT)
# define fcs_int  int
#elif defined(FCS_INT_IS_LONG)
# define fcs_int  long
#elif defined(FCS_INT_IS_LONG_LONG)
# define fcs_int  long long
# error FCS int data type long long not supported by OpenCL
#else
# error FCS int data type is unknown
#endif


/**
 * @brief redirect math functions used by fcs_math.h to the generic versions of OpenCL
 */
#if defined(FCS_FLOAT_IS_FLOAT)
# define sqrtf(_x_)      sqrt(_x_)
# define fabsf(_x_)      fabs(_x_)
# define floorf(_x_)     floor(_x_)
# define ceilf(_x_)      ceil(_x_)
# define expf(_x_)       exp(_x_)
# define sinf(_x_)       sin(_x_)
# define cosf(_x_)       cos(_x_)
# define sinhf(_x_)      sinh(_x_)
# define coshf(_x_)      cosh(_x_)
# define logf(_x_)       log(_x_)
# define erff(_x_)       erf(_x_)
# define erfcf(_x_)      erfc(_x_)
# define powf(_x_, _y_)  pow(_x_, _y_)
#endif


#endif /* __FCS_OCL_CL_H__ */
