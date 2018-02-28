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

#ifndef __FCS_MATH_H__
#define __FCS_MATH_H__


#undef FCS_CONST
#undef FCS_MATH

#if defined(FCS_FLOAT_IS_FLOAT)
# define FCS_CONST(_c_)  _c_##F
# define FCS_MATH(_f_)   _f_##f
#elif defined(FCS_FLOAT_IS_DOUBLE)
# define FCS_CONST(_c_)  _c_
# define FCS_MATH(_f_)   _f_
#elif defined(FCS_FLOAT_IS_LONG_DOUBLE)
# define FCS_CONST(_c_)  _c_##L
# define FCS_MATH(_f_)   _f_##l
#else
# error FCS float data type is unknown
#endif

/**
 * @brief definitions of mathematical constants with FCS namespace (from gcc's math.h)
 */
#define FCS_E         FCS_CONST(2.7182818284590452353602874713526625)  /* e */
#define FCS_LOG2E     FCS_CONST(1.4426950408889634073599246810018921)  /* log_2 e */
#define FCS_LOG10E    FCS_CONST(0.4342944819032518276511289189166051)  /* log_10 e */
#define FCS_LN2       FCS_CONST(0.6931471805599453094172321214581766)  /* log_e 2 */
#define FCS_LN10      FCS_CONST(2.3025850929940456840179914546843642)  /* log_e 10 */
#define FCS_PI        FCS_CONST(3.1415926535897932384626433832795029)  /* pi */
#define FCS_PI_2      FCS_CONST(1.5707963267948966192313216916397514)  /* pi/2 */
#define FCS_PI_4      FCS_CONST(0.7853981633974483096156608458198757)  /* pi/4 */
#define FCS_1_PI      FCS_CONST(0.3183098861837906715377675267450287)  /* 1/pi */
#define FCS_2_PI      FCS_CONST(0.6366197723675813430755350534900574)  /* 2/pi */
#define FCS_2_SQRTPI  FCS_CONST(1.1283791670955125738961589031215452)  /* 2/sqrt(pi) */
#define FCS_SQRT2     FCS_CONST(1.4142135623730950488016887242096981)  /* sqrt(2) */
#define FCS_SQRT1_2   FCS_CONST(0.7071067811865475244008443621048490)  /* 1/sqrt(2) */

/**
 * @brief definitions of mathematical constants with FCS namespace (not included in gcc's math.h)
 */
#define FCS_1_SQRTPI  FCS_CONST(0.5641895835477562869480794515607726)  /* 1/sqrt(pi) */
#define FCS_SQRTPI    FCS_CONST(1.7724538509055160272981674833411452)  /* sqrt(pi) */
#define FCS_PISQR     FCS_CONST(9.8696044010893586188344909998761511)  /* pi^2 */
#define FCS_EULER     FCS_CONST(0.5772156649015328606065120900824024)  /* Euler-Mascheroni constant */

/**
 * @brief definitions of mathematical functions with FCS namespace
 */
#define fcs_sqrt(_x_)      FCS_MATH(sqrt)(_x_)
#define fcs_fabs(_x_)      FCS_MATH(fabs)(_x_)
#define fcs_floor(_x_)     FCS_MATH(floor)(_x_)
#define fcs_ceil(_x_)      FCS_MATH(ceil)(_x_)
#define fcs_exp(_x_)       FCS_MATH(exp)(_x_)
#define fcs_sin(_x_)       FCS_MATH(sin)(_x_)
#define fcs_cos(_x_)       FCS_MATH(cos)(_x_)
#define fcs_sinh(_x_)      FCS_MATH(sinh)(_x_)
#define fcs_cosh(_x_)      FCS_MATH(cosh)(_x_)
#define fcs_log(_x_)       FCS_MATH(log)(_x_)
#define fcs_erf(_x_)       FCS_MATH(erf)(_x_)
#define fcs_erfc(_x_)      FCS_MATH(erfc)(_x_)
#define fcs_pow(_x_, _y_)  FCS_MATH(pow)(_x_, _y_)
#define fcs_creal(_x_)     FCS_MATH(creal)(_x_)
#define fcs_cimag(_x_)     FCS_MATH(cimag)(_x_)

#define fcs_isnan(_x_)     isnan(_x_)
#define fcs_isinf(_x_)     isinf(_x_)

#define fcs_xabs(_x_)      (((_x_) >= 0)?(_x_):(_x_))


#endif /* __FCS_MATH_H__ */
