/*
  Copyright (C) 2018 Thomas Schaller
  
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

#ifndef __NEAR_SORT_H__
#define __NEAR_SORT_H__

#ifdef __cplusplus
extern "C" {
#endif

#include "near.h"

#if FCS_NEAR_OCL_SORT

/*
 * TOGGLES
 */

#define FCS_NEAR_OCL_SORT_MOVE_ON_HOST        0
#define FCS_NEAR_OCL_SORT_MOVE_SPLIT_N        1024 * 1024 * 16 // 2^24, only when moving data on gpu
#define FCS_NEAR_OCL_SORT_NO_SWAP_ON_EQUAL    1

#define FCS_NEAR_OCL_DATA_INDEX_IS_INT        0
#define FCS_NEAR_OCL_DATA_INDEX_IS_LONG_LONG  1

#define FCS_NEAR_OCL_SORT_WORKGROUP_MAX 1024
#define FCS_NEAR_OCL_SORT_WORKGROUP_MIN 64

// configuration for radix sort
#define FCS_NEAR_OCL_SORT_RADIX       4
#define FCS_NEAR_OCL_SORT_RADIX_BITS  2

// configuration for hybrid sort
#define FCS_NEAR_OCL_SORT_HYBRID_INDEX_GLOBAL 0

// configuration for bucket sort
// set only either to true
#define FCS_NEAR_OCL_SORT_BUCKET_SKEW_SAMPLES    0
#define FCS_NEAR_OCL_SORT_BUCKET_OPTIMIZE_OFFSET 1

#if FCS_NEAR_OCL_SORT_BUCKET_SKEW_SAMPLES && FCS_NEAR_OCL_SORT_BUCKET_OPTIMIZE_OFFSET
#error Cannot enable both optimizations at once
#endif

// enum for algo types
#define FCS_NEAR_OCL_SORT_ALGO_RADIX          0
#define FCS_NEAR_OCL_SORT_ALGO_BITONIC        1
#define FCS_NEAR_OCL_SORT_ALGO_BITONIC_INDEX  2
#define FCS_NEAR_OCL_SORT_ALGO_HYBRID         3
#define FCS_NEAR_OCL_SORT_ALGO_HYBRID_INDEX   4
#define FCS_NEAR_OCL_SORT_ALGO_BUCKET         5

#ifdef FCS_ENABLE_CHECK_NEAR
#define DO_CHECK
#endif

/**
 * @brief sort particles, ghost particles and associated data into boxes using OpenCl
 * @param fcs_near_t* near field solver object
 */
void fcs_ocl_sort(fcs_near_t* near);

/*
 * MACROS
 */

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define max(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


#if FCS_NEAR_OCL_DATA_INDEX_IS_INT
  typedef int sort_index_t;
#elif FCS_NEAR_OCL_DATA_INDEX_IS_LONG_LONG
  typedef long long sort_index_t;
#else
# error Type for box_t not available
#endif


#endif /* FCS_NEAR_OCL_SORT */

#ifdef __cplusplus
}
#endif

#endif /* __NEAR_SORT_H__ */
