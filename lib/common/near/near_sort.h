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
 * TOGGLES AND CONFIGURATION
 */

#define FCS_NEAR_OCL_SORT_ABORT               0   // aborts the programm after sorting is done (obviously use only for debugging/timing)
#define FCS_NEAR_OCL_SORT_USE_INDEX           1   // set 0 for using swap along, 1 for data index
#define FCS_NEAR_OCL_SORT_MOVE_ON_HOST        0   // 1 if data should only be moved on host (only applies to data index)
#define FCS_NEAR_OCL_SORT_MOVE_AUTO           1   // 1 for automatic usage of split move when the defined limit is reached
#define FCS_NEAR_OCL_SORT_MOVE_SPLIT_N        1024 * 1024 * 32 // 2^25, threshold until data will be moved all at the same time, only when moving data on gpu without auto-mode
#define FCS_NEAR_OCL_SORT_NO_SWAP_ON_EQUAL    1   // set to 0 if comparisons should not check for equal exception
#define FCS_NEAR_OCL_SORT_KEEP_BUFFERS        1   // set to 1 if buffers should stay on device for ocl compute
#define FCS_NEAR_OCL_SORT_USE_SUBBUFFERS      1   // will use OpenCL Subbuffers only for most important cases if 0, needs to be 0 for most CPUs

#define FCS_NEAR_OCL_DATA_INDEX_IS_INT        0
#define FCS_NEAR_OCL_DATA_INDEX_IS_LONG_LONG  1
#define FCS_NEAR_OCL_SORT_HISTOGRAM_IS_INT        1
#define FCS_NEAR_OCL_SORT_HISTOGRAM_IS_LONG_LONG  0

#define FCS_NEAR_OCL_SORT_WORKGROUP_MAX 1024  // should not be higher than the workgroup max the the device
#define FCS_NEAR_OCL_SORT_WORKGROUP_MIN 64    // for very small number of elements this limit may be violated

// configuration for radix sort
#define FCS_NEAR_OCL_SORT_RADIX_BITS      3
#define FCS_NEAR_OCL_SORT_RADIX_QUOTA     16  // amout of sort items per workitem
#define FCS_NEAR_OCL_SORT_RADIX_TRANSPOSE 1   // only when quota > 1 !!
#define FCS_NEAR_OCL_SORT_RADIX_SCALE     1   // 1 to activate multilevel scan
// radix is automatic
#define FCS_NEAR_OCL_SORT_RADIX (1 << FCS_NEAR_OCL_SORT_RADIX_BITS)

// configuration for bitonic sort
#define FCS_NEAR_OCL_SORT_BITONIC_GLOBAL_4  1
#define FCS_NEAR_OCL_SORT_BITONIC_GLOBAL_8  1
#define FCS_NEAR_OCL_SORT_BITONIC_GLOBAL_16 1
#define FCS_NEAR_OCL_SORT_BITONIC_GLOBAL_32 0

// configuration for hybrid sort
#define FCS_NEAR_OCL_SORT_HYBRID_WORKGROUP_MAX  256 // should not be higher than FCS_NEAR_OCL_SORT_WORKGROUP_MAX
#define FCS_NEAR_OCL_SORT_HYBRID_MIN_QUOTA      2   // minimum quota (number of elements per workitem), must be power of 2 and >= 2
#define FCS_NEAR_OCL_SORT_HYBRID_MAX_QUOTA      2   // maximum quota, must be power of 2, -1 for none
#define FCS_NEAR_OCL_SORT_HYBRID_INDEX_GLOBAL   0   // set to 1 if the local kernel should keep data index in global memory
#define FCS_NEAR_OCL_SORT_HYBRID_COALESCE       1   // set to 0 if the local kernel should not coalesce memory reads
#define FCS_NEAR_OCL_SORT_HYBRID_PAIRWISE       0   // set to 1 if the local kernel should do pairwise swaps (requires lots of sychronization)

// configuration for bucket sort
#define FCS_NEAR_OCL_SORT_BUCKET_USE_RADIX        0   // if this is 1, bucket sort will use radix for sorting the buckets
#define FCS_NEAR_OCL_SORT_BUCKET_MULTIQUEUE       1   // set to 0 to deactivate the usage of multiqueue for sorting the buckts
#define FCS_NEAR_OCL_SORT_BUCKET_INDEXER_LOCAL    1   // set to 0 if the indexer kernel should not work on local memory
#define FCS_NEAR_OCL_SORT_BUCKET_LOCAL_SAMPLES    8  // local sample rate, must be power of 2
#define FCS_NEAR_OCL_SORT_BUCKET_GLOBAL_SAMPLES   128  // global sample rate, is number of buckets, must be power of 2
#define FCS_NEAR_OCL_SORT_BUCKET_MIN_OFFSET       1   // minimize the offset of bucket sort (may cause problems with sampling that can be solved with other optimizations)
// set only either to true
#define FCS_NEAR_OCL_SORT_BUCKET_SKEW_SAMPLES     1   // set to 1 to skew sampling area so that number of zero-samples is reduced
#define FCS_NEAR_OCL_SORT_BUCKET_OPTIMIZE_OFFSET  0   // set to 1 to manipulate sample selection so that all zero-elements are put in the first bucket that is then discarded

// configuration for auto-mode
#define FCS_NEAR_OCL_SORT_AUTO_LOW_ALGO         4                 // use hybrid for smaller n
#define FCS_NEAR_OCL_SORT_AUTO_MAIN_ALGO        5                 // use bucket as main algorithm
#define FCS_NEAR_OCL_SORT_AUTO_HIGH_ALGO        6                 // use radix for high n
#define FCS_NEAR_OCL_SORT_AUTO_SCALE_ALGO       4                 // use hybrid again when radix can't do it anymore
#define FCS_NEAR_OCL_SORT_AUTO_MAIN_THRESHOLD   (1 << 22)         // 2^22, just above 24^3 * 300 and below 25^3 * 300
#define FCS_NEAR_OCL_SORT_AUTO_HIGH_THRESHOLD   (76*76*76) * 300  // 76^3 * 300 is the last we can confidently do (just below 2^27)
#define FCS_NEAR_OCL_SORT_AUTO_SCALE_THRESHOLD  (85*85*85) * 300  // 85^3 * 300 is the last that radix can do for sure

#if FCS_NEAR_OCL_SORT_BUCKET_SKEW_SAMPLES && FCS_NEAR_OCL_SORT_BUCKET_OPTIMIZE_OFFSET
#error Cannot enable both optimizations at once
#endif

#if !FCS_NEAR_OCL_SORT_USE_INDEX && FCS_NEAR_OCL_SORT_RADIX_TRANSPOSE
#warning Radix Sort cannot run with transpose and swap along
#endif

// enum for algo types
#define FCS_NEAR_OCL_SORT_ALGO_AUTO           0
#define FCS_NEAR_OCL_SORT_ALGO_BITONIC        2
#define FCS_NEAR_OCL_SORT_ALGO_HYBRID         4
#define FCS_NEAR_OCL_SORT_ALGO_BUCKET         5
#define FCS_NEAR_OCL_SORT_ALGO_RADIX          6

#ifdef FCS_ENABLE_CHECK_NEAR
#define DO_CHECK
#endif

/**
 * @brief Sort the particles and ghosts into their associated boxes
 * 
 * @param near fcs_near_t* Near field solver object
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

#define min(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a < _b ? _a : _b; })

#if FCS_NEAR_OCL_DATA_INDEX_IS_INT
  typedef int sort_index_t;
#elif FCS_NEAR_OCL_DATA_INDEX_IS_LONG_LONG
  typedef long long sort_index_t;
#else
# error Type for box_t not available
#endif

#if FCS_NEAR_OCL_SORT_HISTOGRAM_IS_INT
  typedef unsigned int histogram_t;
#elif FCS_NEAR_OCL_SORT_HISTOGRAM_IS_LONG_LONG
  typedef unsigned long long histogram_t;
#else
# error Type for histogram_t not available
#endif


#endif /* FCS_NEAR_OCL_SORT */

#ifdef __cplusplus
}
#endif

#endif /* __NEAR_SORT_H__ */
