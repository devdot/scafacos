#ifndef __NEAR_SORT_H__
#define __NEAR_SORT_H__

#ifdef __cplusplus
extern "C" {
#endif

#define FCS_NEAR_OCL_SORT 1
#if !HAVE_OPENCL
# undef FCS_NEAR_OCL_SORT
#endif

#if FCS_NEAR_OCL_SORT

#define FCS_NEAR_OCL_SORT_CHECK 1

// configuration for radix sort
#define FCS_NEAR_OCL_SORT_RADIX       4
#define FCS_NEAR_OCL_SORT_RADIX_BITS  2

/*
 * MACROS
 */

#define STR_HELPER(x) #x
#define STR(x) STR_HELPER(x)

#define max(a, b) \
   ({ __typeof__ (a) _a = (a); \
       __typeof__ (b) _b = (b); \
     _a > _b ? _a : _b; })


#endif /* FCS_NEAR_OCL_SORT */

#ifdef __cplusplus
}
#endif

#endif /* __NEAR_SORT_H__ */
