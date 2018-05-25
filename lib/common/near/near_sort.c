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

#include "near_common.h"

#if FCS_NEAR_OCL_SORT

/*
 * Macros
 */

#ifdef DO_TIMING

#define T_START(index, str) { ocl->timing_names[index] = str; TIMING_START(ocl->_timing[index]); }
#define T_STOP(index) { TIMING_STOP(ocl->_timing[index]); }

#endif // DO_TIMING

/*
 * OpenCL kernel strings
 */
static const char *fcs_ocl_cl_sort_config =

  // OpenCL long is equal to C99 long long
  "typedef long fcs_gridsort_index_t;\n"
  // for data index arrays
#ifdef FCS_NEAR_OCL_DATA_INDEX_IS_INT
  "typedef int index_t;\n"
#else
# error Type for box_t not available
#endif
  // key type (OpenCL long is 64bits)
#if FCS_NEAR_BOX_IS_LONG_LONG
  "typedef long key_t;\n"
#else
# error Type for box_t not available
#endif

  "#define RADIX " STR(FCS_NEAR_OCL_SORT_RADIX) "\n"
  "#define RADIX_BITS " STR(FCS_NEAR_OCL_SORT_RADIX_BITS) "\n"
  ;
  // built in Makefile.am/.in  like near.cl_str.h
static const char* fcs_ocl_cl_sort =
#include "near_sort.cl_str.h"
  ;

static const char* fcs_ocl_cl_sort_bitonic = 
#include "near_sort_bitonic.cl_str.h"
  ;

static const char* fcs_ocl_cl_sort_hybrid = 
#include "near_sort_hybrid.cl_str.h"
  ;

static const char* fcs_ocl_cl_sort_radix =
#include "near_sort_radix.cl_str.h"
  ;

/*
 * helper functions
 */

#ifdef DO_CHECK
static int fcs_ocl_sort_check(fcs_int n, box_t* boxes) {
  // check the sort results to be correct
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-sort: check sort\n"););
  for(int i = 1; i < n; i++) {
    if(boxes[i] < boxes[i - 1]) {
      // we got a fail in sorting
      printf("ocl-sort: failed to sort correctly at element #%d, value %lld\n", i, boxes[i]);
      abort();
    }
  }
  return 1;
}
#endif

unsigned int fcs_ocl_helper_next_power_of_2(unsigned int n) {
  unsigned count = 0;

  // already a power of 2?
  if (n && !(n & (n - 1)))
    return n;
 
  while(n != 0) {
    n  >>= 1;
    count += 1;
  }
 
  return 1 << count;
}

void fcs_ocl_sort_move_data(fcs_ocl_context_t *ocl, fcs_int nlocal, int offset, cl_mem mem_data, fcs_float *positions, fcs_float *charges, fcs_gridsort_index_t *indices, fcs_float *field, fcs_float *potentials)
{
  const size_t global_size_move_data = nlocal;

  // make new buffers for data
  T_START(41, "move_data_write");
  cl_mem mem_positionsIn  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, nlocal * 3 * sizeof(fcs_float), NULL, &_err));
  cl_mem mem_positionsOut = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, nlocal * 3 * sizeof(fcs_float), NULL, &_err));
  cl_mem mem_chargesIn    = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, nlocal * sizeof(fcs_float), NULL, &_err));
  cl_mem mem_chargesOut   = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, nlocal * sizeof(fcs_float), NULL, &_err));
  cl_mem mem_indicesIn   = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, nlocal * sizeof(fcs_gridsort_index_t), NULL, &_err));
  cl_mem mem_indicesOut  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, nlocal * sizeof(fcs_gridsort_index_t), NULL, &_err));
  cl_mem mem_fieldIn = NULL, mem_fieldOut = NULL;
  if(field != NULL) {
    mem_fieldIn   = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, nlocal * 3 * sizeof(fcs_float), NULL, &_err));
    mem_fieldOut  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, nlocal * 3 * sizeof(fcs_float), NULL, &_err));
  }
  cl_mem mem_potentialsIn = NULL, mem_potentialsOut = NULL;
  if(potentials != NULL) {
    mem_potentialsIn  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_ONLY, nlocal * sizeof(fcs_float), NULL, &_err));
    mem_potentialsOut = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, nlocal * sizeof(fcs_float), NULL, &_err));
  }

  // write in buffers
  CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, mem_positionsIn,  CL_FALSE, 0, nlocal * 3 * sizeof(fcs_float), positions, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, mem_chargesIn,    CL_FALSE, 0, nlocal * sizeof(fcs_float), charges, 0, NULL, NULL));
  CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, mem_indicesIn,    CL_FALSE, 0, nlocal * sizeof(fcs_gridsort_index_t), indices, 0, NULL, NULL));
  if(field != NULL)
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, mem_fieldIn,      CL_FALSE, 0, nlocal * 3 * sizeof(fcs_float), field, 0, NULL, NULL));
  if(potentials != NULL)
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, mem_potentialsIn, CL_FALSE, 0, nlocal * sizeof(fcs_float), potentials, 0, NULL, NULL));

  CL_CHECK(clFinish(ocl->command_queue));
  T_STOP(41);

  // now move the data arrays around
  T_START(42, "move_data_move");
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float, 0, sizeof(cl_mem), &mem_data));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float, 1, sizeof(int), &offset));

  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float, 2, sizeof(cl_mem), &mem_chargesIn));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float, 3, sizeof(cl_mem), &mem_chargesOut));
  CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_move_data_float, 1, NULL, &global_size_move_data, NULL, 0, NULL, &ocl->sort_kernel_completion));
  if(potentials != NULL) {
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float, 2, sizeof(cl_mem), &mem_potentialsIn));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float, 3, sizeof(cl_mem), &mem_potentialsOut));
    CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_move_data_float, 1, NULL, &global_size_move_data, NULL, 0, NULL, &ocl->sort_kernel_completion));
  }
  // triples
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float_triple, 0, sizeof(cl_mem), &mem_data));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float_triple, 1, sizeof(int), &offset));

  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float_triple, 2, sizeof(cl_mem), &mem_positionsIn));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float_triple, 3, sizeof(cl_mem), &mem_positionsOut));
  CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_move_data_float_triple, 1, NULL, &global_size_move_data, NULL, 0, NULL, &ocl->sort_kernel_completion));
  if(field != NULL) {
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float_triple, 2, sizeof(cl_mem), &mem_fieldIn));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_float_triple, 3, sizeof(cl_mem), &mem_fieldOut));
    CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_move_data_float_triple, 1, NULL, &global_size_move_data, NULL, 0, NULL, &ocl->sort_kernel_completion));
  }
  // gridsort_index
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_gridsort_index, 0, sizeof(cl_mem), &mem_data));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_gridsort_index, 1, sizeof(int), &offset));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_gridsort_index, 2, sizeof(cl_mem), &mem_indicesIn));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_move_data_gridsort_index, 3, sizeof(cl_mem), &mem_indicesOut));
  CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_move_data_gridsort_index, 1, NULL, &global_size_move_data, NULL, 0, NULL, &ocl->sort_kernel_completion));

  // let the kernels all finish the movement
  CL_CHECK(clFinish(ocl->command_queue));
  T_STOP(42);

  // read back the buffers for data
  T_START(43, "move_data_read");
  CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, mem_positionsOut, CL_FALSE, 0, nlocal * 3 * sizeof(fcs_float), positions, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, mem_chargesOut, CL_FALSE, 0, nlocal * sizeof(fcs_float), charges, 0, NULL, NULL));
  CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, mem_indicesOut, CL_FALSE, 0, nlocal * sizeof(fcs_gridsort_index_t), indices, 0, NULL, NULL));
  if(field != NULL)
    CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, mem_fieldOut, CL_FALSE, 0, nlocal * 3 * sizeof(fcs_float), field, 0, NULL, NULL));
  if(potentials != NULL)
    CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, mem_potentialsOut, CL_FALSE, 0, nlocal * sizeof(fcs_float), potentials, 0, NULL, NULL));

  CL_CHECK(clFinish(ocl->command_queue));
  T_STOP(43);

  // release our buffers
  CL_CHECK(clReleaseMemObject(mem_positionsIn));
  CL_CHECK(clReleaseMemObject(mem_positionsOut));
  CL_CHECK(clReleaseMemObject(mem_chargesIn));
  CL_CHECK(clReleaseMemObject(mem_chargesOut));
  CL_CHECK(clReleaseMemObject(mem_indicesIn));
  CL_CHECK(clReleaseMemObject(mem_indicesOut));
  if(field != NULL) {
    CL_CHECK(clReleaseMemObject(mem_fieldIn));
    CL_CHECK(clReleaseMemObject(mem_fieldOut));
  }
  if(potentials != NULL) {
    CL_CHECK(clReleaseMemObject(mem_potentialsIn));
    CL_CHECK(clReleaseMemObject(mem_potentialsOut));
  }
}


/*
 * radix sort
 */

static void fcs_ocl_sort_radix_prepare(fcs_ocl_context_t *ocl) {
  cl_int ret;

  // combine the program
  const char* sources[] = {
    fcs_ocl_cl_config,
    fcs_ocl_cl,
    fcs_ocl_math_cl,
    fcs_ocl_cl_sort_config,
    fcs_ocl_cl_sort,
    fcs_ocl_cl_sort_radix
  };

  ocl->sort_program_radix = CL_CHECK_ERR(clCreateProgramWithSource(ocl->context, sizeof(sources) / sizeof(sources[0]), sources, NULL, &_err));

  ret = clBuildProgram(ocl->sort_program_radix, 1, &ocl->device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    size_t length;
    char buffer[32*1024];
    CL_CHECK(clGetProgramBuildInfo(ocl->sort_program_radix, ocl->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length));
    printf("ocl-radix: failed to build radix program %d\nocl build info: %.*s\n", ret, (int) length, buffer);
    return;
  }

  // build all the kernels
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-radix: building kernels\n"););
  ocl->sort_kernel_radix_histogram          = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_radix, "radix_histogram", &_err));
  ocl->sort_kernel_radix_histogram_paste    = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_radix, "radix_histogram_paste", &_err));
  ocl->sort_kernel_radix_scan               = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_radix, "radix_scan", &_err));
  ocl->sort_kernel_radix_reorder            = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_radix, "radix_reorder", &_err));

  ocl->sort_kernel_move_data_float          = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_radix, "move_data_float", &_err));
  ocl->sort_kernel_move_data_float_triple   = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_radix, "move_data_float_triple", &_err));
  ocl->sort_kernel_move_data_gridsort_index = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_radix, "move_data_gridsort_index", &_err));
}

static void fcs_ocl_sort_radix_release(fcs_ocl_context_t *ocl) {
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-radix: releasing\n"););
  CL_CHECK(clReleaseKernel(ocl->sort_kernel_radix_histogram));
  CL_CHECK(clReleaseKernel(ocl->sort_kernel_radix_histogram_paste));
  CL_CHECK(clReleaseKernel(ocl->sort_kernel_radix_scan));
  CL_CHECK(clReleaseKernel(ocl->sort_kernel_radix_reorder));

  CL_CHECK(clReleaseKernel(ocl->sort_kernel_move_data_float));
  CL_CHECK(clReleaseKernel(ocl->sort_kernel_move_data_float_triple));
  CL_CHECK(clReleaseKernel(ocl->sort_kernel_move_data_gridsort_index));

  CL_CHECK(clReleaseProgram(ocl->sort_program_radix));
}


static void fcs_ocl_sort_radix(fcs_ocl_context_t *ocl, fcs_int nlocal, box_t *boxes, fcs_float *positions, fcs_float *charges, fcs_gridsort_index_t *indices, fcs_float *field, fcs_float *potentials)
{
  // workgroup sizes
  size_t local_size = FCS_NEAR_OCL_SORT_WORKGROUP_MAX;
  size_t scan_size  = FCS_NEAR_OCL_SORT_WORKGROUP_MAX;

  // auto scale for normal groups
  if(nlocal / 4 < FCS_NEAR_OCL_SORT_WORKGROUP_MAX) {
      local_size = fcs_ocl_helper_next_power_of_2(nlocal / 8);

      if(local_size < FCS_NEAR_OCL_SORT_WORKGROUP_MIN)
        local_size = FCS_NEAR_OCL_SORT_WORKGROUP_MIN;
  }

  int n = (nlocal % local_size == 0)? n : nlocal + local_size - (nlocal % local_size);
  int offset = n - nlocal;

  // auto scale for scan (following the fomula)
  scan_size = fcs_ocl_helper_next_power_of_2(FCS_NEAR_OCL_SORT_RADIX * n) / (2 * FCS_NEAR_OCL_SORT_WORKGROUP_MAX);
  if(scan_size < FCS_NEAR_OCL_SORT_WORKGROUP_MIN)
    scan_size = FCS_NEAR_OCL_SORT_WORKGROUP_MIN;

  // calculate work sizes
  const size_t global_size_histogram  = n;
  const size_t local_size_histogram   = local_size;

  const size_t global_size_scan   = FCS_NEAR_OCL_SORT_RADIX * n / 2;
  const size_t local_size_scan    = scan_size / 2;

  const size_t scan2_groups_real  = FCS_NEAR_OCL_SORT_RADIX * n / scan_size;
  const size_t scan2_groups       = fcs_ocl_helper_next_power_of_2(scan2_groups_real);
  const size_t scan_buffer_size   = max(scan2_groups_real, scan_size);

  const size_t global_size_scan2  = scan2_groups / 2;
  const size_t local_size_scan2   = global_size_scan2;

  const size_t global_size_histogram_paste = FCS_NEAR_OCL_SORT_RADIX * n / 2;
  const size_t local_size_histogram_paste  = scan_size / 2;

  const size_t global_size_reorder = n;
  const size_t local_size_reorder  = local_size;
  INFO_CMD(
    printf(INFO_PRINT_PREFIX "ocl-radix: Sort %d => %d elements with radixsort\n", nlocal, n);
    printf(INFO_PRINT_PREFIX "ocl-radix: Radix: %d (%dbits)\n", FCS_NEAR_OCL_SORT_RADIX, FCS_NEAR_OCL_SORT_RADIX_BITS);
    printf(INFO_PRINT_PREFIX "ocl-radix: %ld groups, %ld elements each\n", n / local_size, local_size);
    printf(INFO_PRINT_PREFIX "ocl-radix: Scan: %ld groups, %ld elements each\n", global_size_scan / local_size_scan, scan_size);
    printf(INFO_PRINT_PREFIX "ocl-radix: Scan2: %ld => %ld elements \n", scan2_groups_real, scan2_groups);
  );

  if(local_size_scan > FCS_NEAR_OCL_SORT_WORKGROUP_MAX) {
    printf("local size for scan %ld exceeds maximum size %d\n", local_size_scan, FCS_NEAR_OCL_SORT_WORKGROUP_MAX);
    abort();
  }

  if(local_size_scan2 > FCS_NEAR_OCL_SORT_WORKGROUP_MAX) {
    printf("local size for scan2 %ld exceeds maximum size %d\n", local_size_scan2, FCS_NEAR_OCL_SORT_WORKGROUP_MAX);
    abort();
  }

  // create buffers
  T_START(11, "write_buffers");
  cl_mem mem_keys       = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(box_t), NULL, &_err));
  cl_mem mem_keys_swap  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(box_t), NULL, &_err));

  cl_mem mem_data       = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(data_index_t), NULL, &_err));
  cl_mem mem_data_swap  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(data_index_t), NULL, &_err));

  cl_mem mem_histograms = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE,  sizeof(int) * FCS_NEAR_OCL_SORT_RADIX * n, NULL, &_err));
  cl_mem mem_histograms_sum = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(int) * scan2_groups, NULL, &_err));
  cl_mem mem_histograms_sum_tmp = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, sizeof(int) * scan2_groups, NULL, &_err));


  // write keys to buffer
  CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, mem_keys, CL_FALSE, offset * sizeof(box_t), nlocal * sizeof(box_t), boxes, 0, NULL, NULL));
  // fill up the front with zeros
  const int zero = 0;
  CL_CHECK(clEnqueueFillBuffer(ocl->command_queue, mem_keys, &zero, sizeof(zero), 0, offset * sizeof(box_t), 0, NULL, NULL));

  // let it finish
  CL_CHECK(clFinish(ocl->command_queue));
  T_STOP(11);

  // set kernel arguments
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_histogram, 1, sizeof(cl_mem), &mem_histograms));
  // local buffer for histograms
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_histogram, 2, sizeof(int) * FCS_NEAR_OCL_SORT_RADIX * local_size_histogram, NULL));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_histogram, 4, sizeof(int), &n));

  CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_scan, 1, sizeof(int) * scan_buffer_size, NULL));

  CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_histogram_paste, 0, sizeof(cl_mem), &mem_histograms));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_histogram_paste, 1, sizeof(cl_mem), &mem_histograms_sum));

  CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_reorder, 4, sizeof(cl_mem), &mem_histograms));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_reorder, 5, sizeof(int) * FCS_NEAR_OCL_SORT_RADIX * local_size, NULL));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_reorder, 7, sizeof(int), &n));

  // calculate the amount of passes from the datatype of boxes
  int pass_max = (sizeof(box_t) * 8) / FCS_NEAR_OCL_SORT_RADIX_BITS;
  // and start the main loop of radix sort
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-radix: start radix sort\n"););
  T_START(12, "sort");
  for(int pass = 0; pass < pass_max; pass++) {
    // 1. histogram
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_histogram, 0, sizeof(cl_mem), &mem_keys));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_histogram, 3, sizeof(int), &pass));

    CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_radix_histogram, 1, NULL, &global_size_histogram, &local_size_histogram, 0, NULL, &ocl->sort_kernel_completion));
    CL_CHECK(clFinish(ocl->command_queue));

    // 2. scan histogram
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_scan, 0, sizeof(cl_mem), &mem_histograms));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_scan, 2, sizeof(cl_mem), &mem_histograms_sum));

    CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_radix_scan, 1, NULL, &global_size_scan, &local_size_scan, 0, NULL, &ocl->sort_kernel_completion));
    CL_CHECK(clFinish(ocl->command_queue));

    // second scan on histogram_sum
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_scan, 0, sizeof(cl_mem), &mem_histograms_sum));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_scan, 2, sizeof(cl_mem), &mem_histograms_sum_tmp));

    CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_radix_scan, 1, NULL, &global_size_scan2, &local_size_scan2, 0, NULL, &ocl->sort_kernel_completion));
    CL_CHECK(clFinish(ocl->command_queue));

    // 3. paste histograms
    CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_radix_histogram_paste, 1, NULL, &global_size_histogram_paste, &local_size_histogram_paste, 0, NULL, &ocl->sort_kernel_completion));
    CL_CHECK(clFinish(ocl->command_queue));

    // 4. reorder
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_reorder, 0, sizeof(cl_mem), &mem_keys));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_reorder, 1, sizeof(cl_mem), &mem_keys_swap));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_reorder, 2, sizeof(cl_mem), &mem_data));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_reorder, 3, sizeof(cl_mem), &mem_data_swap));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_radix_reorder, 6, sizeof(int), &pass));

    CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_radix_reorder, 1, NULL, &global_size_reorder, &local_size_reorder, 0, NULL, &ocl->sort_kernel_completion));
    CL_CHECK(clFinish(ocl->command_queue));

    // swap the swap buffers
    cl_mem mem_tmp = mem_keys;
    mem_keys = mem_keys_swap;
    mem_keys_swap = mem_tmp;
    mem_tmp = mem_data;
    mem_data = mem_data_swap;
    mem_data_swap = mem_tmp;
  }
  T_STOP(12);

  // read back keys
  T_START(13, "read_keys");
  CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, mem_keys, CL_FALSE, offset * sizeof(box_t), nlocal * sizeof(box_t), boxes, 0, NULL, NULL));
  CL_CHECK(clFinish(ocl->command_queue));
  T_STOP(13);


  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-radix: move data\n"););

  // release unneeded buffers
  CL_CHECK(clReleaseMemObject(mem_keys));
  CL_CHECK(clReleaseMemObject(mem_keys_swap));
  CL_CHECK(clReleaseMemObject(mem_data_swap));
  CL_CHECK(clReleaseMemObject(mem_histograms));
  CL_CHECK(clReleaseMemObject(mem_histograms_sum));
  CL_CHECK(clReleaseMemObject(mem_histograms_sum_tmp));

  T_START(14, "move_data");
  fcs_ocl_sort_move_data(ocl, nlocal, offset, mem_data, positions, charges, indices, field, potentials);
  T_STOP(14);

  // destroy remaining buffers
  CL_CHECK(clReleaseMemObject(mem_data));
}


/*
 * bitonic sort
 */

static void fcs_ocl_sort_bitonic_prepare(fcs_ocl_context_t *ocl) {
  cl_int ret;

  const char* bitonic_use_index = "-D BITONIC_USE_INDEX=1";

  // first combine program
  const char* sources[] = {
    fcs_ocl_cl_config,
    fcs_ocl_cl,
    fcs_ocl_math_cl,
    fcs_ocl_cl_sort_config,
    fcs_ocl_cl_sort,
    fcs_ocl_cl_sort_bitonic
  };

  ocl->sort_program_bitonic = CL_CHECK_ERR(clCreateProgramWithSource(ocl->context, sizeof(sources) / sizeof(sources[0]), sources, NULL, &_err));

  // build the program
  if(ocl->use_index)
    ret = clBuildProgram(ocl->sort_program_bitonic, 1, &ocl->device_id, bitonic_use_index, NULL, NULL);
  else
    ret = clBuildProgram(ocl->sort_program_bitonic, 1, &ocl->device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    // if there are any errors, just print them
    size_t length;
    char buffer[32*1024];
    CL_CHECK(clGetProgramBuildInfo(ocl->sort_program_bitonic, ocl->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length));
    printf("ocl-bitonic: build fail %d\nocl build info: %.*s\n", ret, (int) length, buffer);
    return;
  }

  // finally create the kernel
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-bitonic: creating kernels\n"););
  ocl->sort_kernel_bitonic_global_2 = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_bitonic, "bitonic_global_2", &_err));

  if(ocl->use_index) {
    // create additional kernels
    ocl->sort_kernel_move_data_float          = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_bitonic, "move_data_float", &_err));
    ocl->sort_kernel_move_data_float_triple   = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_bitonic, "move_data_float_triple", &_err));
    ocl->sort_kernel_move_data_gridsort_index = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_bitonic, "move_data_gridsort_index", &_err));
  }
}

static void fcs_ocl_sort_bitonic_release(fcs_ocl_context_t *ocl) {
  cl_int ret;
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-bitonic: releasing\n"););
  // destroy our kernel and program
  CL_CHECK(clReleaseKernel(ocl->sort_kernel_bitonic_global_2));
  CL_CHECK(clReleaseProgram(ocl->sort_program_bitonic));
  if(ocl->use_index) {
    // let the move kernels go
    CL_CHECK(clReleaseKernel(ocl->sort_kernel_move_data_float));
    CL_CHECK(clReleaseKernel(ocl->sort_kernel_move_data_float_triple));
    CL_CHECK(clReleaseKernel(ocl->sort_kernel_move_data_gridsort_index));
  }
}

static void fcs_ocl_sort_bitonic(fcs_ocl_context_t *ocl, fcs_int nlocal, box_t *boxes, fcs_float *positions, fcs_float *charges, fcs_gridsort_index_t *indices, fcs_float *field, fcs_float *potentials)
{
  // sort for param boxes
  // use OpenCL to sort into boxes  
  
  // first get next power of two for bitonic
  int n = fcs_ocl_helper_next_power_of_2(nlocal);
  int offset = n - nlocal;

  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-bitonic: bitonic (use index: %d) [%" FCS_LMOD_INT "d] => [%"  FCS_LMOD_INT "d]\n", ocl->use_index, nlocal, n););

  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-bitonic: initializing buffers\n"););
  // then initialize memory and write to it
  ocl->mem_boxes      = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(box_t), NULL, &_err));
  // data all read/write for swapping
  if(ocl->use_index) {
    ocl->mem_data = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(data_index_t), NULL, &_err));
  }
  else {
    // we offset this too, can't use CL_MEM_USE_HOST_PTR nor size nlocal nor host_ptr
    ocl->mem_positions  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * 3 * sizeof(fcs_float), NULL, &_err));
    ocl->mem_charges    = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(fcs_float), NULL, &_err));
    ocl->mem_indices    = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(fcs_gridsort_index_t), NULL, &_err));
    if(field != NULL)
      ocl->mem_field      = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * 3 * sizeof(fcs_float), NULL, &_err));
    if(potentials != NULL)
      ocl->mem_potentials = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(fcs_float), NULL, &_err));
  }

  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-bitonic: writing\n"););

  T_START(11, "write_buffers");
  CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_boxes, CL_FALSE, offset * sizeof(box_t), nlocal * sizeof(box_t), boxes, 0, NULL, NULL));
  // write zeros to fill up buffer for bitonic
  const int zero = 0;
  CL_CHECK(clEnqueueFillBuffer(ocl->command_queue, ocl->mem_boxes, &zero, sizeof(zero), 0, offset * sizeof(box_t), 0, NULL, NULL));

  if(!ocl->use_index) {
    // data offsets don't need to be filled with zeros
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_positions,  CL_FALSE, offset * 3 * sizeof(fcs_float), nlocal * 3 * sizeof(fcs_float), positions, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_charges,    CL_FALSE, offset * sizeof(fcs_float), nlocal * sizeof(fcs_float), charges, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_indices,    CL_FALSE, offset * sizeof(fcs_gridsort_index_t), nlocal * sizeof(fcs_gridsort_index_t), indices, 0, NULL, NULL));
    if(field != NULL)
      CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_field,      CL_FALSE, offset * 3 * sizeof(fcs_float), nlocal * 3 * sizeof(fcs_float), field, 0, NULL, NULL));
    if(potentials != NULL)
      CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_potentials, CL_FALSE, offset * sizeof(fcs_float), nlocal * sizeof(fcs_float), potentials, 0, NULL, NULL));
  }

  // wait for everything to be finished before continuing to work on the data
  CL_CHECK(clFinish(ocl->command_queue));
  T_STOP(11);
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-bitonic: bitonic\n"););

  // set kernel arguments
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 0, sizeof(cl_mem), &ocl->mem_boxes));
  if(ocl->use_index) {
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 3, sizeof(cl_mem), &ocl->mem_data));
  }
  else {
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 3, sizeof(cl_mem), &ocl->mem_positions));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 4, sizeof(cl_mem), &ocl->mem_charges));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 5, sizeof(cl_mem), &ocl->mem_indices));
    if(field != NULL)
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 6, sizeof(cl_mem), &ocl->mem_field));
    else
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 6, sizeof(cl_mem), NULL));
    if(potentials != NULL)
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 7, sizeof(cl_mem), &ocl->mem_potentials));
    else
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 7, sizeof(cl_mem), NULL));
  }

  // basically just run the job now
  size_t global_work_size = n / 2;
  
  // do the bitonic sort thing
  T_START(12, "sort");
  for(unsigned int stage = 1; stage < n; stage *= 2) {
    // set stage param for kernel
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 1, sizeof(int), (void*)&stage));

    for(unsigned int dist = stage; dist > 0; dist /= 2) {
      // set kernel argument for dist
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 2, sizeof(int), (void*)&dist));

      // and finally run the kernel
      CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_bitonic_global_2, 1, NULL, &global_work_size, NULL, 0, NULL, &ocl->sort_kernel_completion));
      CL_CHECK(clWaitForEvents(1, &ocl->sort_kernel_completion));
      CL_CHECK(clFinish(ocl->command_queue));
    }
  }

  // wait for the sort to finish
  CL_CHECK(clWaitForEvents(1, &ocl->sort_kernel_completion));
  CL_CHECK(clReleaseEvent(ocl->sort_kernel_completion));
  CL_CHECK(clFinish(ocl->command_queue));
  T_STOP(12);

  // read back the results
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-bitonic: reading back\n"););
  T_START(13, "read_back");
  CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_boxes, CL_FALSE, offset * sizeof(box_t), nlocal * sizeof(box_t), boxes, 0, NULL, NULL));
  // data
  if(ocl->use_index) {
    // let it finish
    CL_CHECK(clFinish(ocl->command_queue));
    T_STOP(13);
    // release right away
    CL_CHECK(clReleaseMemObject(ocl->mem_boxes));
    // and hand of to helper function
    INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-bitonic: move data\n"););
    T_START(14, "move_data");
    fcs_ocl_sort_move_data(ocl, nlocal, offset, ocl->mem_data, positions, charges, indices, field, potentials);
    T_STOP(14);
    // release data index
    CL_CHECK(clReleaseMemObject(ocl->mem_data));
  }
  else {
    CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_positions, CL_FALSE, offset * 3 * sizeof(fcs_float), nlocal * 3 * sizeof(fcs_float), positions, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_charges, CL_FALSE, offset * sizeof(fcs_float), nlocal * sizeof(fcs_float), charges, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_indices, CL_FALSE, offset * sizeof(fcs_gridsort_index_t), nlocal * sizeof(fcs_gridsort_index_t), indices, 0, NULL, NULL));
    if(field != NULL)
      CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_field, CL_FALSE, offset * 3 * sizeof(fcs_float), nlocal * 3 * sizeof(fcs_float), field, 0, NULL, NULL));
    if(potentials != NULL)
      CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_potentials, CL_FALSE, offset * sizeof(fcs_float), nlocal * sizeof(fcs_float), potentials, 0, NULL, NULL));
    // wait for all reads to be done
    CL_CHECK(clFinish(ocl->command_queue));
    T_STOP(13);


    INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-bitonic: releasing buffers\n"););
    // now destory our objects
    CL_CHECK(clReleaseMemObject(ocl->mem_boxes));
    // data
    CL_CHECK(clReleaseMemObject(ocl->mem_positions));
    CL_CHECK(clReleaseMemObject(ocl->mem_charges));
    CL_CHECK(clReleaseMemObject(ocl->mem_indices));
    if(field != NULL)
      CL_CHECK(clReleaseMemObject(ocl->mem_field));
    if(potentials != NULL)
      CL_CHECK(clReleaseMemObject(ocl->mem_potentials));
  }
}


/*
 * hybrid sort
 */

static void fcs_ocl_sort_hybrid_prepare(fcs_ocl_context_t *ocl) {
  cl_int ret;

  const char* hybrid_use_index = "-D HYBRID_USE_INDEX=1 -D BITONIC_USE_INDEX=1";

  // first combine program
  const char* sources[] = {
    fcs_ocl_cl_config,
    fcs_ocl_cl,
    fcs_ocl_math_cl,
    fcs_ocl_cl_sort_config,
    fcs_ocl_cl_sort,
    fcs_ocl_cl_sort_bitonic,
    fcs_ocl_cl_sort_hybrid
  };

  ocl->sort_program_hybrid = CL_CHECK_ERR(clCreateProgramWithSource(ocl->context, sizeof(sources) / sizeof(sources[0]), sources, NULL, &_err));

  // build the program
  if(ocl->use_index)
    ret = clBuildProgram(ocl->sort_program_hybrid, 1, &ocl->device_id, hybrid_use_index, NULL, NULL);
  else
    ret = clBuildProgram(ocl->sort_program_hybrid, 1, &ocl->device_id, NULL, NULL, NULL);
  if (ret != CL_SUCCESS) {
    // if there are any errors, just print them
    size_t length;
    char buffer[32*1024];
    CL_CHECK(clGetProgramBuildInfo(ocl->sort_program_hybrid, ocl->device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length));
    printf("ocl-hybrid: build fail %d\nocl build info: %.*s\n", ret, (int) length, buffer);
    return;
  }

  // finally create the kernel
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-hybrid: creating kernels\n"););
  ocl->sort_kernel_bitonic_local    = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_hybrid, "bitonic_local", &_err));
  ocl->sort_kernel_bitonic_global_2 = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_hybrid, "bitonic_global_2", &_err));

  if(ocl->use_index) {
    // create additional kernels
    ocl->sort_kernel_move_data_float          = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_hybrid, "move_data_float", &_err));
    ocl->sort_kernel_move_data_float_triple   = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_hybrid, "move_data_float_triple", &_err));
    ocl->sort_kernel_move_data_gridsort_index = CL_CHECK_ERR(clCreateKernel(ocl->sort_program_hybrid, "move_data_gridsort_index", &_err));
  }
}

static void fcs_ocl_sort_hybrid_release(fcs_ocl_context_t *ocl) {
  cl_int ret;

  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-hybrid: releasing\n"););
  // destroy our kernel and program
  CL_CHECK(clReleaseKernel(ocl->sort_kernel_bitonic_local));
  CL_CHECK(clReleaseKernel(ocl->sort_kernel_bitonic_global_2));
  CL_CHECK(clReleaseProgram(ocl->sort_program_hybrid));
  if(ocl->use_index) {
    // let the move kernels go
    CL_CHECK(clReleaseKernel(ocl->sort_kernel_move_data_float));
    CL_CHECK(clReleaseKernel(ocl->sort_kernel_move_data_float_triple));
    CL_CHECK(clReleaseKernel(ocl->sort_kernel_move_data_gridsort_index));
  }
}


static void fcs_ocl_sort_hybrid(fcs_ocl_context_t *ocl, fcs_int nlocal, box_t *boxes, fcs_float *positions, fcs_float *charges, fcs_gridsort_index_t *indices, fcs_float *field, fcs_float *potentials)
{
  // bitonic sort only works for n as real power of 2
  int n = fcs_ocl_helper_next_power_of_2(nlocal);
  int offset = n - nlocal;

  // calculate parameters
  size_t local_size_local = FCS_NEAR_OCL_SORT_WORKGROUP_MAX;
  size_t bytesPerElement = sizeof(box_t);
  if(ocl->use_index)
    bytesPerElement += sizeof(data_index_t);
  
  int quota = ocl->local_memory / (local_size_local * bytesPerElement);

  while (n / (2 * quota) < local_size_local && local_size_local > FCS_NEAR_OCL_SORT_WORKGROUP_MIN) {
    // we need to adjust and make smaller groups
    local_size_local /= 2;
  }

  // check if the planned size was not met
  if(quota == 0) {
    quota = 1;
    local_size_local = fcs_ocl_helper_next_power_of_2((ocl->local_memory / bytesPerElement) / 2);
  }

  const int workgroupElementsNum = quota * local_size_local;

  size_t global_size_local = n / quota;
  const size_t global_size_global = n/2;

  INFO_CMD(
    printf(INFO_PRINT_PREFIX "ocl-hybrid: bitonic hybrid (use index: %d) [%" FCS_LMOD_INT "d] => [%"  FCS_LMOD_INT "d]\n", ocl->use_index, nlocal, n);
    printf(INFO_PRINT_PREFIX "ocl-hybrid: %ld groups, %d elements each (quota %d)\n", global_size_local / local_size_local, workgroupElementsNum, quota);
    printf(INFO_PRINT_PREFIX "ocl-hybrid: local memory: %ld of %ld bytes\n", bytesPerElement * workgroupElementsNum, ocl->local_memory);
  );

  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-hybrid: initializing buffers\n"););
  // then initialize memory and write to it
  T_START(11, "write_buffers");
  ocl->mem_boxes = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(box_t), NULL, &_err));
  // data all read/write for swapping
  if(ocl->use_index) {
    ocl->mem_data = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(data_index_t), NULL, &_err));
  }
  else {
    // we offset this too, can't use CL_MEM_USE_HOST_PTR nor size nlocal nor host_ptr
    ocl->mem_positions  = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * 3 * sizeof(fcs_float), NULL, &_err));
    ocl->mem_charges    = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(fcs_float), NULL, &_err));
    ocl->mem_indices    = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(fcs_gridsort_index_t), NULL, &_err));
    if(field != NULL)
      ocl->mem_field      = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * 3 * sizeof(fcs_float), NULL, &_err));
    if(potentials != NULL)
      ocl->mem_potentials = CL_CHECK_ERR(clCreateBuffer(ocl->context, CL_MEM_READ_WRITE, n * sizeof(fcs_float), NULL, &_err));
  }

  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-hybrid: writing\n"););

  CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_boxes, CL_FALSE, offset * sizeof(box_t), nlocal * sizeof(box_t), boxes, 0, NULL, NULL));
  // write zeros to fill up buffer
  const int zero = 0;
  CL_CHECK(clEnqueueFillBuffer(ocl->command_queue, ocl->mem_boxes, &zero, sizeof(zero), 0, offset * sizeof(box_t), 0, NULL, NULL));

  if(!ocl->use_index) {
    // data offsets don't need to be filled with zeros
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_positions,  CL_FALSE, offset * 3 * sizeof(fcs_float), nlocal * 3 * sizeof(fcs_float), positions, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_charges,    CL_FALSE, offset * sizeof(fcs_float), nlocal * sizeof(fcs_float), charges, 0, NULL, NULL));
    CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_indices,    CL_FALSE, offset * sizeof(fcs_gridsort_index_t), nlocal * sizeof(fcs_gridsort_index_t), indices, 0, NULL, NULL));
    if(field != NULL)
      CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_field,      CL_FALSE, offset * 3 * sizeof(fcs_float), nlocal * 3 * sizeof(fcs_float), field, 0, NULL, NULL));
    if(potentials != NULL)
      CL_CHECK(clEnqueueWriteBuffer(ocl->command_queue, ocl->mem_potentials, CL_FALSE, offset * sizeof(fcs_float), nlocal * sizeof(fcs_float), potentials, 0, NULL, NULL));
  }

  // wait for everything to be finished before continuing to work on the data
  CL_CHECK(clFinish(ocl->command_queue));
  T_STOP(11);

  // set kernel arguments for bitonic local
  int stage = 1;
  int sortOnGlobal = 1; // true
  int sortOnGlobalFactor = 1;
  int desc = 0;
  int startStage = 1;
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 0, sizeof(cl_mem), &ocl->mem_boxes));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 1, workgroupElementsNum * sizeof(box_t), NULL));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 2, sizeof(int), (void*)&quota));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 3, sizeof(int), (void*)&desc));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 4, sizeof(int), (void*)&stage));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 5, sizeof(int), (void*)&sortOnGlobal));
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 6, sizeof(int), (void*)&sortOnGlobalFactor));
   if(ocl->use_index) {
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 7, sizeof(cl_mem), &ocl->mem_data));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 8, workgroupElementsNum * sizeof(int), NULL));
  }
  else {
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 7, sizeof(cl_mem), &ocl->mem_positions));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 8, sizeof(cl_mem), &ocl->mem_charges));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 9, sizeof(cl_mem), &ocl->mem_indices));
    if(field != NULL)
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 10, sizeof(cl_mem), &ocl->mem_field));
    else
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 10, sizeof(cl_mem), NULL));
    if(potentials != NULL)
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 11, sizeof(cl_mem), &ocl->mem_potentials));
    else
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 11, sizeof(cl_mem), NULL));
  }

  // set kernel arguments for bitonic global
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 0, sizeof(cl_mem), &ocl->mem_boxes));
  if(ocl->use_index) {
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 3, sizeof(cl_mem), &ocl->mem_data));
  }
  else {
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 3, sizeof(cl_mem), &ocl->mem_positions));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 4, sizeof(cl_mem), &ocl->mem_charges));
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 5, sizeof(cl_mem), &ocl->mem_indices));
    if(field != NULL)
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 6, sizeof(cl_mem), &ocl->mem_field));
    else
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 6, sizeof(cl_mem), NULL));
    if(potentials != NULL)
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 7, sizeof(cl_mem), &ocl->mem_potentials));
    else
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 7, sizeof(cl_mem), NULL));
  }

  // run the first set! (local kernel)
  CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_bitonic_local, 1, NULL, &global_size_local, &local_size_local, 0, NULL, &ocl->sort_kernel_completion));

  // set argument for local kernel, will always start in the stage of minDist
  int minDist = workgroupElementsNum / 2;
  CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 4, sizeof(int), (void*)&minDist));

  // let everything finish first
  CL_CHECK(clFinish(ocl->command_queue));

  // main loop of bitonic hybrid sort
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-hybrid: start bitonic hybrid\n"););
  T_START(12, "sort");
  for(stage = workgroupElementsNum; stage < n; stage *= 2) {
    // set stage argument for global kernel
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 1, sizeof(int), (void*)&stage));

    for(unsigned int dist = stage; dist > minDist; dist /= 2) {
      // set kernel argument for dist
      CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_global_2, 2, sizeof(int), (void*)&dist));

      // and finally run the kernel
      CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_bitonic_global_2, 1, NULL, &global_size_global, NULL, 0, NULL, &ocl->sort_kernel_completion));
      CL_CHECK(clWaitForEvents(1, &ocl->sort_kernel_completion));
      CL_CHECK(clFinish(ocl->command_queue));
    }

    // now it's small enough to use the local kernel
    // increse sort on global factor to simulate a sorting of a higher stage
    sortOnGlobalFactor *= 2;
    CL_CHECK(clSetKernelArg(ocl->sort_kernel_bitonic_local, 6, sizeof(int), (void*)&sortOnGlobalFactor));
    // run the local kernel
    CL_CHECK(clEnqueueNDRangeKernel(ocl->command_queue, ocl->sort_kernel_bitonic_local, 1, NULL, &global_size_local, &local_size_local, 0, NULL, &ocl->sort_kernel_completion));
    CL_CHECK(clWaitForEvents(1, &ocl->sort_kernel_completion));
    CL_CHECK(clFinish(ocl->command_queue));
  }
  T_STOP(12);

  // the magic is already done

  CL_CHECK(clReleaseEvent(ocl->sort_kernel_completion));

  // read back the results
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-hybrid: reading back\n"););
  T_START(13, "read_back");
  CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_boxes, CL_FALSE, offset * sizeof(box_t), nlocal * sizeof(box_t), boxes, 0, NULL, NULL));
  // data
  if(ocl->use_index) {
    // let it finish
    CL_CHECK(clFinish(ocl->command_queue));
    T_STOP(13);
    // release right away
    CL_CHECK(clReleaseMemObject(ocl->mem_boxes));
    // and hand of to helper function
    INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-hybrid: move data\n"););
    T_START(14, "move_data");
    fcs_ocl_sort_move_data(ocl, nlocal, offset, ocl->mem_data, positions, charges, indices, field, potentials);
    T_STOP(14);
    // release remaining
    CL_CHECK(clReleaseMemObject(ocl->mem_data));
  }
  else {
    CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_positions, CL_FALSE, offset * 3 * sizeof(fcs_float), nlocal * 3 * sizeof(fcs_float), positions, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_charges, CL_FALSE, offset * sizeof(fcs_float), nlocal * sizeof(fcs_float), charges, 0, NULL, NULL));
    CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_indices, CL_FALSE, offset * sizeof(fcs_gridsort_index_t), nlocal * sizeof(fcs_gridsort_index_t), indices, 0, NULL, NULL));
    if(field != NULL)
      CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_field, CL_FALSE, offset * 3 * sizeof(fcs_float), nlocal * 3 * sizeof(fcs_float), field, 0, NULL, NULL));
    if(potentials != NULL)
      CL_CHECK(clEnqueueReadBuffer(ocl->command_queue, ocl->mem_potentials, CL_FALSE, offset * sizeof(fcs_float), nlocal * sizeof(fcs_float), potentials, 0, NULL, NULL));
    // wait for all reads to be done
    CL_CHECK(clFinish(ocl->command_queue));
    T_STOP(13);


    INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-hybrid: releasing buffers\n"););
    // now destory our objects
    CL_CHECK(clReleaseMemObject(ocl->mem_boxes));
    // data
    CL_CHECK(clReleaseMemObject(ocl->mem_positions));
    CL_CHECK(clReleaseMemObject(ocl->mem_charges));
    CL_CHECK(clReleaseMemObject(ocl->mem_indices));
    if(field != NULL)
      CL_CHECK(clReleaseMemObject(ocl->mem_field));
    if(potentials != NULL)
      CL_CHECK(clReleaseMemObject(ocl->mem_potentials));
  }
}


/*
 * entry functions into ocl sort
 */

void fcs_ocl_sort(fcs_near_t* near) {
  fcs_ocl_context_t* ocl = &near->context->ocl;
  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-sort: start\n"););

#ifdef DO_TIMING
  for(int i = 0; i < sizeof(ocl->timing) / sizeof(ocl->timing[0]); i++) {
    ocl->timing[i] = -1.f;
    ocl->timing_ghost[i] = -1.f;
    ocl->timing_names[i] = NULL;
  }
  ocl->_timing = ocl->timing;
#endif // DO_TIMING

  T_START(0, "sum");
   // set usage of index to 1
  near->context->ocl.use_index = 1;

  T_START(1, "sum_prepare");
  switch(near->near_param.ocl_sort_algo)
  {
    case FCS_NEAR_OCL_SORT_ALGO_RADIX:
      near->context->ocl.use_index = 0;
      fcs_ocl_sort_radix_prepare(ocl);
      break;
    case FCS_NEAR_OCL_SORT_ALGO_BITONIC:
      near->context->ocl.use_index = 0;
    case FCS_NEAR_OCL_SORT_ALGO_BITONIC_INDEX:
      fcs_ocl_sort_bitonic_prepare(ocl);
      break;
    case FCS_NEAR_OCL_SORT_ALGO_HYBRID:
      near->context->ocl.use_index = 0;
    case FCS_NEAR_OCL_SORT_ALGO_HYBRID_INDEX:
      fcs_ocl_sort_hybrid_prepare(ocl);
      break;
    default:
      printf("ocl_sort_algo = %d is unknown!\n", near->near_param.ocl_sort_algo);
      abort();
  }
  T_STOP(1);

  T_START(2, "sum_sort");
  switch(near->near_param.ocl_sort_algo)
  {
    case FCS_NEAR_OCL_SORT_ALGO_RADIX:
      fcs_ocl_sort_radix(ocl, near->nparticles, near->context->real_boxes, near->positions, near->charges, near->indices, near->field, near->potentials);
      break;
    case FCS_NEAR_OCL_SORT_ALGO_BITONIC:
    case FCS_NEAR_OCL_SORT_ALGO_BITONIC_INDEX:
      fcs_ocl_sort_bitonic(ocl, near->nparticles, near->context->real_boxes, near->positions, near->charges, near->indices, near->field, near->potentials);
      break;
    case FCS_NEAR_OCL_SORT_ALGO_HYBRID:
    case FCS_NEAR_OCL_SORT_ALGO_HYBRID_INDEX:
      fcs_ocl_sort_hybrid(ocl, near->nparticles, near->context->real_boxes, near->positions, near->charges, near->indices, near->field, near->potentials);
      break;
  }
  T_STOP(2);
#ifdef DO_CHECK
  fcs_ocl_sort_check(near->nparticles, near->context->real_boxes);
#endif

  // check for ghost boxed
  if(near->context->ghost_boxes) {
    ocl->_timing = ocl->timing_ghost;
    T_START(3, "sum_sort");
    switch(near->near_param.ocl_sort_algo)
    {
      case FCS_NEAR_OCL_SORT_ALGO_RADIX:
        fcs_ocl_sort_radix(ocl, near->nghosts, near->context->ghost_boxes, near->ghost_positions, near->ghost_charges, near->ghost_indices, NULL, NULL);
        break;
      case FCS_NEAR_OCL_SORT_ALGO_BITONIC:
      case FCS_NEAR_OCL_SORT_ALGO_BITONIC_INDEX:
        fcs_ocl_sort_bitonic(ocl, near->nghosts, near->context->ghost_boxes, near->ghost_positions, near->ghost_charges, near->ghost_indices, NULL, NULL);
        break;
      case FCS_NEAR_OCL_SORT_ALGO_HYBRID:
      case FCS_NEAR_OCL_SORT_ALGO_HYBRID_INDEX:
        fcs_ocl_sort_hybrid(ocl, near->nghosts, near->context->ghost_boxes, near->ghost_positions, near->ghost_charges, near->ghost_indices, NULL, NULL);
        break;
    }
    T_STOP(3);
    ocl->_timing = ocl->timing;
#ifdef DO_CHECK
    fcs_ocl_sort_check(near->nghosts, near->context->ghost_boxes);
#endif
  }

  T_START(4, "sum_release");
  switch(near->near_param.ocl_sort_algo)
  {
    case FCS_NEAR_OCL_SORT_ALGO_RADIX:
      fcs_ocl_sort_radix_release(ocl);
      break;
    case FCS_NEAR_OCL_SORT_ALGO_BITONIC:
    case FCS_NEAR_OCL_SORT_ALGO_BITONIC_INDEX:
      fcs_ocl_sort_bitonic_release(ocl);
      break;
    case FCS_NEAR_OCL_SORT_ALGO_HYBRID:
    case FCS_NEAR_OCL_SORT_ALGO_HYBRID_INDEX:
      fcs_ocl_sort_hybrid_release(ocl);
      break;
  }
  T_STOP(4);

  // done with everything
  T_STOP(0);

  INFO_CMD(printf(INFO_PRINT_PREFIX "ocl-sort: end\n"););

#ifdef DO_TIMING
  for(int i = 0; i < sizeof(ocl->timing) / sizeof(ocl->timing[0]); i++) {
    if(ocl->timing[i] >= 0.f)
      printf("ocl-sort-timing: %s %f\n", ocl->timing_names[i], ocl->timing[i]);
  }
  for(int i = 0; i < sizeof(ocl->timing_ghost) / sizeof(ocl->timing_ghost[0]); i++) {
    if(ocl->timing_ghost[i] >= 0.f)
      printf("ocl-sort-timing-ghost: %s %f\n", ocl->timing_names[i], ocl->timing_ghost[i]);
  }
#endif // DO_TIMING
}


#endif // FCS_NEAR_OCL_SORT
