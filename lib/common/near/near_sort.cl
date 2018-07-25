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

typedef void HERE_COMES_THE_CODE;


#define NULL 0

#ifdef cl_nv_pragma_unroll
#define NVIDIA
#endif

// swap keys (long) using XOR
#define swap_keys(a, b) { a = a ^ b; b = a ^ b; a = a ^ b; }

// generic swap
// we have to hand in the data type separatly because typeof will keep the address space type (eg return __global int instead of int)
#define swap_data(a, b, type) { __private type tmp = a; a = b; b = tmp; }
#define swap_data_index(a, b, arr) { swap_data(arr[a], arr[b], index_t); }


#ifndef USE_INDEX
#define USE_INDEX 0
#endif

#if !USE_INDEX

// generic swap on global array
#define swap_data_global(i, j, array, type) { swap_data(array[i], array[j], type); }

static void inline swap_data_float_triple_global(int i, int j, __global fcs_float* array) {
    int k = i * 3;
    int l = j * 3;

    fcs_float tmp0 = array[k];
    fcs_float tmp1 = array[k + 1];
    fcs_float tmp2 = array[k + 2];

    array[k] = array[l];
    array[++k] = array[l + 1];
    array[++k] = array[l + 2];

    array[l] = tmp0;
    array[++l] = tmp1;
    array[++l] = tmp2;
}

static void inline swap_data_all_global(int i, int j,
    __global fcs_float* positions,
    __global fcs_float* charges, 
    __global fcs_gridsort_index_t* indices, 
    __global fcs_float* field, 
    __global fcs_float* potentials)
{
    swap_data_float_triple_global(i, j, positions);
    swap_data_global(i, j, charges, fcs_float);
    swap_data_global(i, j, indices, fcs_gridsort_index_t);
    if(field != NULL)
        swap_data_float_triple_global(i, j, field);
    if(potentials != NULL)
        swap_data_global(i, j, potentials, fcs_float);
}
#endif // !USE_INDEX

#if USE_INDEX

__kernel void init_index(__global index_t* index) {
    index_t i = get_global_id(0);
  
#if !USE_SUBBUFFERS
    // need to subtract offset from value but keep the position so we shift the array
    size_t offset = get_global_offset(0);
    index += offset;
    i -= offset;
#endif // !USE_SUBBUFFERS

    index[i] = i;
}

// one work-item per memory item
__kernel void move_data_float(__global index_t* index, const int offset,
    __global const fcs_float* in,
    __global fcs_float* out)
{
    // shift index array
    index += offset;

    // get our index
    index_t indexOut  = get_global_id(0);
    index_t indexIn = index[indexOut];

    // and now just write in to out
    out[indexOut] = in[indexIn];
}

// one work-item per memory item
__kernel void move_data_float_triple(__global index_t* index, const int offset,
    __global fcs_float* in,
    __global fcs_float* out)
{
    // shift index array
    index += offset;

    // get our index
    index_t indexOut  = get_global_id(0);
    index_t indexIn = index[indexOut];

    // adjust for triple
    indexIn *= 3;
    indexOut *= 3;

    // and now just write in to out
    out[indexOut] = in[indexIn];
    out[++indexOut] = in[++indexIn];
    out[++indexOut] = in[++indexIn];
}

// one work-item per memory item
__kernel void move_data_gridsort_index(__global index_t* index, const int offset,
    __global fcs_gridsort_index_t* in,
    __global fcs_gridsort_index_t* out)
{
    // shift index array
    index += offset;

    // get our index
    index_t indexOut  = get_global_id(0);
    index_t indexIn = index[indexOut];

    // and now just write in to out
    out[indexOut] = in[indexIn];
}

#endif // USE_INDEX
