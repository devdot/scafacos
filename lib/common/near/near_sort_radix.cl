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

#if RADIX_BITS < 8
    typedef char shortkey_t;
#elif RADIX_BITS < 16
    typedef short shortkey_t;
#else
    typedef int shortkey_t;
#endif

// one workitem per quota keys
__kernel void radix_histogram(const __global key_t* keys, __global histogram_t* histograms, __local histogram_t* local_histograms, const int pass, const int n) {
    // get the identifiers
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);

    int workgroups = get_num_groups(0);
    int local_workitems = get_local_size(0);

    // amount of keys that are analyzed by this workitem
    int quota = n / workgroups / local_workitems;

    // set local histograms to zero
    for(int i = 0; i < RADIX; i++)
        local_histograms[i * local_workitems + local_id] = 0;

    // sync up with all the workitems in our group
    barrier(CLK_LOCAL_MEM_FENCE);

    // variables for loop
    key_t key, shortkey;

    // index is calculated based on transposition
#if RADIX_TRANSPOSE
    int index = global_id;
    const int indexIncrement = workgroups * local_workitems;
#else
    int index = global_id * quota;;
    const int indexIncrement = 1;
#endif // RADIX_TRANSPOSE

    for(int i = 0; i < quota; i++) {
        key = keys[index];

        // extract the current radix (that is used for sorting in this pass)
        //   shift to the position that we want to use and then cut of all bits to the left
        shortkey = ((key >> (pass * RADIX_BITS)) & (RADIX - 1));

        // increment the histogram that is associated with this shortkey
        local_histograms[shortkey * local_workitems + local_id]++;
        
        // increment index
        index += indexIncrement;
    }

    // sync up again
    barrier(CLK_LOCAL_MEM_FENCE);

    // copy our histograms from local memory back to global
    for(int i = 0; i < RADIX; i++)
        histograms[local_workitems * (i * workgroups + group_id) + local_id] = local_histograms[i * local_workitems + local_id];
}


// parallel prefix sum
// two memory items per workitem
__kernel void radix_scan(__global histogram_t* histograms, __local histogram_t* buffer, __global histogram_t* sum) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    int n = get_local_size(0) * 2;

    // add offset to global pointer
    histograms += 2 * global_id;

    // load into local buffer
    buffer[2 * local_id] = histograms[0];
    buffer[2 * local_id + 1] = histograms[1];

    // run prefix sum
    int dist = 1;
    for(int d = n >> 1; d > 0; d >>= 1) {
        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id < d) {
            int ai = dist * (2 * local_id + 1) - 1;
            int bi = dist * (2 * local_id + 2) - 1;
            buffer[bi] += buffer[ai];
        }
        dist *= 2;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if(local_id == 0) {
        // now store last element in global sum
        sum[group_id] = buffer[n - 1];
        // set to 0
        buffer[n - 1] = 0;
    }

    // sweep back down
    for(int d = 1; d < n; d <<= 1) {
        dist >>= 1;
        barrier(CLK_LOCAL_MEM_FENCE);

        if(local_id < d) {
            int ai = dist * (2 * local_id + 1) - 1;
            int bi = dist * (2 * local_id + 2) - 1;

            histogram_t tmp = buffer[ai];
            buffer[ai] = buffer[bi];
            buffer[bi] += tmp;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // now push that back to global
    histograms[0] = buffer[2 * local_id];
    histograms[1] = buffer[2 * local_id + 1];
}


__kernel void radix_scan_paste(__global histogram_t* histograms, __global histogram_t* sum) {
    histogram_t s = sum[get_group_id(0)];

    // offset the global array
    histograms += 2 * get_global_id(0);

    // write to histograms
    histograms[0] += s;
    histograms[1] += s;
}


// handles quota amount of  keys
__kernel void radix_reorder(const __global key_t* keysIn, __global key_t* keysOut, __global histogram_t* histograms, __local histogram_t* local_histograms, const int pass, const int n,
#if USE_INDEX
    const __global index_t* dataIn,
    __global index_t* dataOut
#else // USE_INDEX
    __global fcs_float* positionsIn,
    __global fcs_float* positionsOut,
    __global fcs_float* chargesIn,
    __global fcs_float* chargesOut, 
    __global fcs_gridsort_index_t* indicesIn,
    __global fcs_gridsort_index_t* indicesOut,  
    __global fcs_float* fieldIn,
    __global fcs_float* fieldOut,  
    __global fcs_float* potentialsIn,
    __global fcs_float* potentialsOut
#endif // USE_INDEX
) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    int workgroups = get_num_groups(0);
    int local_workitems = get_local_size(0);

    int quota = n / workgroups / local_workitems;

    // write histogram to local buffer
    for(int i = 0; i < RADIX; i++)
        local_histograms[local_workitems * i + local_id] = histograms[local_workitems * (i * workgroups + group_id) + local_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    // pre calculate the incrementing variables
#if RADIX_TRANSPOSE
    int index = global_id;
    const int indexIncrement = workgroups * local_workitems;

    // also quota exponent e (q = 2^e)
    // get the exponent by finding the first bit that's a 1
    int quotaExponent = 0;
    for(int q = quota; (q & 1) == 0; q >>= 1)
        quotaExponent++;
#else
    int index = global_id * quota;
    const int indexIncrement = 1;
#endif // RADIX_TRANSPOSE

    // move items 
    int indexOut;
    key_t key, shortkey;
    for(int i = 0; i < quota; i++) {
        key = keysIn[index];
        
        // get the shortkey again
        shortkey = ((key >> (pass * RADIX_BITS)) & (RADIX - 1));
        
        // calculate the index for the out array
#if RADIX_TRANSPOSE
        int indexOutTmp = local_histograms[shortkey * local_workitems + local_id];
        int t1 = indexOutTmp >> quotaExponent; // row (is equal to (indexOutTmp / quota)
        int t2 = indexOutTmp & (quota - 1); // col (is equal to (indexOutTmp % quota))
        indexOut = t2 * (indexIncrement) + t1;
#else
        indexOut = local_histograms[shortkey * local_workitems + local_id];
#endif // RADIX_TRANSPOSE

        // write
        keysOut[indexOut] = key;
        // depending on swap along or index
#if USE_INDEX
        dataOut[indexOut] = dataIn[index];
#else // USE_INDEX
        int tripleIn = index * 3;
        int tripleOut = indexOut * 3;

        positionsOut[tripleOut]     = positionsIn[tripleIn];
        positionsOut[tripleOut+1]   = positionsIn[tripleIn+1];
        positionsOut[tripleOut+2]   = positionsIn[tripleIn+2];
        chargesOut[indexOut]        = chargesIn[index];
        indicesOut[indexOut]        = indicesIn[index];
        if(fieldIn != NULL) {
            fieldOut[tripleOut]     = fieldIn[tripleIn];
            fieldOut[tripleOut+1]   = fieldIn[tripleIn+1];
            fieldOut[tripleOut+2]   = fieldIn[tripleIn+2];
        }
        if(potentialsIn != NULL)
            potentialsOut[indexOut] = potentialsIn[index];
#endif // USE_INDEX

        // and increment in the histogram
        local_histograms[shortkey * local_workitems + local_id]++;

        // increment the index
        index += indexIncrement;
    }
}

#if RADIX_TRANSPOSE
// works on 2D workitems!
// tilesize is "quota" for this kernel
__kernel void radix_transpose(const __global key_t* keysIn,
    __global key_t* keysOut,
    const __global index_t* dataIn,
    __global index_t* dataOut,
    __local key_t* keysBuffer,
    __local index_t* dataBuffer,
    const int cols,
    const int rows,
    const int tilesize
    )
{
    size_t global_id0 = get_global_id(0); 
    size_t global_id1 = get_global_id(1); // column
    size_t local_id1 = get_local_id(1); // local column
    size_t group_id1 = get_group_id(1);

    int rowIn = global_id0 * tilesize; // first row
    int rowOut = group_id1 * tilesize; // first row transposed

    // fill cache
    int pos, posBuffer;
    for(int i = 0; i < tilesize; i++) {
        pos = (rowIn + i) * cols + global_id1;
        posBuffer = (i * tilesize) + local_id1;

        keysBuffer[posBuffer] = keysIn[pos];
        dataBuffer[posBuffer] = dataIn[pos];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // first row in transpose
    // move cache to out array
    for(int i = 0; i < tilesize; i++) {
        pos = (rowOut + i) * rows + rowIn + local_id1;
        posBuffer = (local_id1 * tilesize) + i;

        keysOut[pos] = keysBuffer[posBuffer];
        dataOut[pos] = dataBuffer[posBuffer];
    }
}
#endif // RADIX_TRANSPOSE
