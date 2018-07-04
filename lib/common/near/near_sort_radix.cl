

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
    int offset = global_id * quota;

    // set local histograms to zero
    for(int i = 0; i < RADIX; i++)
        local_histograms[i * local_workitems + local_id] = 0;

    // sync up with all the workitems in our group
    barrier(CLK_LOCAL_MEM_FENCE);

    // compute index 
    int index;
    key_t key, shortkey;

    for(int i = 0; i < quota; i++) {
#if RADIX_TRANSPOSE
        index = workgroups * local_workitems * i + global_id;
#else 
        index = offset + i;
#endif // RADIX_TRANSPOSE

        key = keys[index];

        // extract the current radix (that is used for sorting in this pass)
        //   shift to the position that we want to use and then cut of all bits to the left
        shortkey = ((key >> (pass * RADIX_BITS)) & (RADIX - 1));

        // increment the histogram that is associated with this shortkey
        local_histograms[shortkey * local_workitems + local_id]++;
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
__kernel void radix_reorder(const __global key_t* keysIn, __global key_t* keysOut, const __global index_t* dataIn, __global index_t* dataOut, __global histogram_t* histograms, __local histogram_t* local_histograms, const int pass, const int n) {
    int local_id = get_local_id(0);
    int global_id = get_global_id(0);
    int group_id = get_group_id(0);
    int workgroups = get_num_groups(0);
    int local_workitems = get_local_size(0);

    int quota = n / workgroups / local_workitems;
    int offset = global_id * quota;

    // write histogram to local buffer
    for(int i = 0; i < RADIX; i++)
        local_histograms[local_workitems * i + local_id] = histograms[local_workitems * (i * workgroups + group_id) + local_id];

    barrier(CLK_LOCAL_MEM_FENCE);

    // move items 
    int index;
    int indexOut;
    key_t key, shortkey;
    for(int i = 0; i < quota; i++) {
#if RADIX_TRANSPOSE
        index = workgroups * local_workitems * i + global_id;
#else
        index = i + offset;
#endif // RADIX_TRANSPOSE

        key = keysIn[index];
        
        // get the shortkey again
        shortkey = ((key >> (pass * RADIX_BITS)) & (RADIX - 1));
        
        // calculate the index for the out array
#if RADIX_TRANSPOSE
        int indexOutTmp = local_histograms[shortkey * local_workitems + local_id];
        int t1 = indexOutTmp / quota; // row
        int t2 = indexOutTmp % quota; // col
        indexOut = t2 * (workgroups * local_workitems) + t1;
#else
        indexOut = local_histograms[shortkey * local_workitems + local_id];
#endif // RADIX_TRANSPOSE

        // write
        keysOut[indexOut] = key;
        dataOut[indexOut] = dataIn[index];

        // and increment in the histogram
        local_histograms[shortkey * local_workitems + local_id]++;
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
