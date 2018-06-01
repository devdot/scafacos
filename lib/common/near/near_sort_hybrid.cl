// bitonic sort kernel for sorting the maximum amount of local storage
// this kernel is supposed to be fit by local memory size, not thread-to-element rate

// IMPORTANT: quota is assumed to be 2 or a power of 2 (actually works with quota 1)
// quota is: number of sorting elements in workgroup / number of threads in workgroup

typedef void HERE_COMES_THE_CODE;

#ifndef HYBRID_USE_INDEX
#define HYBRID_USE_INDEX 0
#endif

#ifndef HYBRID_INDEX_GLOBAL
#define HYBRID_INDEX_GLOBAL 0
#endif

// int sortDesc, sortOnGlobal are to be treated like a bool
// usually will start in stage 1
// use sortOnGlobalFactor: 1 if every second should be in the other direction. 2 if it should toggle every 2 groups etc.
__kernel void bitonic_local(__global key_t* key,
    __local key_t* elements,
    const int quota,
    int sortDesc,
    int stage,
    const int sortOnGlobal,
    const int sortOnGlobalFactor,
#if HYBRID_USE_INDEX
    __global index_t* data
#if !HYBRID_INDEX_GLOBAL
    ,__local index_t* dataBuffer
#endif
#else // HYBRID_USE_INDEX
    __global fcs_float* positions,
    __global fcs_float* charges, 
    __global fcs_gridsort_index_t* indices, 
    __global fcs_float* field, 
    __global fcs_float* potentials
#endif
    ) 
{
	// check whether we are to sort on a global scope
	if(sortOnGlobal) {
		// every second group will sort the other way
		sortDesc  = sortDesc ^ ((get_group_id(0) & (sortOnGlobalFactor)) != 0);
	}
	
	int iBase = get_local_id(0) * quota;
	// total number of elements is the amout of threads * elements per threads (quota)
	int len = get_local_size(0) * quota; 

	// put the global offset onto the global pointers
    int globalOffset = len * get_group_id(0);
	key += globalOffset;
#if HYBRID_USE_INDEX
    data += globalOffset;
#if HYBRID_INDEX_GLOBAL
	__global index_t* dataBuffer = data;
#endif
#else // HYBRID_USE_INDEX
    positions += globalOffset * 3;
    charges += globalOffset;
    indices += globalOffset;
    if(field != NULL)
        field += globalOffset * 3;
    if(potentials != NULL)
        potentials += globalOffset;
#endif

	// load global keys into local
	for(int i = 0; i < quota; i++) {
		elements[i + iBase] = key[i + iBase];

#if HYBRID_USE_INDEX && !HYBRID_INDEX_GLOBAL
        dataBuffer[i + iBase] = data[i + iBase];
#endif
    }

	// sync up with other threads before continuing
	barrier(CLK_LOCAL_MEM_FENCE);

	// first of all, use this thread alone for the first stages until this kernel is actuall doing anything synced with the others
	int i, j;
	int iBasePlusQuota = iBase + quota;
	for(;stage < quota; stage <<= 1) {
		// go through distances of this stage
		for(int dist = stage; dist > 0; dist >>= 1) {
			// calculate the i's through this loop
			for(i = iBase; i < iBasePlusQuota; i = j + 1) {
				// the loop will jump through the last assignment of i = j + 1
				// all i's in each iteration share the same direction
				bool desc = ((i & (stage << 1)) != 0) ^ sortDesc;
				// go through those who share the same direction and distance
				for(int l = 0; l < dist; l++) {
					// calculate j
					j = i ^ dist; // I guess since we are always approaching from 'top', the xor should be the same as just addition

					// calculate swap if the elements are not in the desired order (using xor)
					bool swap = (elements[i] > elements[j]) ^ desc;
#ifdef NO_SWAP_ON_EQUAL
					swap = swap && (elements[i] != elements[j]);
#endif
					if(swap) {
						swap_keys(elements[i], elements[j]);
                        // move data along
#if HYBRID_USE_INDEX
                        swap_data_global(i, j, dataBuffer);
#else
                        swap_data_all_global(i, j, positions, charges, indices, field, potentials);
#endif
                    }

					// increase i each time
					i++;
				}
			}
		}	
	}
	
	// at this point we got a bitonic sequence of size quota
	// now onto the truly parallel path
	// first, sync up
	barrier(CLK_LOCAL_MEM_FENCE);

	for(; stage < len; stage <<= 1) {
		// all our elements now have the same direction for each stage
		bool desc = ((iBase & (stage << 1)) != 0) ^ sortDesc;

		// we can do pairwise (two threads per comparison) swaps when the distance is not smaller than our quota
		int dist = stage;
		for(; dist >= quota; dist >>= 1) {
			// go through each of our elements
			for(i = iBase; i < iBasePlusQuota; i++) {
				// calculate the partner
				j = i ^ dist; // now we need xor!

				// save keys
				key_t iElement = elements[i];
				key_t jElement = elements[j];
#if HYBRID_USE_INDEX
                index_t iData = dataBuffer[i];
                index_t jData = dataBuffer[j];
#endif
				// calculate whether we have to swap
				bool swap = (jElement < iElement) ^ (j < i) ^ desc;
#if HYBRID_USE_INDEX
				// we need to make sure not to swap on equal,
				//   this wouldn't matter for key, but data will be corrupted otherwise
				swap = (jElement != iElement) && swap;
#endif
				// sync up the memory before and after swap
				barrier(CLK_LOCAL_MEM_FENCE);
				elements[i] = swap?jElement:iElement;
#if HYBRID_USE_INDEX
                dataBuffer[i] = swap?jData:iData;
#endif
				barrier(CLK_LOCAL_MEM_FENCE);

#if !HYBRID_USE_INDEX
#ifdef NO_SWAP_ON_EQUAL
				swap = swap && (elements[i] != elements[j]);
#endif
                // and move data on global arrays
                if(swap && j > i) {
                    // only the first of both work items works this
                    swap_data_all_global(i, j, positions, charges, indices, field, potentials);
                }
#endif
			}
		}

		// now the remaining distances are within our own realm (quota)
		for(; dist > 0; dist >>= 1) {
			// going through all those i's is still the same as in the first part above
			for(i = iBase; i < iBasePlusQuota; i = j + 1) {
				// the loop will jump through the last assignment of i = j + 1
				// all i's in each iteration share the same direction
				bool desc = ((i & (stage << 1)) != 0) ^ sortDesc;
				// go through those who share the same direction and distance
				for(int l = 0; l < dist; l++) {
					// calculate j
					j = i ^ dist; // I guess since we are always approaching from 'top', the xor should be the same as just addition

					// calculate swap if the elements are not in the desired order (using xor)
					bool swap = (elements[i] > elements[j]) ^ desc;
#ifdef NO_SWAP_ON_EQUAL
					swap = swap && (elements[i] != elements[j]);
#endif
					if(swap) {
						swap_keys(elements[i], elements[j]);
                        // and swap data along
#if HYBRID_USE_INDEX
                        swap_data_global(i, j, dataBuffer);
#else
                        swap_data_all_global(i, j, positions, charges, indices, field, potentials);
#endif
                    }

					// increase i each time
					i++;
				}
			}
		}
		// barrier here, because the part above is working single and we need to sync up again
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	
	// don't need a barrier here, we got one after the last write
	// save back the results to global memory
	for(int i = 0; i < quota; i++) {
		key[i + iBase] = elements[i + iBase];
#if HYBRID_USE_INDEX && !BITONIC_INDEX_GLOBAL
        data[i + iBase] = dataBuffer[i + iBase];
#endif
    }
}
