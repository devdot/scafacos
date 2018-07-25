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

// bitonic sort kernel for sorting the maximum amount of local storage
// this kernel is supposed to be fit by local memory size, not thread-to-element rate

// IMPORTANT: quota is assumed to be 2 or a power of 2
// quota is: number of sort elements per workitem OR: (number of sorting elements in workgroup) / (number of threads in workgroup) 

typedef void HERE_COMES_THE_CODE;

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
#if USE_INDEX
    __global index_t* data
#if !HYBRID_INDEX_GLOBAL
    ,__local index_t* dataBuffer
#endif
#else // USE_INDEX
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
	
	int local_id = get_local_id(0);
	int local_workitems = get_local_size(0);
	int iBase = local_id * quota;
	int iBasePlusQuota = iBase + quota;
	// total number of elements is the amout of threads * elements per threads (quota)
	int len = local_workitems * quota;

#if !HYBRID_PAIRWISE
	// middle section is not done pairwise, only use half quota
	int iBaseHalf = iBase >> 1;
	int iBaseHalfPlusQuota = iBasePlusQuota >> 1;
#endif // HYBRID_PAIRWISE

	// put the global offset onto the global pointers
    int globalOffset = len * get_group_id(0);
	key += globalOffset;
#if USE_INDEX
    data += globalOffset;
#if HYBRID_INDEX_GLOBAL
	__global index_t* dataBuffer = data;
#endif
#else // USE_INDEX
    positions += globalOffset * 3;
    charges += globalOffset;
    indices += globalOffset;
    if(field != NULL)
        field += globalOffset * 3;
    if(potentials != NULL)
        potentials += globalOffset;
#endif // USE_INDEX

	// load global keys into local
	int index;
	for(int i = 0; i < quota; i++) {
#if HYBRID_COALESCE
		index = local_id + local_workitems * i;
#else
		index = i + iBase;
#endif // HYBRID_COALESCE
		elements[index] = key[index];

#if USE_INDEX && !HYBRID_INDEX_GLOBAL
        dataBuffer[index] = data[index];
#endif
    }

	// sync up with other threads before continuing
	barrier(CLK_LOCAL_MEM_FENCE);

	// first of all, use this thread alone for the first stages until this kernel is actuall doing anything synced with the others
	int i, j;
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
#if USE_INDEX
                        swap_data_index(i, j, dataBuffer);
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
#if HYBRID_PAIRWISE
			for(i = iBase; i < iBasePlusQuota; i++) {
				// calculate the partner
				j = i ^ dist;

				// save keys
				key_t iElement = elements[i];
				key_t jElement = elements[j];
#if USE_INDEX
                index_t iData = dataBuffer[i];
                index_t jData = dataBuffer[j];
#endif // USE_INDEX
				// calculate whether the elements should be swapped
				bool swap = (jElement < iElement) ^ (j < i) ^ desc;
#if USE_INDEX
				// we need to make sure not to swap on equal,
				//   this wouldn't matter for key, but data will be corrupted otherwise
				swap = (jElement != iElement) && swap;
#endif // USE_INDEX
				// sync up the threads before and after swap
				barrier(CLK_LOCAL_MEM_FENCE);
				elements[i] = swap?jElement:iElement;
#if USE_INDEX
                dataBuffer[i] = swap?jData:iData;
#endif // USE_INDEX
				barrier(CLK_LOCAL_MEM_FENCE);

#if !USE_INDEX
#ifdef NO_SWAP_ON_EQUAL
				swap = swap && (elements[i] != elements[j]);
#endif // NO_SWAP_ON_EQUAL
                // and move data on global arrays
                if(swap && j > i) {
                    // only the first of both work items works this
                    swap_data_all_global(i, j, positions, charges, indices, field, potentials);
                }
#endif // !USE_INDEX
			}
#else // HYBRID_PAIRWISE
			for(int k = iBaseHalf; k < iBaseHalfPlusQuota; k++) {
				// calculate i and j
				i = (k << 1) - (k & (dist - 1));
				j = i + dist;

				// calculate the swap
				bool swap = (elements[i] > elements[j]) ^ desc;
#ifdef NO_SWAP_ON_EQUAL
				swap = swap && (elements[i] != elements[j]);
#endif // NO_SWAP_ON_EQUAL
				if(swap) {
					swap_keys(elements[i], elements[j]);
					// move data along
#if USE_INDEX
					swap_data_index(i, j, dataBuffer);
#else // USE_INDEX
					swap_data_all_global(i, j, positions, charges, indices, field, potentials);
#endif // USE_INDEX
				}
			}
			// finally sync after the non-pairwise section
			barrier(CLK_LOCAL_MEM_FENCE);
#endif // HYBRID_PAIRWISE
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
#if USE_INDEX
                        swap_data_index(i, j, dataBuffer);
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
#if HYBRID_COALESCE
		index = local_id + local_workitems * i;
#else
		index = i + iBase;
#endif // HYBRID_COALESCE

		key[index] = elements[index];
#if USE_INDEX && !BITONIC_INDEX_GLOBAL
        data[index] = dataBuffer[index];
#endif
    }
}
