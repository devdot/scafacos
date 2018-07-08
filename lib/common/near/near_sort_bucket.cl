/*
 * kernel code for GPU Bucket Sort
 * 
 */

typedef void HERE_COMES_THE_CODE;


// will get one sample from keys into samples, each have dist distance between them
__kernel void bucket_sample(__global const key_t* keys, int dist, __global key_t* samples) {
	size_t global_id = get_global_id(0);
	samples[global_id] = keys[global_id * dist];
}

// kernel for indexing of global samples
// returns the next bigger element when the searched sample was not found
// global size: matrix size = buckets * sort groups
// local size: buckets
__kernel void bucket_index_samples(__global key_t* keysGlobal, __global const key_t* samples, __global int* offsets, __global int* sizes, const int groupSize
#if BUCKET_INDEXER_LOCAL
	, __local key_t* buffer
#endif // BUCKET_INDEXER_LOCAL
) {
	// offset the keys and matrix pointer to our group
	size_t local_id = get_local_id(0);
	size_t local_size = get_local_size(0);
	size_t group_id = get_group_id(0);
	
	int matrixRow = local_size * group_id;
	offsets += matrixRow;
	sizes += matrixRow;


	// load elements into buffer if necessary
	keysGlobal += groupSize * group_id;
#if BUCKET_INDEXER_LOCAL
	__local key_t* keys = buffer;
	int quota = groupSize / local_size;
	// load coalesced
	for(int i = 0; i < quota; i++) {
		int index = local_id + local_size * i;
		buffer[index] = keysGlobal[index];
	}
	// sync all threads now
	barrier(CLK_GLOBAL_MEM_FENCE);
#else // BUCKET_INDEXER_LOCAL
	__global key_t * keys = keysGlobal;
#endif // BUCKET_INDEXER_LOCAL

	key_t sample = samples[local_id];

	// now perform binary search for sample
	int l = 0;
	int r = groupSize;
	key_t element;
	int index;
	int pos = -1;
	// take the left most matching
	while(r - l > 1) {
		// calculate the element index to select
		// (l + r) /2
		index = l + ((r - l) / 2);
		// retrieve the element
		element = keys[index];
		if(element >= sample) {
			// search left
			r = index;
		}
		else {
			// go right
			l = index;
		}
	}
	// exception for finding elements at index 0 when they're the same as the element on index 1
	if(l == 0 && r == 1) {
		index = 0;
		element = keys[index];
		if(element >= sample) {
			r = index;
		}
		// l is alread 0
	}
	pos = r;

	// write our result back
	offsets[local_id] = (pos != -1) ? pos : r+1;

	// now calculate sizes
	barrier(CLK_GLOBAL_MEM_FENCE);
	if(local_id != local_size - 1)
		sizes[local_id] = offsets[local_id + 1] - offsets[local_id];
	else
		// next element would be at position groupSize
		sizes[local_id] = groupSize - offsets[local_id];
}

// one group per column (bucket)
// quota has to be 2 or power of 2 (if it's one, up to half of all workitems won't do any work)
// matrix is [m x b], where m is sort groups (subarrays) and b is buckets
__kernel void bucket_prefix_columns(__global int* matrix, const int rows, unsigned int quota) {
	// each thread will do one column
	size_t group_id = get_group_id(0);
	size_t local_id = get_local_id(0);
	size_t local_id_orig = local_id;
	size_t global_size = get_global_size(0);
	size_t local_size = get_local_size(0);

	int cols = global_size / local_size;

	// calculate offset
	int n = local_size * quota;
	int offset = n - rows;

	// first, shift matrix to our column
	matrix += group_id;

	// now divide quota by two (when possible)
	// this can be done because the scan needs only one thread per two items, no need to simulate the upper half of threads
	if(quota > 1)
		quota >>= 1;

	// do the blelloch scan with quota
	// offset is simulated to be on the left, containing zeros
	// upsweep
	int dist = 1;
	for(int d = n >> 1; d > 0; d >>= 1) {
		// go through the quota amount of blocks
		for(int i = 0; i < quota; i++) {
			if(local_id < d) {
				int ai = (dist * (2 * local_id + 1) - 1) - offset;
				int bi = (dist * (2 * local_id + 2) - 1) - offset;

				// only add when were not in offset, bi is always greater than ai
				if(ai >= 0)
					matrix[bi * cols] += matrix[ai * cols];
			}
			// go to next local id in quota
			local_id += local_size;
		}
		local_id = local_id_orig;
        dist <<= 1;
		barrier(CLK_GLOBAL_MEM_FENCE);
	}

	// no insertion of 0 because we're not doing the orginal prefix sum

	// downsweep
	// working a little different than blelloch
	dist >>= 2;
	for(int d = 2; d < n; d <<= 1) {
		barrier(CLK_GLOBAL_MEM_FENCE);
		// handle quota, now right to left
		local_id += quota * local_size;
		for(int i = quota; i > 0; i--) {
			// decrement right away, start one to high
			local_id -= local_size;
			if(local_id < d - 1) {
				int ai = (dist * 2 * (local_id + 1)) - 1 - offset;
				int bi = ai + dist;

				// check if ai is negative, if so we don't need to do anything
				if(ai >= 0) {
					matrix[bi * cols] += matrix[ai * cols];
				}
			}
		}
		dist >>= 1;
	}
}

// one thread per column
__kernel void bucket_prefix_final(__global const int* matrix,
	__local unsigned int* row,
	__local unsigned int* bufferContainerPrefix,
	__global unsigned int* bucketPos,
	__global unsigned int* bucketContainers,
	__global unsigned int* bucketInnerOffsets,
	__global unsigned int* bucketContainerPrefix,
	const int rows
#if BUCKET_USE_RADIX
	, const int radixLocalSize
#endif // BUCKET_USE_RADIX	
)
{
	int i = get_local_id(0);
	int cols = get_local_size(0);

	// offset the matrix to it's last row
	matrix += (rows - 1) * cols;

	// load data
	row[i] = matrix[i];
	barrier(CLK_LOCAL_MEM_FENCE);

	// we got the sizes of each bucket in row
#if BUCKET_USE_RADIX
	// look for next multiple of current size for radix
	bufferContainerPrefix[i] = row[i];
	int rem = row[i] % radixLocalSize;
	// only add when needed
	if(rem != 0)
		bufferContainerPrefix[i] += radixLocalSize - rem;
#else // BUCKET_USE_RADIX
	// find the next bigger power of 2 for container size
	bufferContainerPrefix[i] = 1;
	while(bufferContainerPrefix[i] < row[i])
		bufferContainerPrefix[i] <<= 1;
#endif //BUCKET_USE_RADIX

	// fix 0 size buckets (should not happen too much though because that kills performance)
	if(row[i] == 0)
		bufferContainerPrefix[i] = 0;

	bucketContainers[i] = bufferContainerPrefix[i];

	// and calculate the offset within the container
	bucketInnerOffsets[i] = bufferContainerPrefix[i] - row[i];

	// now sum up the positions
	int j;
	unsigned int temp, tempContainer;
	// prepare prefix sum of container
	// move all elements one to the left and insert 0 at start
	j = i - 1;
	barrier(CLK_LOCAL_MEM_FENCE);
	tempContainer = j < 0 ? 0 : bufferContainerPrefix[j];
	barrier(CLK_LOCAL_MEM_FENCE);
	bufferContainerPrefix[i] = tempContainer;

	for(int dist = 1; dist < cols; dist <<= 1) {
		j = i + dist;
		
		// make sure we don't read and write at the same time
		temp = row[i];
		tempContainer = bufferContainerPrefix[i];
		barrier(CLK_LOCAL_MEM_FENCE);

		if(j < cols) {
			row[j] += temp;
			bufferContainerPrefix[j] += tempContainer;
		}

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// save back to global arrays
	bucketPos[i] = row[i];
	bucketContainerPrefix[i] = bufferContainerPrefix[i];
}

// quota workitems per row
// one group per sort group (subarray)
__kernel void bucket_relocate(__global key_t* keysIn,
	__global index_t* indexIn,
	__global key_t* keysOut,
	__global index_t* indexOut,
	const int quota,
	const int bucketsNum,
	__local int*  bufferPartitionOffset,
	__global int* matrixPartitionOffset,
	__global int* matrixBucketOffset,
	__global int* bucketContainerInnerOffsets,
	__global int* bucketContainerOffsets
) {
	size_t local_id = get_local_id(0);
	size_t local_size = get_local_size(0);
	// group id is number of sort group, row in matrices
	size_t group_id = get_group_id(0);

	// offset input array to group
	int globalOffset = local_size * quota * group_id;
	keysIn += globalOffset;
	indexIn += globalOffset;

	// offset matrices
	matrixPartitionOffset += group_id * bucketsNum;
	matrixBucketOffset += (group_id - 1) * bucketsNum;

	// load partition prefix into local memory
	int i = local_id;
	while(i < bucketsNum) {
		bufferPartitionOffset[i] = matrixPartitionOffset[i];
		i += local_size;
	}
	barrier(CLK_LOCAL_MEM_FENCE);

	if(group_id == 0 && local_id == 0) {
		for(int i = 0; i < bucketsNum; i++) {
			//printf("%d: %d\n", i, bufferPartitionOffset[i]);
		}
	}

	// do this for each quota elements
	unsigned int bucket;
	int bucketOffset;
	for(i = 0; i < quota; i++) {
		// figure out which bucket we belong to
		bucket = 1;
		for(; bucket < bucketsNum; bucket++) {
			if(local_id < bufferPartitionOffset[bucket]) {
				break;
			}
		}
		bucket--;

#if BUCKET_SKIP_FIRST
		// only skip the first bucket when it's really set as empty
		if(!(bucket == 0 && bucketContainerOffsets[1] == 0)) {
#endif // BUCKET_SKIP_FIRST


		// calculate the offset of this element in the bucket arrays
		bucketOffset = bucketContainerInnerOffsets[bucket] + bucketContainerOffsets[bucket]; // TODO calc in local
		bucketOffset += local_id - bufferPartitionOffset[bucket];
		// rows in this matrix are offset by one
		bucketOffset += (group_id == 0) ? 0 : matrixBucketOffset[bucket];

		// now we got both offsets, just copy
		keysOut[bucketOffset] = keysIn[local_id];
		indexOut[bucketOffset] = indexIn[local_id];
#if BUCKET_SKIP_FIRST
		}
#endif // BUCKET_SKIP_FIRST

		// increase local_id
		local_id += local_size;
	}
}
