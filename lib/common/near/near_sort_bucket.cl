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
__kernel void bucket_index_samples(__global const key_t* keysGlobal, __global const key_t* samples, __global int* offsets, __global int* sizes, const int groupSize
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

// TODO: use proper scan!
// one thread per column
__kernel void bucket_prefix_columns(__global int* matrix, const int rows) {
	// each thread will do one column
	int index = get_global_id(0);
	int cols = get_global_size(0);

	int last = matrix[index];

	for(int row = 1; row < rows; row++) {
		index += cols;

		last += matrix[index];
		matrix[index] = last;
	}
}

// one thread per column
__kernel void bucket_prefix_final(__global const int* matrix,
	__local int* row,
	__global unsigned int* bucketPos,
	__global unsigned int* bucketContainers,
	__global unsigned int* bucketOffsets,
	const int rows)
{
	int i = get_local_id(0);
	int cols = get_local_size(0);

	// offset the matrix to it's last row
	matrix += (rows - 1) * cols;

	// load data
	row[i] = matrix[i];
	barrier(CLK_LOCAL_MEM_FENCE);

	// we got the sizes of each bucket in row

	// find the next bigger power of 2 for container size
	int container = 1;
	while(container < row[i])
		container <<= 1;

	// fix 0 size buckets (should not happen too much though because that kills performance)
	if(row[i] == 0)
		container = 0;

	bucketContainers[i] = container;

	// and calculate the offset within the container
	bucketOffsets[i] = container - row[i];

	// now sum up the positions
	int j;
	int temp;
	for(int dist = 1; dist < cols; dist <<= 1) {
		j = i + dist;
		
		// make sure we don't read and write at the same time
		temp = row[i];
		barrier(CLK_LOCAL_MEM_FENCE);

		if(j < cols)
			row[j] += temp;

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	// save back to array
	bucketPos[i] = row[i];
}

// one thread per row (and bucket)
// global size is number of rows (is called bucket-wise)
__kernel void bucket_relocate(__global key_t* keys,
    __global index_t* index,
    __global key_t* bucket_keys,
    __global index_t* bucket_index,
    const int bucketNum,
    const int cols,
    const int elementsPerRow,
    __global const int* bucketInnerOffsets,
    __global const int* matrixOffsets,
    __global const int* matrixPrefix)
{
	int rows = get_global_size(0);
	int row = get_global_id(0);
	
	// offset the matrix
	__global const int* rowOffsets = matrixOffsets + row * cols;

	// calculate the offset in global
	// offset of the row in global arrays
	int globalOffset = row * elementsPerRow;
	// offset of the bucket within the row
	globalOffset += rowOffsets[bucketNum];

	// now calculate the size of this partition
	int size = ((bucketNum < cols -1) ? rowOffsets[bucketNum + 1] : elementsPerRow) - rowOffsets[bucketNum];

	// and calculate the offset within the bucket
	int bucketOffset = bucketInnerOffsets[bucketNum];
	// the 0th row contains the offset for the first row (because it's the size of the 0th row)
	bucketOffset += (row == 0) ? 0 : matrixPrefix[(row - 1) * cols + bucketNum];

	// now move data
	keys += globalOffset;
    index += globalOffset;
	bucket_keys += bucketOffset;
	bucket_index += bucketOffset;

	for(int i = 0; i < size; i++) {
		bucket_keys[i]  = keys[i];
        bucket_index[i] = index[i];
    }
}
