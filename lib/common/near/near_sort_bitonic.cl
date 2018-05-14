// this kernel will deal with 2 elements,
//   it has to be called n/2 times in parallel for n elements

typedef void HERE_COMES_THE_CODE;

#ifndef BITONIC_USE_INDEX
#define BITONIC_USE_INDEX 0
#endif

__kernel void bitonic_global_2(__global key_t* key, const int stage, const int dist,
#if BITONIC_USE_INDEX
    __global index_t* data
#else
    __global fcs_float* positions,
    __global fcs_float* charges, 
    __global fcs_gridsort_index_t* indices, 
    __global fcs_float* field, 
    __global fcs_float* potentials
#endif
    )
{
    // long in OpenCL is 64 bits, therefore equal to  C99 long long 

    int k = get_global_id(0);

    // calculate the position of our element
    int low = k & (dist - 1);
    int i = (k << 1) - low;
    int j = i + dist;

    // calculate the direction of sort
    bool desc = ((i & (stage << 1)) != 0);

    // load keys
    key_t keyA = key[i];
    key_t keyB = key[j];

#if BITONIC_USE_INDEX
    // init data if needed
    if(stage == 1) {
        data[i] = i;
        data[j] = j;
    }
#endif

    // calculate swap and check
    bool swap = (keyA > keyB) ^ desc;
    if(swap) {
        swap_keys(keyA, keyB);

        // and save keys to global memory
        key[i] = keyA;
        key[j] = keyB;

        // now swap the data arrays
#if BITONIC_USE_INDEX
        swap_data_global(i, j, data);
#else
        swap_data_all_global(i, j, positions, charges, indices, field, potentials);
#endif
    }
}
