// global kernels for bitonic sort
// kernels 8, 16 and 32 build on kernel 4 using macros.
// if macros were easier for OpenCL (multiline isn't working correctly),
//   they should be condensed into macros completely

typedef void HERE_COMES_THE_CODE;

// this kernel will deal with 2 elements,
//   it has to be called n/2 times in parallel for n elements
__kernel void bitonic_global_2(__global key_t* key, const int stage, const int dist,
#if USE_INDEX
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

    // calculate swap and check
    bool swap = (keyA > keyB) ^ desc;
#ifdef NO_SWAP_ON_EQUAL
	swap = swap && (keyA != keyB);
#endif
    if(swap) {
        swap_keys(keyA, keyB);

        // and save keys to global memory
        key[i] = keyA;
        key[j] = keyB;

        // now swap the data arrays
#if USE_INDEX
        swap_data_index(i, j, data);
#else
        swap_data_all_global(i, j, positions, charges, indices, field, potentials);
#endif
    }
}


// check if any higher level is enabled
#if BITONIC_GLOBAL_4 | BITONIC_GLOBAL_8 | BITONIC_GLOBAL_16 | BITONIC_GLOBAL_32
// macros for higher level kernels

#ifdef NO_SWAP_ON_EQUAL
#define CALC_SWAP(keyA, keyB, desc) { ((keyA > keyB) ^ desc) && (keyA != keyB) }
#else
#define CALC_SWAP(keyA, keyB, desc) { (keyA > keyB) ^ desc }
#endif

#if USE_INDEX
#define SWAP(keyA, keyB, i, j, swap) { if(swap) { swap_keys(keyA, keyB); swap_data_index(i, j, data); } }
#else
#define SWAP(keyA, keyB, i, j, swap) { if(swap) { swap_keys(keyA, keyB); swap_data_all_global(i, j, positions, charges, indices, field, potentials); } }
#endif

#define CHECK_AND_SWAP(keyA, keyB, i, j, desc) { bool swap = CALC_SWAP(keyA, keyB, desc); SWAP(keyA, keyB, i, j, swap); }

// recursive check and swap for our high tier kernels

#define CHECK_AND_SWAP_AB(a, b) { CHECK_AND_SWAP(key[a], key[b], i[a], i[b], desc0); }

#define CHECK_AND_SWAP_2(k) { CHECK_AND_SWAP_AB(k, k + 1); }
#define CHECK_AND_SWAP_4(k) { for(int i4 = 0; i4 < 2; i4++)      { CHECK_AND_SWAP_AB(k + i4, k + i4 + 2) }    CHECK_AND_SWAP_2(k)  CHECK_AND_SWAP_2(k + 2) }
#define CHECK_AND_SWAP_8(k) { for(int i8 = 0; i8 < 4; i8++)      { CHECK_AND_SWAP_AB(k + i8, k + i8 + 4) }    CHECK_AND_SWAP_4(k)  CHECK_AND_SWAP_4(k + 4) }
#define CHECK_AND_SWAP_16(k) { for(int i16 = 0; i16 < 8; i16++)  { CHECK_AND_SWAP_AB(k + i16, k + i16 + 8) }  CHECK_AND_SWAP_8(k)  CHECK_AND_SWAP_8(k + 8) }
#define CHECK_AND_SWAP_32(k) { for(int i32 = 0; i32 < 16; i32++) { CHECK_AND_SWAP_AB(k + i32, k + i32 + 16) } CHECK_AND_SWAP_16(k) CHECK_AND_SWAP_16(k + 16) }


#endif // bitonic higher tiers

#if BITONIC_GLOBAL_4
// bitonic kernel for 4 elements each   
__kernel void bitonic_global_4(__global key_t* keys, const int stage, int dist,
#if USE_INDEX
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
    // arrays for our elements
    __private int i[4];
    __private key_t key[4];

    // divide distance by two as we now handle 4 elements
    dist >>= 1;

    int k = get_global_id(0);

    // calculate the position of our element
    int low = k & (dist - 1);
    i[0] = ((k - low) << 2) + low; // insert two zeros (00) at bit position dist
    
    // set remaining indicies
    for(int p = 1; p < 4; p++)
        i[p] = i[p - 1] + dist;
    
    // load keys
    for(int p = 0; p < 4; p++)
        key[p] = keys[i[p]];

    // calculate the direction of sort
    bool desc0 = ((i[0] & (stage << 1)) != 0);
    
    // same as CHECK_AND_SWAP_4(0);
    CHECK_AND_SWAP_AB(0, 2);
    CHECK_AND_SWAP_AB(1, 3);
    CHECK_AND_SWAP_AB(0, 1);
    CHECK_AND_SWAP_AB(2, 3);

    // save all keys back to global
    for(int p = 0; p < 4; p++)
        keys[i[p]] = key[p];
}
#endif // BITONIC_GLOBAL_4

#if BITONIC_GLOBAL_8
// bitonic kernel for 8 elements each   
__kernel void bitonic_global_8(__global key_t* keys, const int stage, int dist,
#if USE_INDEX
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
    // arrays for our elements
    __private int i[8];
    __private key_t key[8];

    dist >>= 2;

    int k = get_global_id(0);

    // calculate the position of our element
    int low = k & (dist - 1);
    i[0] = ((k - low) << 3) + low; // insert three zeros (000) at bit position dist
    
    // set remaining indicies
    for(int p = 1; p < 8; p++)
        i[p] = i[p - 1] + dist;
    
    // load keys
    for(int p = 0; p < 8; p++)
        key[p] = keys[i[p]];

    // calculate the direction of sort
    bool desc0 = ((i[0] & (stage << 1)) != 0);
    
    CHECK_AND_SWAP_8(0);

    // save all keys back to global
    for(int p = 0; p < 8; p++)
        keys[i[p]] = key[p];
}
#endif // BITONIC_GLOBAL_8

#if BITONIC_GLOBAL_16
// bitonic kernel for 16 elements each   
__kernel void bitonic_global_16(__global key_t* keys, const int stage, int dist,
#if USE_INDEX
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
    __private int i[16];
    __private key_t key[16];

    dist >>= 3;

    int k = get_global_id(0);

    // calculate the position of our element
    int low = k & (dist - 1);
    i[0] = ((k - low) << 4) + low; // insert four zeros (0000) at bit position dist
    
    // set remaining indicies
    for(int p = 1; p < 16; p++)
        i[p] = i[p - 1] + dist;
    
    // load keys
    for(int p = 0; p < 16; p++)
        key[p] = keys[i[p]];

    // calculate the direction of sort
    bool desc0 = ((i[0] & (stage << 1)) != 0);

    CHECK_AND_SWAP_16(0);

    // save all keys back to global
    for(int p = 0; p < 16; p++)
        keys[i[p]] = key[p];
}
#endif // BITONIC_GLOBAL_16

#if BITONIC_GLOBAL_32
// bitonic kernel for 32 elements each   
__kernel void bitonic_global_32(__global key_t* keys, const int stage, int dist,
#if USE_INDEX
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
    __private int i[32];
    __private key_t key[32];

    dist >>= 4;

    int k = get_global_id(0);

    // calculate the position of our element
    int low = k & (dist - 1);
    i[0] = ((k - low) << 5) + low; // insert five zeros (00000) at bit position dist
    
    // set remaining indicies
    for(int p = 1; p < 32; p++)
        i[p] = i[p - 1] + dist;
    
    // load keys
    for(int p = 0; p < 32; p++)
        key[p] = keys[i[p]];

    // calculate the direction of sort
    bool desc0 = ((i[0] & (stage << 1)) != 0);

    CHECK_AND_SWAP_32(0);

    // save all keys back to global
    for(int p = 0; p < 32; p++)
        keys[i[p]] = key[p];
}
#endif // BITONIC_GLOBAL_32
