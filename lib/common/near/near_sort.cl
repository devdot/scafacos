

typedef void HERE_COMES_THE_CODE;

// OpenCL long is equal to C99 long long
typedef long fcs_gridsort_index_t;

#define NULL 0

static void inline ocl_sort_swap_float_triple(int i, int j, __global fcs_float* array) {
    i = i * 3;
    j = j * 3;
    fcs_float tmp0 = array[i];
    fcs_float tmp1 = array[i + 1];
    fcs_float tmp2 = array[i + 2];

    array[i] = array[j];
    array[i + 1] = array[j + 1];
    array[i + 2] = array[j + 2];

    array[j] = tmp0;
    array[i + 1] = tmp1;
    array[i + 2] = tmp2;
}

static void inline ocl_sort_swap_float(int i, int j, __global fcs_float* array) {
    fcs_float tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
}

static void inline ocl_sort_swap_gridsort_index(int i, int j, __global fcs_gridsort_index_t* array) {
    fcs_gridsort_index_t tmp = array[i];
    array[i] = array[j];
    array[j] = tmp;
}

// this kernel will deal with 2 elements,
//   it has to be called n/2 times in parallel for n elements
__kernel void bitonic_global_2(__global long* key, const int stage, const int dist, const int offset,
    __global fcs_float* positions,
    __global fcs_float* charges, 
    __global fcs_gridsort_index_t* indices, 
    __global fcs_float* field, 
    __global fcs_float* potentials)
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
    long keyA = key[i];
    long keyB = key[j];

    // calculate swap and check
    bool swap = (keyA > keyB) ^ desc;
    if(swap) {
        // now swap the keys around using XOR
        keyA = keyA ^ keyB;
        keyB = keyA ^ keyB;
        keyA = keyA ^ keyB;

        // and save keys to global memory
        key[i] = keyA;
        key[j] = keyB;

        // now swap the data arrays

        ocl_sort_swap_float_triple(i, j, positions);
        ocl_sort_swap_float(i, j, charges);
        ocl_sort_swap_gridsort_index(i, j, indices);
        if(field != NULL)
            ocl_sort_swap_float_triple(i, j, field);
        if(potentials != NULL)
            ocl_sort_swap_float(i, j, potentials);
    }
}