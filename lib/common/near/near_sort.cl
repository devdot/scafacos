

typedef void HERE_COMES_THE_CODE;


static void inline ocl_sort_swap_float_triple(int i, int j, fcs_float* array) {
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

// this kernel will deal with 2 elements,
//   it has to be called n/2 times in parallel for n elements
__kernel void bitonic_global_2(__global long* key, int stage, int dist) {
    // long in OpenCL is 64 bits, therefore equal to  C99 long long 

    int k = get_global_id(0);

    // calculate the position of our element
    int low = k & (dist - 1);
    int i = (k << 1) - low;
    int j = i + dist;

    // this is not operating on power of two, check for max size
    int max = get_global_size(0);
    if(j >= max || i >= max)
        return; // for now just do a return here, we don't even need to compare anymore.

    // calculate the direction of sort
    bool desc = ((i & (stage << 1)) != 0);

    // load keys
    int keyA = key[i];
    int keyB = key[j];

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
    }
}