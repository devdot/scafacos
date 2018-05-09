

typedef void HERE_COMES_THE_CODE;


#define NULL 0

// swap keys (long) using XOR
#define swap_keys(a, b) { a = a ^ b; b = a ^ b; a = a ^ b; }

// generic swap
#define swap_data(a, b) { typeof(a) tmp = a; a = b; b = tmp; }

// generic swap on global array
#define swap_data_global(i, j, array) { swap_data(array[i], array[j]); }

static void inline swap_data_float_triple_global(int i, int j, __global fcs_float* array) {
    int k = i * 3;
    int l = j * 3;

    fcs_float tmp0 = array[k];
    fcs_float tmp1 = array[k + 1];
    fcs_float tmp2 = array[k + 2];

    array[k] = array[l];
    array[++k] = array[l + 1];
    array[++k] = array[l + 2];

    array[l] = tmp0;
    array[++l] = tmp1;
    array[++l] = tmp2;
}

static void inline swap_data_all_global(int i, int j,
    __global fcs_float* positions,
    __global fcs_float* charges, 
    __global fcs_gridsort_index_t* indices, 
    __global fcs_float* field, 
    __global fcs_float* potentials)
{
    swap_data_float_triple_global(i, j, positions);
    swap_data_global(i, j, charges);
    swap_data_global(i, j, indices);
    if(field != NULL)
        swap_data_float_triple_global(i, j, field);
    if(potentials != NULL)
        swap_data_global(i, j, potentials);
}
