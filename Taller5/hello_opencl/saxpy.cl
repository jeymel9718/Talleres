__kernel void saxpy_par(__global const int *A, __global const int *B, __global int *C) {

    // Get the index of the current element to be processed
    int i = get_global_id(0);

    // Do the operation
    C[i] = 50*A[i] + B[i];
}
