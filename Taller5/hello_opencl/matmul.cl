__kernel void mat_mul_par(__global const int* A,
                      __global const int* B,
                      __global int* C) {

    // Thread identifiers
    const int globalRow = get_global_id(0); // Row ID of C (0..M)
    const int globalCol = get_global_id(1); // Col ID of C (0..N)

    // Compute a single element (loop over K)
    int acc = 0;
    for (int k=0; k<4; k++) {
        acc += A[k*4 + globalRow] * B[globalCol*4 + k];
    }
    // Store the result
    C[globalCol*4 + globalRow] = acc;
}
