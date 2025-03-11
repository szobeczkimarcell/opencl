__kernel void matrix_mult_kernel(__global int* A, __global int* B, __global int* C, int N) {
    int row = get_global_id(0);
    int col = get_global_id(1);
    int sum = 0;
    for (int k = 0; k < N; ++k) {
        sum += A[row * N + k] * B[k * N + col];
    }
    C[row * N + col] = sum;
}