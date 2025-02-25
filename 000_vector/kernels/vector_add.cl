__kernel void vector_add_kernel(
    __global int* buffer_a, 
    __global int* buffer_b, 
    __global int* buffer_result, 
    const int n
) {
    int id = get_global_id(0);
    if (id < n) {
        buffer_result[id] = buffer_a[id] + buffer_b[id];
    }
}
