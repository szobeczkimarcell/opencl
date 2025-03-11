#include "kernel_loader.h"

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define MIN 1
#define MAX 100

int main(void)
{
    // Initialize
    int i;
    cl_int err;
    int error_code;
    int MATRIX_SIZE = 4;

    // Get platform
    cl_uint n_platforms;
    cl_platform_id platform_id;
    err = clGetPlatformIDs(1, &platform_id, &n_platforms);
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetPlatformIDs. Error code: %d\n", err);
        return 0;
    }

    // Get device
    cl_device_id device_id;
    cl_uint n_devices;
    err = clGetDeviceIDs(
        platform_id,
        CL_DEVICE_TYPE_GPU,
        1,
        &device_id,
        &n_devices
    );
    if (err != CL_SUCCESS) {
        printf("[ERROR] Error calling clGetDeviceIDs. Error code: %d\n", err);
        return 0;
    }

    // Create OpenCL context
    cl_context context = clCreateContext(NULL, n_devices, &device_id, NULL, NULL, NULL);

    // Build the program
    const char* kernel_code = load_kernel_source("kernels/matrix_mult.cl", &error_code);
    if (error_code != 0) {
        printf("Source code loading error!\n");
        return 0;
    }
    cl_program program = clCreateProgramWithSource(context, 1, &kernel_code, NULL, NULL);
    const char options[] = "";
    err = clBuildProgram(
        program,
        1,
        &device_id,
        options,
        NULL,
        NULL
    );
    if (err != CL_SUCCESS) {
        printf("Build error! Code: %d\n", err);
        size_t real_size;
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            0,
            NULL,
            &real_size
        );
        char* build_log = (char*)malloc(sizeof(char) * (real_size + 1));
        err = clGetProgramBuildInfo(
            program,
            device_id,
            CL_PROGRAM_BUILD_LOG,
            real_size + 1,
            build_log,
            &real_size
        );
        printf("Build log: %s\n", build_log);
        free(build_log);
        return 0;
    }

    cl_kernel kernel = clCreateKernel(program, "matrix_mult_kernel", NULL);

    // Create the host buffers and initialize them
    int* host_buffer_a = (int*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    int* host_buffer_b = (int*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));
    int* host_buffer_result = (int*)malloc(MATRIX_SIZE * MATRIX_SIZE * sizeof(int));

    for (i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        // Random number between MIN and MAX
        host_buffer_a[i] = MIN + rand() % (MAX - MIN + 1); 
        host_buffer_b[i] = MIN + rand() % (MAX - MIN + 1); 
    }

    for (i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        printf("%d ", host_buffer_a[i]);
        if((i+1) % MATRIX_SIZE == 0){
            printf("\n");
        }
    }

    for (i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        printf("%d ", host_buffer_b[i]);
        if((i+1) % MATRIX_SIZE == 0){
            printf("\n");
        }
    }

    void matrix_mult_kernel(__global int* A, __global int* B, __global int* C, int N) {
        printf("kernel");
        int row = get_global_id(0);
        int col = get_global_id(1);
        int sum = 0;
        for (int k = 0; k < N; ++k) {
            sum += A[row * N + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }

    matrix_mult_kernel((void*)&device_buffer_a,(void*)&device_buffer_b,(void*)&device_buffer_result,(void*)&MATRIX_SIZE);

    for (i = 0; i < MATRIX_SIZE * MATRIX_SIZE; ++i) {
        printf("%d ", host_buffer_result[i]);
        if((i+1) % MATRIX_SIZE == 0){
            printf("\n");
        }
    }

    // Create the device buffers
    cl_mem device_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), NULL, NULL);
    cl_mem device_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), NULL, NULL);
    cl_mem device_buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, MATRIX_SIZE * MATRIX_SIZE * sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_buffer_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_buffer_result);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&MATRIX_SIZE);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(
        context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer_a,
        CL_FALSE,
        0,
        MATRIX_SIZE * MATRIX_SIZE * sizeof(int),
        host_buffer_a,
        0,
        NULL,
        NULL
    );
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer_b,
        CL_FALSE,
        0,
        MATRIX_SIZE * MATRIX_SIZE * sizeof(int),
        host_buffer_b,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 16;
    size_t global_work_size[2] = { MATRIX_SIZE, MATRIX_SIZE };

    // Apply the kernel on the range
    cl_event event;
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        2,
        NULL,
        global_work_size,
        &local_work_size,
        0,
        NULL,
        &event
    );
    clFinish(command_queue);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_buffer_result,
        CL_TRUE,
        0,
        MATRIX_SIZE * MATRIX_SIZE * sizeof(int),
        host_buffer_result,
        0,
        NULL,
        NULL
    );

    free(host_buffer_a);
    free(host_buffer_b);
    free(host_buffer_result);

    // Release Resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);
}