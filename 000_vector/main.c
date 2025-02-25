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
//Initialize
    int i;
    cl_int err;
    int error_code;
    int VECTOR_SIZE = 1;

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
    const char* kernel_code = load_kernel_source("kernels/vector_add.cl", &error_code);
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
//Initialize

    cl_kernel kernel = clCreateKernel(program, "vector_add_kernel", NULL);



    // File for data
    FILE *file = fopen("output.txt", "w");
    if (file == NULL) {
        // Error handling if the file couldn't be opened
        printf("Error opening file!\n");
        return 1;
    }

for (int j = 0; j<5000; ++j){
    VECTOR_SIZE *= 2;
    srand(time(NULL));

    // Create the host buffers and initialize them
    int* host_buffer_a = (int*)malloc(VECTOR_SIZE * sizeof(int));
    int* host_buffer_b = (int*)malloc(VECTOR_SIZE * sizeof(int));
    int* host_buffer_result = (int*)malloc(VECTOR_SIZE * sizeof(int));

    for (i = 0; i < VECTOR_SIZE; ++i) {
        // Random number between MIN and MAX
        host_buffer_a[i] = MIN + rand() % (MAX - MIN + 1); 
        host_buffer_b[i] = MIN + rand() % (MAX - MIN + 1); 
    }

    // Create the device buffers
    cl_mem device_buffer_a = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(int), NULL, NULL);
    cl_mem device_buffer_b = clCreateBuffer(context, CL_MEM_READ_ONLY, VECTOR_SIZE * sizeof(int), NULL, NULL);
    cl_mem device_buffer_result = clCreateBuffer(context, CL_MEM_WRITE_ONLY, VECTOR_SIZE * sizeof(int), NULL, NULL);

    // Set kernel arguments
    clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&device_buffer_a);
    clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&device_buffer_b);
    clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&device_buffer_result);
    clSetKernelArg(kernel, 3, sizeof(int), (void*)&VECTOR_SIZE);

    // Create the command queue
    cl_command_queue command_queue = clCreateCommandQueue(
        context, device_id, CL_QUEUE_PROFILING_ENABLE, NULL);

    // Host buffer -> Device buffer
    clEnqueueWriteBuffer(
        command_queue,
        device_buffer_a,
        CL_FALSE,
        0,
        VECTOR_SIZE * sizeof(int),
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
        VECTOR_SIZE * sizeof(int),
        host_buffer_b,
        0,
        NULL,
        NULL
    );

    // Size specification
    size_t local_work_size = 256;
    size_t n_work_groups = (VECTOR_SIZE + local_work_size - 1) / local_work_size;
    size_t global_work_size = n_work_groups * local_work_size;

    // Apply the kernel on the range
    cl_event event;
    clEnqueueNDRangeKernel(
        command_queue,
        kernel,
        1,
        NULL,
        &global_work_size,
        &local_work_size,
        0,
        NULL,
        &event
    );
    clFinish(command_queue);

    // Show profiling information
    cl_ulong start_ns;
    cl_ulong end_ns;
    err = clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_QUEUED,
        sizeof(start_ns),
        &start_ns,
        NULL
    );
    if (err == CL_PROFILING_INFO_NOT_AVAILABLE) {
        printf("Profiling info not available!\n");
        return 0;
    } else if (err != CL_SUCCESS) {
        printf("Error code: %d\n", err);
        return 0;
    }
    clGetEventProfilingInfo(
        event,
        CL_PROFILING_COMMAND_END,
        sizeof(end_ns),
        &end_ns,
        NULL
    );
    double total_time = (double)(end_ns-start_ns) / 1000000000.0;
    printf("%d - Total length in secs: %.6f\n", VECTOR_SIZE, total_time);
    fprintf(file, "%d %.6f\n", VECTOR_SIZE, total_time);
    fflush(file);

    // Host buffer <- Device buffer
    clEnqueueReadBuffer(
        command_queue,
        device_buffer_result,
        CL_TRUE,
        0,
        VECTOR_SIZE * sizeof(int),
        host_buffer_result,
        0,
        NULL,
        NULL
    );

    free(host_buffer_a);
    free(host_buffer_b);
    free(host_buffer_result);
}

//Print Values
    for (i = 0; i < VECTOR_SIZE; ++i) {
        //printf("Vector A: \t%d\t\n", host_buffer_a[i]);
    }

    for (i = 0; i < VECTOR_SIZE; ++i) {
        //printf("Vector B: \t%d\t\n", host_buffer_b[i]);
    }

    for (i = 0; i < VECTOR_SIZE; ++i) {
        //printf("Result: \t%d\t\n", host_buffer_result[i]);
    }
//Print Values

//Release Resources
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseContext(context);
    clReleaseDevice(device_id);

    fclose(file);
//Release Resources
}
