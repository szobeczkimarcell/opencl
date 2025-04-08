#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "kernel_loader.h"

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void loadPalette(const char* paletteFile, float** paletteOut, int* paletteSizeOut) {
    FILE* fp = fopen(paletteFile, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open palette file: %s\n", paletteFile);
        exit(EXIT_FAILURE);
    }

    // Allocate space for up to 1024 colors, each with 4 floats (R, G, B, A/pad)
    float* tmp = (float*)malloc(1024 * 4 * sizeof(float));
    int count = 0;

    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '#') {
            unsigned int r, g, b;
            if (sscanf(line + 1, "%02x%02x%02x", &r, &g, &b) == 3) {
                tmp[count * 4 + 0] = (float)r;    // R
                tmp[count * 4 + 1] = (float)g;    // G
                tmp[count * 4 + 2] = (float)b;    // B
                tmp[count * 4 + 3] = 0.0f;        // Padding or optional Alpha
                count++;
                if (count >= 1024) {
                    fprintf(stderr, "Palette too large, max 1024\n");
                    break;
                }
            }
        }
    }
    fclose(fp);

    // Shrink to exact size
    *paletteOut = (float*)malloc(count * 4 * sizeof(float));
    memcpy(*paletteOut, tmp, count * 4 * sizeof(float));
    for (int i = 0; i < count; ++i) {
        float r = (*paletteOut)[i * 4 + 0];
        float g = (*paletteOut)[i * 4 + 1];
        float b = (*paletteOut)[i * 4 + 2];
        printf("  Color %d: R=%.0f, G=%.0f, B=%.0f\n", i, r, g, b);
    }
    *paletteSizeOut = count;

    free(tmp);
}

#define CHECK_CL(err, msg) \
    if ((err) != CL_SUCCESS) { \
        fprintf(stderr, "OpenCL Error %d at %s\n", (int)(err), msg); \
        exit(EXIT_FAILURE); \
    }

int main(int argc, char** argv){
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <palette.txt> <input_image> <output_image>\n", argv[0]);
        return EXIT_FAILURE;
    }

    float* palette = NULL;
    int    paletteSize = 0;
    loadPalette(argv[1], &palette, &paletteSize);
    if (paletteSize == 0) {
        fprintf(stderr, "Palette is empty or could not be read.\n");
        return EXIT_FAILURE;
    }
    printf("Loaded %d colors in the palette.\n", paletteSize);

    int width, height, channels;
    unsigned char* inputImage = stbi_load(argv[2], &width, &height, &channels, 4);
    if (!inputImage) {
        fprintf(stderr, "Failed to load image: %s\n", argv[2]);
        free(palette);
        return EXIT_FAILURE;
    }
    int numPixels = width * height;
    printf("Number of pixels: %d\n", numPixels);

    unsigned char* outputImage = (unsigned char*)malloc(numPixels * 4);

    // OpenCL
    cl_int err;
    cl_uint numPlatforms;
    err = clGetPlatformIDs(0, NULL, &numPlatforms);
    CHECK_CL(err, "clGetPlatformIDs (num)");
    if (numPlatforms == 0) {
        fprintf(stderr, "No OpenCL platforms found.\n");
        return EXIT_FAILURE;
    }
    cl_platform_id* platforms = (cl_platform_id*)malloc(sizeof(cl_platform_id) * numPlatforms);
    err = clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_CL(err, "clGetPlatformIDs");

    cl_platform_id platform = platforms[0];
    free(platforms);

    cl_device_id device;
    err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
    if (err != CL_SUCCESS) {
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, &device, NULL);
        CHECK_CL(err, "clGetDeviceIDs (CPU fallback)");
    }

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
    CHECK_CL(err, "clCreateContext");

    cl_command_queue queue = clCreateCommandQueue(context, device, 0, &err);
    CHECK_CL(err, "clCreateCommandQueue");

    int error_code;
    const char* kernelSource = load_kernel_source("kernels/quantization.cl", &error_code);
    size_t srcSize = strlen(kernelSource);
    cl_program program = clCreateProgramWithSource(context, 1, &kernelSource, &srcSize, &err);
    CHECK_CL(err, "clCreateProgramWithSource");

    clBuildProgram(program, 1, &device, NULL, NULL, NULL);

    cl_kernel kernel = clCreateKernel(program, "mapToPalette", &err);
    CHECK_CL(err, "clCreateKernel");

    cl_mem inputBuf = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        numPixels * 4 * sizeof(unsigned char),
        inputImage,
        &err
    );
    CHECK_CL(err, "clCreateBuffer(inputBuf)");

    cl_mem outputBuf = clCreateBuffer(
        context,
        CL_MEM_WRITE_ONLY,
        numPixels * 4 * sizeof(unsigned char),
        NULL,
        &err
    );
    CHECK_CL(err, "clCreateBuffer(outputBuf)");

    cl_mem paletteBuf = clCreateBuffer(
        context,
        CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        paletteSize * 4 * sizeof(float),
        palette,
        &err
    );
    CHECK_CL(err, "clCreateBuffer(paletteBuf)");

    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &inputBuf);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &outputBuf);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &paletteBuf);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &paletteSize);
    CHECK_CL(err, "clSetKernelArg");

    size_t globalWorkSize = numPixels;
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalWorkSize, NULL, 0, NULL, NULL);
    CHECK_CL(err, "clEnqueueNDRangeKernel");

    err = clEnqueueReadBuffer(
        queue,
        outputBuf,
        CL_TRUE,
        0,
        numPixels * 4 * sizeof(unsigned char),
        outputImage,
        0,
        NULL,
        NULL
    );
    CHECK_CL(err, "clEnqueueReadBuffer");


    clReleaseMemObject(inputBuf);
    clReleaseMemObject(outputBuf);
    clReleaseMemObject(paletteBuf);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

 
    if (!stbi_write_png(argv[3], width, height, 4, outputImage, width * 4)) {
        fprintf(stderr, "Failed to write output image: %s\n", argv[3]);
    } else {
        printf("Output image saved as %s\n", argv[3]);
    }

    free(inputImage);
    free(outputImage);
    free(palette);

    return EXIT_SUCCESS;
}
