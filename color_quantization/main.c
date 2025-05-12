#define CL_TARGET_OPENCL_VERSION 220
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image.h"
#include "include/stb_image_write.h"
#include "kernel_loader.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <CL/cl.h>

#define MAX_ITERATIONS 100

// Error check
#define CL_CHECK(x) do{ cl_int err = x; if(err!=CL_SUCCESS){fprintf(stderr,"OpenCL error %d at %s:%d\n",err,__FILE__,__LINE__); exit(EXIT_FAILURE);} }while(0)

static cl_context cl_ctx;
static cl_command_queue cl_q;
static cl_program cl_prog;
static cl_kernel k_assign, k_map;
static const char *kernel_src;

static void init_opencl(void){
    int error_code;
    kernel_src = load_kernel_source("kernels/quantization.cl", &error_code);
    cl_platform_id pf;
    cl_device_id dev;
    CL_CHECK(clGetPlatformIDs(1,&pf,NULL));
    CL_CHECK(clGetDeviceIDs(pf,CL_DEVICE_TYPE_GPU,1,&dev,NULL));
    cl_ctx = clCreateContext(NULL,1,&dev,NULL,NULL,NULL);
    cl_q = clCreateCommandQueue(cl_ctx,dev,0,NULL);
    cl_prog = clCreateProgramWithSource(cl_ctx,1,&kernel_src,NULL,NULL);
    CL_CHECK(clBuildProgram(cl_prog,1,&dev,"",NULL,NULL));
    k_assign = clCreateKernel(cl_prog,"assign_labels",NULL);
    k_map = clCreateKernel(cl_prog,"map_palette",NULL);
}

typedef struct{
    float r, g, b;
} Color;

static inline float dist2(float r1, float g1, float b1, float r2, float g2, float b2) {
    float dr = r1 - r2;
    float dg = g1 - g2;
    float db = b1 - b2;
    return dr*dr + dg*dg + db*db;
}

int is_number(const char *s) {
    if (!*s) return 0;
    while (*s) {
        if (*s < '0' || *s > '9') return 0;
        s++;
    }
    return 1;
}

Color *load_palette(const char *paletteFile, int *paletteSizeOut) {
    FILE *fp = fopen(paletteFile, "r");
    if (!fp) {
        fprintf(stderr, "Failed to open palette file");
        exit(EXIT_FAILURE);
    }

    int capacity = 1024;
    Color *palette = malloc(capacity * sizeof(Color));
    if (!palette) {
        fprintf(stderr, "Out of memory\n");
        exit(EXIT_FAILURE);
    }

    int count = 0;
    char line[256];
    while (fgets(line, sizeof(line), fp)) {
        if (line[0] != '#') continue;
        unsigned int r, g, b;
        if (sscanf(line + 1, "%02x%02x%02x", &r, &g, &b) != 3) continue;

        palette[count].r = (float)r;
        palette[count].g = (float)g;
        palette[count].b = (float)b;
        printf("Color %d: R:%.0f, G:%.0f, B:%.0f\n", count+1, palette[count].r, palette[count].g, palette[count].b);

        if (++count >= capacity) {
            fprintf(stderr, "Palette too large, max %d\n", capacity);
            break;
        }
    }
    fclose(fp);

    if (count < capacity) {
        palette = realloc(palette, count * sizeof(Color));
        if (!palette && count > 0) {
            fprintf(stderr, "Realloc failed\n");
            exit(EXIT_FAILURE);
        }
    }

    *paletteSizeOut = count;
    return palette;
}

Color *kmeans_palette(unsigned char *img, int w, int h, int k, int max_iter) {
    int npix = w*h;
    Color *centroids = malloc(k * sizeof *centroids);
    int *labels = malloc(npix * sizeof *labels);
    srand((unsigned)time(NULL));
    for(int i = 0; i < k; i++) {
        int index=rand()%npix;
        centroids[i].r=img[4*index+0];
        centroids[i].g=img[4*index+1];
        centroids[i].b=img[4*index+2];
    }

    // OpenCL Buffers
    cl_mem d_img = clCreateBuffer(cl_ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,npix*4, img,NULL);
    cl_mem d_cent= clCreateBuffer(cl_ctx,CL_MEM_READ_ONLY, k*3*sizeof(float),NULL,NULL);
    cl_mem d_lbl = clCreateBuffer(cl_ctx,CL_MEM_READ_WRITE, npix*sizeof(int),NULL,NULL);

    size_t gsz = npix;
    for(int it = 0; it < max_iter; it++){
        int changed=0;
        float *cent_flat=malloc(k*3*sizeof(float));
        for(int j=0;j<k;j++){
            cent_flat[3*j+0]=centroids[j].r;
            cent_flat[3*j+1]=centroids[j].g;
            cent_flat[3*j+2]=centroids[j].b;
        }

        CL_CHECK(clEnqueueWriteBuffer(cl_q,d_cent,CL_TRUE,0,k*3*sizeof(float),cent_flat,0,NULL,NULL));
        clSetKernelArg(k_assign,0,sizeof(cl_mem),&d_img);
        clSetKernelArg(k_assign,1,sizeof(cl_mem),&d_cent);
        clSetKernelArg(k_assign,2,sizeof(int),&k);
        clSetKernelArg(k_assign,3,sizeof(cl_mem),&d_lbl);
        clSetKernelArg(k_assign,4,sizeof(int),&npix);
        CL_CHECK(clEnqueueNDRangeKernel(cl_q,k_assign,1,NULL,&gsz,NULL,0,NULL,NULL));
        CL_CHECK(clFinish(cl_q));
        CL_CHECK(clEnqueueReadBuffer(cl_q,d_lbl,CL_TRUE,0,npix*sizeof(int),labels,0,NULL,NULL));
        free(cent_flat);

        long *sr=calloc(k,sizeof*sr), *sg=calloc(k,sizeof*sg), *sb=calloc(k,sizeof*sb);
        int *cnt=calloc(k,sizeof*cnt);
        for(int i = 0; i < npix; i++){
            int c=labels[i];
            long pr=img[4*i+0], pg=img[4*i+1], pb=img[4*i+2];
            sr[c] += pr;
            sg[c] += pg;
            sb[c] += pb;
            cnt[c]++;
        }
        for(int j = 0; j <k ; j++){
            if(cnt[j]){
                float nr = sr[j]/(float)cnt[j];
                float ng = sg[j]/(float)cnt[j];
                float nb = sb[j]/(float)cnt[j];
                if(nr!=centroids[j].r || ng != centroids[j].g || nb !=centroids[j].b){
                    changed = 1;
                    centroids[j].r = nr;
                    centroids[j].g = ng;
                    centroids[j].b = nb;
                }
            }
        }
        free(sr);
        free(sg);
        free(sb);
        free(cnt);
        if(!changed) break;
    }
    clReleaseMemObject(d_img);
    clReleaseMemObject(d_cent);
    clReleaseMemObject(d_lbl);
    free(labels);
    for (int i = 0; i < k; i++){
        printf("Color %d: R: %.00f | G: %.00f | B:%.00f\n", i+1, centroids[i].r, centroids[i].g, centroids[i].b);
    }
    return centroids;
}

int main(int argc,char **argv){
    if(argc<4){
        fprintf(stderr,"Usage: %s <palette.txt|number> <input> <output>\n",argv[0]);
        return 1;
    }
    const char *p=argv[1], *in=argv[2], *outf=argv[3];
    int w, h, comp;
    unsigned char *img = stbi_load(in,&w,&h,&comp,4);
    if(!img){
        fprintf(stderr,"Image load fail\n");
        return 1;
    }
    init_opencl();
    int npix = w*h;
    Color *palette;
    int pn;
    clock_t start = clock();
    if(is_number(p)){
        int k=atoi(p);
        palette=kmeans_palette(img,w,h,k,MAX_ITERATIONS);
        pn=k;
    }
    else palette=load_palette(p,&pn);

    cl_mem d_img = clCreateBuffer(cl_ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,npix*4,img,NULL);
    float *pal_flat=malloc(pn*3*sizeof(float));
    for(int j=0;j<pn;j++){
        pal_flat[3*j+0]=palette[j].r;
        pal_flat[3*j+1]=palette[j].g;
        pal_flat[3*j+2]=palette[j].b;
    }
    cl_mem d_pal = clCreateBuffer(cl_ctx,CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR,pn*3*sizeof(float),pal_flat,NULL);
    cl_mem d_out = clCreateBuffer(cl_ctx,CL_MEM_WRITE_ONLY,npix*4,NULL,NULL);

    clSetKernelArg(k_map,0,sizeof(cl_mem),&d_img);
    clSetKernelArg(k_map,1,sizeof(cl_mem),&d_pal);
    clSetKernelArg(k_map,2,sizeof(int),&pn);
    clSetKernelArg(k_map,3,sizeof(cl_mem),&d_out);
    clSetKernelArg(k_map,4,sizeof(int),&npix);
    size_t gsz=npix;
    CL_CHECK(clEnqueueNDRangeKernel(cl_q,k_map,1,NULL,&gsz,NULL,0,NULL,NULL));
    CL_CHECK(clFinish(cl_q));

    unsigned char *out=malloc(npix*4);
    CL_CHECK(clEnqueueReadBuffer(cl_q,d_out,CL_TRUE,0,npix*4,out,0,NULL,NULL));
    stbi_write_png(outf,w,h,4,out,w*4);

    // Free
    free(img);
    free(out);
    free(palette);
    free(pal_flat);
    clReleaseMemObject(d_img);
    clReleaseMemObject(d_pal);
    clReleaseMemObject(d_out);
    clReleaseKernel(k_assign);
    clReleaseKernel(k_map);
    clReleaseProgram(cl_prog);
    clReleaseCommandQueue(cl_q);
    clReleaseContext(cl_ctx);

    // Runtime
    printf("Runtime: %.6f seconds\n", (double)(clock()-start)/CLOCKS_PER_SEC);
    return 0;
}
