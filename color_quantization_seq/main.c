#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image.h"
#include "include/stb_image_write.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MAX_ITERATIONS 100

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
    // Select k random centroids
    srand((unsigned)time(NULL));
    for (int i = 0; i < k; i++) {
        int index = rand() % npix;
        centroids[i].r = img[4*index+0];
        centroids[i].g = img[4*index+1];
        centroids[i].b = img[4*index+2];
    }
    // More iterations = convergent results
    for (int it = 0; it < max_iter; it++) {
        int changed = 0;
        // Assign every pixel to their closest centroid
        for (int i = 0; i < npix; i++) {
            float pr = img[4*i+0], pg = img[4*i+1], pb = img[4*i+2];
            int best = 0;
            float bd = dist2(pr,pg,pb, centroids[0].r,centroids[0].g,centroids[0].b);
            for (int j = 1; j < k; j++) {
                float d = dist2(pr,pg,pb, centroids[j].r,centroids[j].g,centroids[j].b);
                if (d < bd) { 
                    bd = d;
                    best = j;
                }
            }
            if (labels[i] != best) { 
                labels[i] = best;
                changed++;
            }
        }
        if (!changed) break;
        // Move centroids closer to average
        long *sumr = calloc(k, sizeof *sumr);
        long *sumg = calloc(k, sizeof *sumg);
        long *sumb = calloc(k, sizeof *sumb);
        int *cnt = calloc(k, sizeof *cnt);
        for (int i = 0; i < npix; i++) {
            int c = labels[i];
            sumr[c] += img[4*i+0];
            sumg[c] += img[4*i+1];
            sumb[c] += img[4*i+2];
            cnt[c]++;
        }
        for (int j = 0; j < k; j++) {
            if (cnt[j]) {
                centroids[j].r = sumr[j] / (float)cnt[j];
                centroids[j].g = sumg[j] / (float)cnt[j];
                centroids[j].b = sumb[j] / (float)cnt[j];
            }
        }
        free(sumr);
        free(sumg);
        free(sumb);
        free(cnt);
    }
    free(labels);
    for (int i = 0; i < k; i++){
        printf("Color %d: R: %.00f | G: %.00f | B:%.00f\n", i+1, centroids[i].r, centroids[i].g, centroids[i].b);
    }
    return centroids;
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <palette.txt OR number> <input_image> <output_image>\n", argv[0]);
        return EXIT_FAILURE;
    }
    const char *p = argv[1];
    const char *infile = argv[2];
    const char *outfile = argv[3];
    int w, h, comp;
    unsigned char *img = stbi_load(infile, &w, &h, &comp, 4);
    if (!img) exit(1);
    int npix = w*h;
    Color *palette;
    int pn;
    // Start clock
    clock_t start = clock();
    if (is_number(p)) {
        int k = atoi(p);
        if (k < 1) exit(1);
        palette = kmeans_palette(img, w, h, k, MAX_ITERATIONS);
        pn = k;
    } else {
        palette = load_palette(p, &pn);
    }
    unsigned char *out = malloc(npix * 4);
    // Create new image from palette
    for (int i = 0; i < npix; i++) {
        float pr = img[4*i+0], pg = img[4*i+1], pb = img[4*i+2];
        int best = 0;
        float bd = dist2(pr,pg,pb, palette[0].r,palette[0].g,palette[0].b);
        for (int j = 1; j < pn; j++) {
            float d = dist2(pr,pg,pb, palette[j].r,palette[j].g,palette[j].b);
            if (d < bd) {
                bd = d;
                best = j;
            }
        }
        out[4*i+0] = (unsigned char)(palette[best].r);
        out[4*i+1] = (unsigned char)(palette[best].g);
        out[4*i+2] = (unsigned char)(palette[best].b);
        out[4*i+3] = img[4*i+3];
    }
    stbi_write_png(outfile, w, h, 4, out, w*4);
    free(img);
    free(out);
    free(palette);
    // End clock
    printf("Runtime: %.6f seconds\n", (double)(clock()-start)/CLOCKS_PER_SEC);
    return 0;
}
