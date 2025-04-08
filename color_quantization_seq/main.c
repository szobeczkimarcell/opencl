#define STB_IMAGE_IMPLEMENTATION
#include "include/stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "include/stb_image_write.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Color;


int read_palette(const char* palette_filename, Color** palette) {
    FILE* fp = fopen(palette_filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Unable to open palette file '%s'\n", palette_filename);
        return 0;
    }

    char line[128];
    int color_count = 0;
    int capacity = 10;

    *palette = (Color*)malloc(capacity * sizeof(Color));
    if (!(*palette)) {
        fprintf(stderr, "Error: Could not allocate memory for palette.\n");
        fclose(fp);
        return 0;
    }

    while (fgets(line, sizeof(line), fp)) {
        if (line[0] == '\n' || line[0] == '\0') {
            continue;
        }

        unsigned int r, g, b;
        if (sscanf(line, "#%02x%02x%02x", &r, &g, &b) == 3) {
            if (color_count >= capacity) {
                capacity += 5;
                Color* temp = (Color*)realloc(*palette, capacity * sizeof(Color));
                if (!temp) {
                    fprintf(stderr, "Error: Could not reallocate memory.\n");
                    fclose(fp);
                    free(*palette);
                    return 0;
                }
                *palette = temp;
            }

            (*palette)[color_count].r = (unsigned char)r;
            (*palette)[color_count].g = (unsigned char)g;
            (*palette)[color_count].b = (unsigned char)b;

            color_count++;
        }
    }

    fclose(fp);
    return color_count;
}

static inline Color nearest_color(unsigned char r, unsigned char g, unsigned char b,
                                  Color* palette, int palette_size) {
    double min_dist = (double)INT_MAX;
    Color nearest = {0, 0, 0};

    for (int i = 0; i < palette_size; i++) {
        double dr = (double)r - palette[i].r;
        double dg = (double)g - palette[i].g;
        double db = (double)b - palette[i].b;
        double dist = dr*dr + dg*dg + db*db;
        if (dist < min_dist) {
            min_dist = dist;
            nearest = palette[i];
        }
    }

    return nearest;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <palette.txt> <input_image> <output_image>\n", argv[0]);
        return 1;
    }

    const char* palette_filename = argv[1];
    const char* input_filename = argv[2];
    const char* output_filename = argv[3];

    /* Read palette from file */
    Color* palette = NULL;
    int palette_size = read_palette(palette_filename, &palette);
    if (palette_size == 0) {
        fprintf(stderr, "Error: No colors read from palette or file not found.\n");
        return 1;
    }

    int width, height, channels;
    unsigned char* data = stbi_load(input_filename, &width, &height, &channels, 0);
    if (!data) {
        fprintf(stderr, "Error: Could not load image '%s'.\n", input_filename);
        free(palette);
        return 1;
    }

    size_t pixel_count = (size_t)width * height;

    for (size_t i = 0; i < pixel_count; i++) {
        size_t idx = i * channels;

        unsigned char r = data[idx + 0];
        unsigned char g = data[idx + 1];
        unsigned char b = data[idx + 2];
        
        Color c = nearest_color(r, g, b, palette, palette_size);

        data[idx + 0] = c.r;
        data[idx + 1] = c.g;
        data[idx + 2] = c.b;
    }

    if (!stbi_write_png(output_filename, width, height, channels, data, width * channels)) {
        fprintf(stderr, "Error: Failed to write image '%s'.\n", output_filename);
        stbi_image_free(data);
        free(palette);
        return 1;
    }

    stbi_image_free(data);
    free(palette);

    printf("Successfully wrote output to '%s'\n", output_filename);
    return 0;
}
