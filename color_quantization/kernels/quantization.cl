__kernel void mapToPalette(__global const uchar4* inputImage, __global uchar4* outputImage, __global const float4* palette, const uint paletteSize)
{
    int gid = get_global_id(0);

    // Extract pixel from input
    float3 pixColor = (float3)(
        (float)inputImage[gid].x,
        (float)inputImage[gid].y,
        (float)inputImage[gid].z
    );

    float minDist = 1e30f;
    int bestColor = 0;

    // Find the palette color with the minimum distance
    for (int i = 0; i < paletteSize; i++) {
        float3 palColor = palette[i].xyz;
        float3 diff = pixColor - palColor;
        float dist = dot(diff, diff);
        if (dist < minDist) {
            minDist = dist;
            bestColor = i;
        }
    }

    // Assign that color to the output
    outputImage[gid] = (uchar4)(
        (uchar)palette[bestColor].x,
        (uchar)palette[bestColor].y,
        (uchar)palette[bestColor].z,
        inputImage[gid].w
    );
};
