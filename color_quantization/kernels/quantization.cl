__kernel void assign_labels(
    __global const uchar4* img,
    __global const float* c,
    int k,
    __global int* lbl,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;

    float3 p = convert_float3(img[i].xyz);
    int best = 0;
    float3 cp0 = (float3)(c[0], c[1], c[2]);
    float bd = dot(p - cp0, p - cp0);

    for (int j = 1; j < k; ++j) {
        float3 cp = (float3)(c[3*j+0], c[3*j+1], c[3*j+2]);
        float d = dot(p - cp, p - cp);
        if (d < bd) {
            bd = d;
            best = j;
        }
    }

    lbl[i] = best;
}

__kernel void map_palette(
    __global const uchar4* img,
    __global const float* pal,
    int pn,
    __global uchar4* out,
    int n
) {
    int i = get_global_id(0);
    if (i >= n) return;

    float3 p = convert_float3(img[i].xyz);
    int best = 0;
    float3 cp0 = (float3)(pal[0], pal[1], pal[2]);
    float bd = dot(p - cp0, p - cp0);

    for (int j = 1; j < pn; ++j) {
        float3 cp = (float3)(pal[3*j+0], pal[3*j+1], pal[3*j+2]);
        float d = dot(p - cp, p - cp);
        if (d < bd) {
            bd = d;
            best = j;
        }
    }

    uchar4 res;
    res.x = (uchar)pal[3*best+0];
    res.y = (uchar)pal[3*best+1];
    res.z = (uchar)pal[3*best+2];
    res.w = img[i].w;
    out[i] = res;
}
