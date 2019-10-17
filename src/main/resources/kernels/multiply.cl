
__kernel void multiply(
    __global float *drain,
    __global float *src1,
    __global float *src2,
    int d
    ){
    unsigned int i = get_global_id(0);

    output[mul24((int)iy, tileSizeX)+ix] = iteration;
}

inline void _i(int i){


}

