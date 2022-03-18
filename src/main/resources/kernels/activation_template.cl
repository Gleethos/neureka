/*
    The kernel define in this file is used to perform element wise
    operations on the data within nd-arrays / tensors.
    Contrary to other element wise kernels, this kernel takes a single nd-array as input.
    The kernel is a rather high level kernel because it take into account the
    entirety of an nd-configuration of a given nd-array / tensor.
    This means that the supplied nd-array might be a slice of another tensor
    which might also have strides or is positioned by an offset within
    some data array...
    This kernel is built to support all of this for element wise operations.

*/
//======================================================================================================================

    // These are from the "utility.cl" file! The methods convert types of indices...
    void _cfg_of_cfg(__global int* cfg, int* prv_cfg, int rank);
    int _i_of_i(int i, int* cfg, int rank);

    float fast_inverse_sqrt( float n );

//======================================================================================================================

    __kernel void activation_template(
        __global float *drn, __global int *drn_conf,
        __global float *src1, __global int *src1_conf,
        int rank,
        int d
    ){
        int prv_drn_cfg[32];
        _cfg_of_cfg(drn_conf, prv_drn_cfg, rank);
        int prv_src1_cfg[32];
        _cfg_of_cfg(src1_conf, prv_src1_cfg, rank);

        unsigned int i = get_global_id( 0 );

        if(d<0){
//-=<OPERATION>=-//
            drn[_i_of_i(i, prv_drn_cfg, rank)] = cos(src1[_i_of_i(i, prv_src1_cfg, rank)]);
//-=<OPERATION>=-//
        }else{
//-=<OPERATION>=-//
            drn[_i_of_i(i, prv_drn_cfg, rank)] = -sin(src1[_i_of_i(i, prv_src1_cfg, rank)]);
//-=<OPERATION>=-//
        }
    }

//======================================================================================================================
