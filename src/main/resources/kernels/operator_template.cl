/*
    The kernel defined in this file takes a pair of tensors
    and combines them element wise to produce an output tensor.
    The kernel is mostly used for operators like "+", "-", "*", "/" ...
*/
//======================================================================================================================

    // These are from the "utility.cl" file! The methods convert types of indices...
    void _cfg_of_cfg(__global int* cfg, int* prv_cfg, int rank);
    int _i_of_i(int i, int* cfg, int rank);

//======================================================================================================================

    __kernel void operator_template(
        __global float *drn, __global int *drn_conf,
        __global float *src1, __global int *src1_conf,
        __global float *src2, __global int *src2_conf,
        int rank,
        int d
    ){
        int prv_drn_cfg[32];
        _cfg_of_cfg(drn_conf, prv_drn_cfg, rank);
        int prv_src1_cfg[32];
        _cfg_of_cfg(src1_conf, prv_src1_cfg, rank);
        int prv_src2_cfg[32];
        _cfg_of_cfg(src2_conf, prv_src2_cfg, rank);

        unsigned int i = get_global_id( 0 );

        if(d<0){
//-=<OPERATION>=-//
            drn[_i_of_i(i, prv_drn_cfg, rank)] = src1[_i_of_i(i, prv_src1_cfg, rank)] + src2[_i_of_i(i, prv_src2_cfg, rank)];
//-=<OPERATION>=-//
        } else {
//-=<OPERATION>=-//
            drn[_i_of_i(i, prv_drn_cfg, rank)] = 1;
//-=<OPERATION>=-//
        }

    }

//======================================================================================================================
