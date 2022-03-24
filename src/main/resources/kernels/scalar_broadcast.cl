/*

*/
//======================================================================================================================

    // These are from the "utility.cl" file! The methods convert types of indices...
    void _cfg_of_cfg( __global int* cfg, int* prv_cfg, int rank );
    int _i_of_i( int i, int* cfg, int rank );

//======================================================================================================================

    __kernel void scalar_broadcast(
        __global float *drn, __global int *drn_conf,
        float value,
        int rank
    ) {
        int prv_drn_cfg[32];
        _cfg_of_cfg(drn_conf, prv_drn_cfg, rank);

        unsigned int i = get_global_id( 0 );

        drn[_i_of_i(i, prv_drn_cfg, rank)] = value;
    }

//======================================================================================================================
