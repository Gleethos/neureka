
void _cfg_of_cfg(__global int* cfg, int* prv_cfg, int rank);
int _i_of_i(int i, int* cfg, int rank);

__kernel void divide_broadcast_inv(
    __global float *drn, __global int *drn_conf,
    __global float *src1, __global int *src1_conf,
    float value,
    int rank,
    int d
    ){
        int prv_drn_cfg[32];
        _cfg_of_cfg(drn_conf, prv_drn_cfg, rank);
        int prv_src1_cfg[32];
        _cfg_of_cfg(src1_conf, prv_src1_cfg, rank);

        unsigned int i = get_global_id(0);
        if(d<0){
            drn[_i_of_i(i, prv_drn_cfg, rank)] = value/src1[_i_of_i(i, prv_src1_cfg, rank)];
        } else {
            if(d==0){
                drn[_i_of_i(i, prv_drn_cfg, rank)] = 1 / src1[_i_of_i(i, prv_src1_cfg, rank)];
            } else {
                drn[_i_of_i(i, prv_drn_cfg, rank)] =
                 (-1/(float)pow(src1[_i_of_i(i, prv_src1_cfg, rank)], (float)2))
                 * value;
            }
        }
    }
