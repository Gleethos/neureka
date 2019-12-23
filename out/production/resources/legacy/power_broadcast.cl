
void _cfg_of_cfg(__global int* cfg, int* prv_cfg, int rank);
int _i_of_i(int i, int* cfg, int rank);

__kernel void power_broadcast(
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
            drn[_i_of_i(i, prv_drn_cfg, rank)] =
                    pow(src1[_i_of_i(i, prv_src1_cfg, rank)], value);
        } else {
            if(d==0){
                drn[_i_of_i(i, prv_drn_cfg, rank)] =
                    value*pow(src1[_i_of_i(i, prv_src1_cfg, rank)], value-(float)1);
            } else {
                drn[_i_of_i(i, prv_drn_cfg, rank)] =
                    pow(src1[_i_of_i(i, prv_src1_cfg, rank)], value)
                        *log(src1[_i_of_i(i, prv_src1_cfg, rank)]);
            }
        }

    }
