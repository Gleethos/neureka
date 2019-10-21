
void cfg_of_cfg(__global int* cfg, int* prv_cfg, int rank);

int i_of_i(int i, int* config, int rank);

__kernel void multiply(
    __global float *drn,
    __global int *drn_conf,
    __global float *src1,
    __global int *src1_conf,
    __global float *src2,
    __global int *src2_conf,
    int rank,
    int d
    ){
        int prv_drn_cfg[32]; cfg_of_cfg(drn_conf, prv_drn_cfg, rank);
        int prv_src1_cfg[32]; cfg_of_cfg(src1_conf, prv_src1_cfg, rank);
        int prv_src2_cfg[32]; cfg_of_cfg(src2_conf, prv_src2_cfg, rank);

        unsigned int i = get_global_id(0);
        drn[i_of_i(i, prv_drn_cfg, rank)]
        = src1[i_of_i(i, prv_src1_cfg, rank)]
            *src2[i_of_i(i, prv_src2_cfg, rank)];
    }
