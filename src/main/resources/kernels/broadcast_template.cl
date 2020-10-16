
void _cfg_of_cfg(__global int* cfg, int* prv_cfg, int rank);
int  _i_of_i(int i, int* config, int rank);
int  _i_of_idx_on_tln(int* conf, int rank);

__kernel void broadcast_template(
    //-=<ARGUMENT>=-//__global float *frn, __global int *frn_conf,
    __global float *drn, __global int *drn_conf,
    __global float *src1, __global int *src1_conf,
    __global float *src2, __global int *src2_conf,
    int rank,
    int d
    ){
        int prv_drn_cfg[32]; _cfg_of_cfg(drn_conf, prv_drn_cfg, rank);
        int prv_src1_cfg[32]; _cfg_of_cfg(src1_conf, prv_src1_cfg, rank);
        int prv_src2_cfg[32]; _cfg_of_cfg(src2_conf, prv_src2_cfg, rank);
        //-=<CONFIGURATION>=-//int prv_frn_cfg[32]; _cfg_of_cfg(frn_conf, prv_frn_cfg, rank);

        int p_shp = 0 * rank;
        int p_tln = 1 * rank;
        int p_idm = 2 * rank;
        int p_idx = 3 * rank;
        int di = _i_of_i(get_global_id( 0 ), prv_drn_cfg, rank);

        //increment src accordingly:
        int ri = 0;
        if(d < 0){
            while (ri < rank) {
                if (prv_src1_cfg[p_shp+ri] == prv_src2_cfg[p_shp+ri]) {
                    prv_src1_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                    prv_src2_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                } else if (prv_src1_cfg[p_shp+ri] > prv_src2_cfg[p_shp+ri]) {
                    prv_src1_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                    prv_src2_cfg[p_idx+ri] = 0;
                } else if(prv_src1_cfg[p_shp+ri] < prv_src2_cfg[p_shp+ri]){
                    prv_src1_cfg[p_idx+ri] = 0;
                    prv_src2_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                }
                ri++;
            }
            //----------
            // multiplication:
            float value = 0;

//-=<OPERATION>=-//
            value += src1[_i_of_idx_on_tln(prv_src1_cfg, rank)] * src2[_i_of_idx_on_tln(prv_src2_cfg, rank)];
//-=<OPERATION>=-//

            //set _value in drn:
            drn[di] = value;

        } else {// conv

            while (ri < rank) {
                if (prv_drn_cfg[p_shp+ri] == prv_src1_cfg[p_shp+ri]) {
                    prv_src1_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                    prv_src2_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                } else if (prv_drn_cfg[p_shp+ri] > prv_src1_cfg[p_shp+ri]) {
                    prv_src1_cfg[p_idx+ri] = 0;
                    prv_src2_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                }
                ri++;
            }
            //----------
            // multiplication:
            float value = 0;
            bool running = true;
            bool incrementing = false;
            while (running) {
                ri = (ri==rank)?0:ri;
                if (!incrementing) {
//-=<OPERATION>=-//
                    value += src1[_i_of_idx_on_tln(prv_src1_cfg, rank)] * src2[_i_of_idx_on_tln(prv_src2_cfg, rank)];
//-=<OPERATION>=-//
                    incrementing = true;
                    ri=0;
                } else {//incrementing:
                    if (prv_drn_cfg[p_shp+ri] < prv_src1_cfg[p_shp+ri]) {
                        prv_src1_cfg[p_idx+ri]++;
                        prv_src2_cfg[p_idx+ri]++;
                        if (prv_src1_cfg[p_idx+ri] == prv_src1_cfg[p_shp+ri]) {
                            prv_src1_cfg[p_idx+ri] = 0;
                            prv_src2_cfg[p_idx+ri] = 0;
                            running = running && !(ri == (rank - 1));
                            ri++;
                        } else {
                            incrementing = false;//return to calculation!
                        }
                    } else {
                        running = running && !(ri == (rank - 1));
                        ri++;
                    }
                }
            }
            drn[di] = value;
        }
    }


