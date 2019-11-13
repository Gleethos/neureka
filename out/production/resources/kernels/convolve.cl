
void _cfg_of_cfg(__global int* cfg, int* prv_cfg, int rank);
int  _i_of_i(int i, int* config, int rank);
int  _i_of_idx_on_tln(int* conf, int rank);
void _increment_idx(int* conf, int rank);
int  _increment_At(int* conf, int ri, int rank);
int  _i_of_idx_on_shp(int gid, int* conf, int rank);

__kernel void convolve(
    __global float *drn, __global int *drn_conf,
    __global float *src1, __global int *src1_conf,
    __global float *src2, __global int *src2_conf,
    int rank,
    int d
    ){
        unsigned int gid = get_global_id(0);
        int prv_drn_cfg[32]; _cfg_of_cfg(drn_conf, prv_drn_cfg, rank);
        int prv_src1_cfg[32]; _cfg_of_cfg(src1_conf, prv_src1_cfg, rank);
        int prv_src2_cfg[32]; _cfg_of_cfg(src2_conf, prv_src2_cfg, rank);

        int p_shp = 0 * rank;
        int p_tln = 1 * rank;
        int p_idm = 2 * rank;
        int p_idx = 3 * rank;
        
        //increment on drain:
        for(int i=0; i<gid; i++){
            _increment_idx(prv_drn_cfg, rank);
        }
        //i_of_i(i, prv_drn_cfg, rank)
        //increment src accordingly:
        int ri = 0;
        if(d >= 0){
            while (ri < rank) {
                if (prv_src2_cfg[p_idx+ri] == prv_src2_cfg[p_shp+ri]) {
                    prv_src1_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                    prv_src2_cfg[p_idx+ri] = 0;
                } else {
                    if (prv_drn_cfg[p_shp+ri] > prv_src1_cfg[p_shp+ri]) {
                        prv_src1_cfg[p_idx+ri] = (prv_drn_cfg[p_idx+ri] - prv_src2_cfg[p_idx+ri]);
                    } else {
                        prv_src1_cfg[p_idx+ri] = (prv_drn_cfg[p_idx+ri] + prv_src2_cfg[p_idx+ri]);
                    }
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
                if (incrementing == false) {
                    bool isMatch = true;
                    for(int i=0; i<rank; i++){
                        if(!(prv_src1_cfg[p_idx+ri] < prv_src1_cfg[i] && prv_src1_cfg[p_idx+ri]>=0)){
                            isMatch = false;
                        }
                    }
                    if(isMatch){
                        int i1 = _i_of_idx_on_tln(prv_src1_cfg, rank);
                        int i2 = _i_of_idx_on_tln(prv_src2_cfg, rank);
                        value += src1[i1] * src2[i2];
                    }
                    incrementing = true;
                    ri=0;
                } else {//incrementing:
                    if (prv_src2_cfg[p_idx+ri] < prv_src2_cfg[p_shp+ri]) {
                        prv_src2_cfg[p_idx+ri]++;
                        if (prv_src2_cfg[p_idx+ri] == prv_src2_cfg[p_shp+ri]) {
                            if (ri == (rank - 1)) {
                                running = false;
                            }
                            prv_src1_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                            prv_src2_cfg[p_idx+ri] = 0;
                            ri++;
                        } else {
                            if (prv_drn_cfg[p_shp+ri] > prv_src1_cfg[p_shp+ri]) {//TODO:THIS IS ADDED
                                prv_src1_cfg[p_idx+ri] = (prv_drn_cfg[p_idx+ri] - prv_src2_cfg[p_idx+ri]);
                            } else {
                                prv_src1_cfg[p_idx+ri] = (prv_drn_cfg[p_idx+ri] + prv_src2_cfg[p_idx+ri]);
                            }
                            incrementing = false;
                            ri=0;
                        }
                    } else {
                        ri++;
                    }
                }
            }
            //set _value in drn:
            int di = _i_of_idx_on_tln(prv_drn_cfg, rank);
            drn[di] = value;
        } else {// conv
            while (ri < rank) {
                if (prv_src1_cfg[p_shp+ri] == prv_src2_cfg[p_shp+ri]) {//setting 0
                    prv_src1_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                    prv_src2_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                } else if (prv_src1_cfg[p_shp+ri] > prv_src2_cfg[p_shp+ri]) {//setting src1 idx to id idx
                    prv_src1_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];
                    prv_src2_cfg[p_idx+ri] = 0;
                } else if (prv_src1_cfg[p_shp+ri] < prv_src2_cfg[p_shp+ri]) {//setting src2 idx to id idx
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
                if (incrementing == false) {
                    int i1 = _i_of_idx_on_tln(prv_src1_cfg, rank);
                    int i2 = _i_of_idx_on_tln(prv_src2_cfg, rank);
                    value += src1[i1] * src2[i2];
                    incrementing = true;
                    ri=0;
                } else {//incrementing:
                    if (prv_src1_cfg[p_idx+ri] < prv_src1_cfg[p_shp+ri] && prv_src2_cfg[p_idx+ri] < prv_src2_cfg[p_shp+ri]) {
                        prv_src1_cfg[p_idx+ri]++;
                        prv_src2_cfg[p_idx+ri]++;
                        if (prv_src1_cfg[p_idx+ri] == prv_src1_cfg[p_shp+ri] || prv_src2_cfg[p_idx+ri] == prv_src2_cfg[p_shp+ri]) {
                            if (ri == (rank - 1)) {
                                running = false;
                            }
                            if (prv_src1_cfg[p_shp+ri] == prv_src2_cfg[p_shp+ri]) {//setting 0
                                prv_src1_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];//mtch[mi];
                                prv_src2_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];//mtch[mi];
                            } else if (prv_src1_cfg[p_shp+ri] > prv_src2_cfg[p_shp+ri]) {//setting hdr1 idx to id idx
                                prv_src1_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];//mtch[mi];
                                prv_src2_cfg[p_idx+ri] = 0;
                            } else if (prv_src1_cfg[p_shp+ri] < prv_src2_cfg[p_shp+ri]) {//setting hdr2 idx to id idx
                                prv_src1_cfg[p_idx+ri] = 0;
                                prv_src2_cfg[p_idx+ri] = prv_drn_cfg[p_idx+ri];//mtch[mi];
                            }
                            ri++;
                        } else {
                            incrementing = false;
                            ri=0;
                        }
                    } else {
                        ri++;
                    }
                }
            }
            //set _value in drn:
            //int di = __i_of_idx_on_tln(prv_drn_cfg, rank);
            int di = _i_of_i(gid, prv_drn_cfg, rank);
            drn[di] = value;
        }

    
    }


