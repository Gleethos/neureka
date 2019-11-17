
void _cfg_of_cfg(__global int* cfg, int* new_cfg, int rank){
    for(int i=0; i<rank*3; i++){
        new_cfg[i] = cfg[i];
    }
    for(int i=rank*3; i<rank*4; i++){
            new_cfg[i]=0;
        }
}

int _i_of_i(int i, int* cfg, int rank){ // cfg:   <[ shape | translation | idxmap | idx ]>
    int* idx = (cfg+rank*3);
    for(int ii=(rank)-1; ii>=0; ii--){
        idx[ii] = i/cfg[2*rank+ii];//is derived from the shape of a tensor. Translates scalar index to dim-Index
        i %= cfg[2*rank+ii];
    }
    for(int ii=0; ii<rank; ii++){
        i += idx[ii]*cfg[rank+ii];
    }
    return i;
}

//...

int _increment_At(int* cfg, int ri, int rank) {
    if (cfg[rank*3+ri] < (cfg[ri])) {
        cfg[rank*3+ri]++;//TODO...
        //idx[idx_ptr+ri]++;
        if (cfg[rank*3+ri] == (cfg[ri])) {
            cfg[rank*3+ri] = 0;
            ri++;
        } else {
            ri = -1;
        }
    } else {
        ri++;
    }
    return ri;
}
void _increment_idx(int* cfg, int rank) {
    int ri = 0;
    while (ri >= 0 && ri < rank) {
        ri = _increment_At(cfg, ri, rank);
    }
}
int _i_of_idx_on_tln(int* cfg, int rank) {
    int i = 0;
    for (int ii = 0; ii < rank; ii++) {
        i += cfg[rank+ii] * cfg[3*rank+ii];
        // i += _translations[p_tln+ii] * idx[idx_ptr+ii];
    }
    return i;
}
int _i_of_idx_on_shp(int gid, int* cfg, int rank){
    for(int i=0; i<gid; i++){
        _increment_idx(cfg, rank);
    }
    return _i_of_idx_on_tln(cfg, rank);
}