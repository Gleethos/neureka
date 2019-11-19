
void _cfg_of_cfg(__global int* cfg, int* new_cfg, int rank){
    for(int i=0; i<rank*5; i++){
        new_cfg[i] = cfg[i];
    }
}

int _i_of_i(int i, int* cfg, int rank){ // cfg:   <[ shape | translation | idxMap | idx | idxScale ]>
    int* idxScale = (cfg+rank*4);
    int* idx = (cfg+rank*3);
    int* idxMap = (cfg+rank*2);
    int* translation = (cfg+rank);
    for(int ii=(rank)-1; ii>=0; ii--){
        idx[ii] += (i/idxMap[ii])*idxScale[ii];//is derived from the shape of a tensor. Translates scalar index to dim-Index
        i %= idxMap[ii];//(i / t._idxmap[ii])*((baseIdx==null)?1:baseIdx[t.rank()+ii])
    }
    for(int ii=0; ii<rank; ii++){
        i += idx[ii]*translation[ii];
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