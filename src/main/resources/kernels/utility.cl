
void _cfg_of_cfg(__global int* cfg, int* new_cfg, int rank)
{
    for(int i=0; i<rank*5; i++)
    {
        if(i>=rank*3 && i<rank*4){
            new_cfg[i+2*rank] = cfg[i];
        } else {
            new_cfg[i] = cfg[i];
        }
    }
}

int _i_of_i(int i, int* cfg, int rank)// cfg:   <[ shape | translation | idxMap | idx | idxScale | idxBase ]>
{
    int* idxBase = (cfg+rank*5);
    int* idxScale = (cfg+rank*4);
    int* idx = (cfg+rank*3);
    int* idxMap = (cfg+rank*2);
    int* translation = (cfg+rank);

    for(int ii=(rank)-1; ii>=0; ii--){
        idx[ii] = (i/idxMap[ii])*idxScale[ii];//is derived from the shape of a tensor. Translates scalar index to dim-Index
        i %= idxMap[ii];
    }
    for(int ii=0; ii<rank; ii++){
        i += (idx[ii]+idxBase[ii])*translation[ii];
    }
    return i;
}

//...

//int _increment_At(int* cfg, int ri, int rank) {
//    if (cfg[rank*3+ri] < (cfg[ri])) {
//        cfg[rank*3+ri]++;//TODO...
//        //idx[idx_ptr+ri]++;
//        if (cfg[rank*3+ri] == (cfg[ri])) {
//            cfg[rank*3+ri] = 0;
//            ri++;
//        } else {
//            ri = -1;
//        }
//    } else {
//        ri++;
//    }
//    return ri;
//}
//void _increment_idx(int* cfg, int rank) {
//    int ri = 0;
//    while (ri >= 0 && ri < rank) {
//        ri = _increment_At(cfg, ri, rank);
//    }
//}


int _i_of_idx_on_tln(int* cfg, int rank) // cfg:   <[ 0:shape | 1:translation | 2:idxMap | 3:idx | 4:idxScale | 5:idxBase ]>
{
    int* idxBase = (cfg+rank*5);
    int* idx = (cfg+rank*3);
    int* translation = (cfg+rank);
    int i = 0;
    for (int ii = 0; ii < rank; ii++) {
        i += (idx[ii]+idxBase[ii]) * translation[ii];
    }
    return i;
}
