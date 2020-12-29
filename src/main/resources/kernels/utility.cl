
//======================================================================================================================

void _cfg_of_cfg(__global int* cfg, int* new_cfg, int rank)
{
    for(int i=0; i<rank*5; i++)
    {
        if(i>=rank*3 && i<rank*4){
            new_cfg[i+2*rank] = cfg[ i ];
        } else {
            new_cfg[ i ] = cfg[ i ];
        }
    }
}

//======================================================================================================================

int _i_of_i(int i, int* cfg, int rank)// cfg:   <[ shape | translation | idxMap | idx | idxScale | idxBase ]>
{
    int* idx = (cfg+rank*3);
    int* idxMap = (cfg+rank*2);
    if(Neureka.instance().settings().indexing().REVERSE_INDEX_TRANSLATION){
        for(int ii=(rank)-1; ii>=0; ii--){
            idx[ ii ] = (i/idxMap[ ii ]);//is derived from the shape of a tensor. Translates scalar indexAlias to dim-Index
            i %= idxMap[ ii ];
        }
    } else {//---
        for(int ii=0; ii<rank; ii++){
            idx[ ii ] = (i/idxMap[ ii ]);//is derived from the shape of a tensor. Translates scalar indexAlias to dim-Index
            i %= idxMap[ ii ];
        }
    }
    return _i_of_idx_on_tln(cfg, rank);
}

//======================================================================================================================

int _i_of_idx_on_tln(int* cfg, int rank) // cfg:   <[ 0:shape | 1:translation | 2:idxMap | 3:idx | 4:idxScale | 5:idxBase ]>
{
    int* idxBase = (cfg+rank*5);
    int* idxScale = (cfg+rank*4);
    int* idx = (cfg+rank*3);
    int* translation = (cfg+rank);
    int i = 0;
    for ( int ii = 0; ii < rank; ii++ ) {
        i += (idx[ ii ]*idxScale[ ii ]+idxBase[ ii ]) * translation[ ii ];
    }
    return i;
}

//======================================================================================================================