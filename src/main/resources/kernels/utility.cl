/*
    This file doe not contain an OpenCL kernel!
    It is merely hosting various utility methods which are generally needed
    by other OpenCL kernels.
*/
//======================================================================================================================

/*
    This method simply copies the values from one array into another.
    The mentioned arrays are called "cfg" which simply means "configuration",
    more specifically they should be called "ndConfigurationAndIteratorArray" because it
    contains everything that is needed for iterating and translating an array of indexed axis
    to a single true index targeting an actual element within the data array of a given nd-array (tensor).
*/
void _cfg_of_cfg(__global int* cfg, int* new_cfg, int rank)
{
    for( int i = 0; i < rank * 5; i++ )
    {
        if( i >= rank * 3 && i < rank * 4 ){
            new_cfg[ i + 2 * rank ] = cfg[ i ];
        } else {
            new_cfg[ i ] = cfg[ i ];
        }
    }
}

//======================================================================================================================

/*
    The following method calculates the true index for an element in the data array
    based on a provided "virtual index" and relevant context information which is needed for translation.
    This virtual index might be different from the true index, for example because it is
    a slice of another larger nd-array, or maybe because it is in fact a reshaped version of another nd-array.
    This virtual index will be turned in an index array which defines the position for every axis.
    Then this index array will be converted into the final and true index for an underlying item.
    The information needed for performing this translation is contained within the "cfg" array,
    which has the size (6 * rank * sizeof(int)) and contains everything
    needed to treat a given block of data as an nd-array!
*/
int _i_of_i( int i, int* cfg, int rank )// cfg:   <[ shape | translation | indicesMap | indices | indicesScale | idxBase ]>
{
    int* idx    = (cfg+rank*3);
    int* idxMap = (cfg+rank*2);
    for( int ii = 0; ii < rank; ii++ ) {
        idx[ ii ] = ( i / idxMap[ ii ] ); // is derived from the shape of a tensor. Translates scalar index to dim-Index
        i %= idxMap[ ii ];
    }
    return _i_of_idx_on_tln( cfg, rank );
}

//======================================================================================================================

/*
    The following method calculates the true index for an element in the data array
    based on a provided index array and relevant context information which are needed for translation.
    All of this is contained within the "cfg" array, which has the size (6 * rank * sizeof(int)) and
    contains everything needed to treat a given block of data as an nd-array!
*/
int _i_of_idx_on_tln( int* cfg, int rank ) // cfg:   <[ 0:shape | 1:translation | 2:idxMap | 3:idx | 4:idxScale | 5:idxBase ]>
{
    int* idxBase     = ( cfg + rank * 5 );
    int* idxScale    = ( cfg + rank * 4 );
    int* idx         = ( cfg + rank * 3 );
    int* translation = ( cfg + rank     );
    int i = 0;
    for ( int ii = 0; ii < rank; ii++ ) {
        i += ( idx[ ii ] * idxScale[ ii ] + idxBase[ ii ] ) * translation[ ii ];
    }
    return i;
}

//======================================================================================================================