package it.calculus.mocks;

import neureka.Neureka;


public class GEMMKernelReferenceImplementation
{

    private CLContext _CLContext;

    public GEMMKernelReferenceImplementation(CLContext CLContext){
        _CLContext = CLContext;
    }

    /*                                                                           col
                                                                    +-------------------------------+
                             o            #                         |       o       #       o       |
                             o            #                         |       o       #       o       |
                             o . . . . . .#. . . . . . . . . . . . .|. . . .o       #       o       |
                             o            #                         |       o       #       o       |
                  o o o o o o             #                         | o o o o o o o # o o o o o o o |
                       .                  #                         |       o       #       o       |
                       .                  #                         |       o       #       o       |
                       .                  #. . . . . . . . . . . . .|. . . .o. . . .#       o       |
                       .                  #                         |       o       #       o       |
                  ########################                          | ############################# |
                       .           .                                |       o       #   .   o       |
                       .           .                                |       o       #   .   o       |
                       .           .                                |       o       #   .   o       |
                       .           .                                |       o       #   .   o       |
                       .           .                                | o o o o o o o # o o o o o o o |
                       .           .                             /  |       o       #   .   o   .   |
                       .           .                     com        |       o       #   .   o   .   |
                       .           .                             \  |       o       #   .   o   .   |
                       .           .                   /     \      |       o       #   .   o   .   |
                +---------------------------------------------------+-------------------------------+
                |      .     o     .      #            o            |                   .       .
                |      .     o     .      #            o            |                   .       .
                | o o o o o o o o o o o o # o o o o o o o o o o o o |                   .       .
                |            o     .      #            o            |                   .       .
                |            o     .      #            o            |                   .       .
                | ################################################# |                ##############
              / |            o            #            o            |               #           .
           row  |            o            #. . . . . . o . . . . . .|. . . . . . . .#           .
              \ | o o o o o o o o o o o o # o o o o o o o o o o o o |               #           .
                |            o            #            o            |               #           .
                |            o            #            o            |               #           .
                | ################################################# |                           .
                |            o            #            o            |                           .
                |            o            #            o            |                           .
                | o o o o o o o o o o o o # o o o o o o o o o o o o |                         o o o
                |            o            #            o . . . . . .|. . . . . . . . . . . . o
                |            o            #            o            |                        o
                +---------------------------------------------------+


     */

    //__kernel
    public void gemm_template // ~-=>  2D register blocking ! :
    (
        //__global
        float[] drain,
        //__global
        int[] drn_conf,
        //const __global
        float[] src1,
        //__global
        int[] src1_conf,
        //const __global
        float[] src2,
        //__global
        int[] src2_conf,

        //int rank, == 2
        //const
        int d,
        //const u
        int max_ts_row,//  = 128, // ts := tile size
        //const u
        int max_ts_col,//  = 128,
        //const u
        int max_ts_com,//  = 16,
        //const u
        int max_wpt_row,// = 8,   // wpt := work per thread
        //const u
        int max_wpt_col // = 8,
    ) {
        //const u
        int max_rts_row = max_ts_row / max_wpt_row; //rts = reduced tile size
        //const u
        int max_rts_col = max_ts_col / max_wpt_col;

        // lpt := loads per thread:
        int max_lpt_src1 = (max_ts_com * max_wpt_row * max_wpt_col) / max_ts_col;


        // Constraints on settings for kernels 6 -- 10
        assertInt( (double)max_ts_row/ (double)max_wpt_row );
        assertInt( (double)max_ts_col/ (double)max_wpt_col );

        // Note: max_ts_row/WIDTH has to be integer // Dont know width...
        // Note: max_ts_col/WIDTH has to be integer
        // Note: ( max_ts_com * max_wpt_row*max_wpt_col )/( max_ts_col*WIDTH ) has to be integer
        // Note: ( max_ts_com * max_wpt_row*max_wpt_col )/( max_ts_row*WIDTH ) has to be integer


        //  drn   =  src1  x  src2
        // [m, n] = [m, k] x [k, n]
        int[] prv_drn_cfg  = new int[ 2 * 6  ]; _cfg_of_cfg(drn_conf, prv_drn_cfg, 2);
        int[] prv_src1_cfg = new int[ 2 * 6 ]; _cfg_of_cfg(src1_conf, prv_src1_cfg, 2);
        int[] prv_src2_cfg = new int[ 2 * 6 ]; _cfg_of_cfg(src2_conf, prv_src2_cfg, 2);

        //const
        int max_row = prv_drn_cfg[ 0 ];
        //const
        int max_col = prv_drn_cfg[ 1 ];   //:= prv_src1_cfg[0]
        //const
        int max_com = prv_src1_cfg[ 1 ];  //:= prv_src2_cfg[0]

        // Thread identifiers
        //const
        int tid_row = _CLContext.get_local_id( 0 );        //:= Local row ID (max: max_ts_row/max_wpt_row)
        //const
        int tid_col = _CLContext.get_local_id( 1 );        //:= Local col ID (max: max_ts_col/max_wpt_col)
        //const
        int offset_max_row = max_ts_row * _CLContext.get_group_id( 0 ); // := Work-group offset
        //const
        int offset_max_col = max_ts_col * _CLContext.get_group_id( 1 ); // := Work-group offset

        // GROUP MEMORY :
        //~~~~~~~~~~~~~~~
        // Local memory to fit a tile of src1 and src2
        //__local
        float[][] loc_tile_src1 = new float[ max_ts_com ][ max_ts_row     ];
        //__local
        float[][] loc_tile_src2 = new float[ max_ts_col ][ max_ts_com + 2 ];

        // REGISTER MEMORY :
        //~~~~~~~~~~~~~~~~~~
        // Allocate register space
        float reg_tile_src1;
        float[] reg_tile_src2 = new float[ max_wpt_col ];
        float[][] reg_tile_drn = new float[ max_wpt_row ][ max_wpt_col ];

        // Initialise the accumulation registers
        for ( int wm = 0; wm < max_wpt_row; wm++ ) {
            for ( int wn = 0; wn < max_wpt_col; wn++ ) {
                reg_tile_drn[ wm ][ wn ] = 0.0f;
            }
        }

        // Loop over all tiles
        int numTiles = max_com / max_ts_com;
        for ( int t = 0; t < numTiles; t++ ) {

            // Load one tile of src1 and src2 into local memory
            for ( int la = 0; la < max_lpt_src1; la++ ) {
                int tid = tid_col * max_rts_row + tid_row;
                int id  = la * max_rts_col*max_rts_row + tid;
                int row = id % max_ts_row; // row index for local memory!
                int col = id / max_ts_row; // col index for local memory!
                int tiledIndex = max_ts_com * t + col;
                loc_tile_src1[ col ][ row ] = src1[ tiledIndex * max_row + offset_max_row + row ];
                loc_tile_src2[ row ][ col ] = src2[ tiledIndex * max_col + offset_max_col + row ];
                // _i_of_idx_on_tln(prv_src1_cfg, 2)

            }

            // Synchronise to make sure the tile is loaded!
            //barrier(CLK_LOCAL_MEM_FENCE);

            // Loop over the values of a single tile
            for ( int k = 0; k < max_ts_com; k++ ) {

                // Cache the values of loc_tile_src2 in registers
                for ( int wn = 0; wn < max_wpt_col; wn++ ) {
                    int col = tid_col + wn * max_rts_col;
                    reg_tile_src2[ wn ] = loc_tile_src2[ col ][ k ];
                }

                // Perform the computation
                for ( int wm = 0; wm < max_wpt_row; wm++ ) {
                    int row = tid_row + wm * max_rts_row;
                    reg_tile_src1 = loc_tile_src1[ k ][ row ];
                    for ( int wn = 0; wn < max_wpt_col; wn++ ) {
                        reg_tile_drn[ wm ][ wn ] += reg_tile_src1 * reg_tile_src2[ wn ];
                    }
                }
            }
            // Synchronise before loading the next tile!
            //barrier(CLK_LOCAL_MEM_FENCE);
        }

        // Store the final results in drain
        for ( int wm = 0; wm < max_wpt_row; wm++ ) {
            int globalRow = offset_max_row+ tid_row + wm * max_rts_row;
            for ( int wn = 0; wn < max_wpt_col; wn++ ) {
                int globalCol = offset_max_col + tid_col + wn * max_rts_col;
                drain[ globalCol * max_row + globalRow ] = reg_tile_drn[ wm ][ wn ];
            }
        }
    }

    //---

    private void assertInt(double num){
        assert ((num == Math.floor(num)) && !Double.isInfinite(num));
    }


    //======================================================================================================================

    private void _cfg_of_cfg(int[] cfg, int[] new_cfg, int rank)
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

//======================================================================================================================

    private int _i_of_i(int i, int[] cfg, int rank)// cfg:   <[ shape | translation | idxMap | idx | idxScale | idxBase ]>
    {
        int[] idx = new int[rank];//(cfg+rank*3);
        int[] idxMap = new int[rank];//(cfg+rank*2);
        for ( int ii = 0 ; ii < rank; ii++) idx[ii] = cfg[rank*3+ii];
        for ( int ii = 0 ; ii < rank; ii++) idxMap[ii] = cfg[rank*2+ii];
        if(Neureka.instance().settings().indexing().isUsingLegacyIndexing()){
            for(int ii=(rank-1); ii>=0; ii--){
                idx[ii] = (i/idxMap[ii]);//is derived from the shape of a tensor. Translates scalar indexAlias to dim-Index
                i %= idxMap[ii];
            }
        } else {//---
            for(int ii=0; ii<rank; ii++){
                idx[ii] = (i/idxMap[ii]);//is derived from the shape of a tensor. Translates scalar indexAlias to dim-Index
                i %= idxMap[ii];
            }
        }
        return _i_of_idx_on_tln(cfg, rank);
    }


    private int _i_of_idx_on_tln(int[] cfg, int rank) // cfg:   <[ 0:shape | 1:translation | 2:idxMap | 3:idx | 4:idxScale | 5:idxBase ]>
    {
        int[] idxBase = new int[rank];//(cfg+rank*5);
        int[] idxScale = new int[rank];//(cfg+rank*4);
        int[] idx = new int[rank];//(cfg+rank*3);
        int[] translation = new int[rank];//(cfg+rank);
        for ( int ii = 0 ; ii < rank; ii++) idxBase[ii] = cfg[rank*5+ii];
        for ( int ii = 0 ; ii < rank; ii++) idxScale[ii] = cfg[rank*4+ii];
        for ( int ii = 0 ; ii < rank; ii++) idx[ii] = cfg[rank*3+ii];
        for ( int ii = 0 ; ii < rank; ii++) translation[ii] = cfg[rank+ii];
        int i = 0;
        for (int ii = 0; ii < rank; ii++) {
            i += (idx[ii]*idxScale[ii]+idxBase[ii]) * translation[ii];
        }
        return i;
    }

//======================================================================================================================



}
