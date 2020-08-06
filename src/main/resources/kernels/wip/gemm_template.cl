
    

    __kernel void gemm_template // ~-=>  2D register blocking ! :
    (
              __global float * drain, __global int *drn_conf,
        const __global float * src1,  __global int *src1_conf,
        const __global float * src2,  __global int *src2_conf,

        //int rank, == 2
        const int d,

        const uint max_ts_row,//  = 128, // ts := tile size
        const uint max_ts_col,//  = 128,
        const uint max_ts_com,//  = 16,
        const uint max_wpt_row,// = 8,   // wpt := work per thread
        const uint max_wpt_col // = 8
    ) {
        const uint max_rts_row = max_ts_row / max_wpt_row; //rts = reduced tile size
        const uint max_rts_col = max_ts_col / max_wpt_col;

        // lpt := loads per thread:
        int max_lpt_src1 = (max_ts_com * max_wpt_row * max_wpt_col) / max_ts_col;

        //  drn   =  src1  x  src2
        // [m, n] = [m, k] x [k, n]
        int prv_drn_cfg[ 2 * 5  ]; _cfg_of_cfg(drn_conf, prv_drn_cfg, rank);
        int prv_src1_cfg[ 2 * 5 ]; _cfg_of_cfg(src1_conf, prv_src1_cfg, rank);
        int prv_src2_cfg[ 2 * 5 ]; _cfg_of_cfg(src2_conf, prv_src2_cfg, rank);

        const int max_row = prv_drn_cfg[ 0 ];
        const int max_col = prv_drn_cfg[ 1 ];   //:= prv_src1_cfg[0]
        const int max_com = prv_src1_cfg[ 1 ];  //:= prv_src2_cfg[0]

        // Thread identifiers
        const int tid_row = get_local_id( 0 );        //:= Local row ID (max: max_ts_row/max_wpt_row)
        const int tid_col = get_local_id( 1 );        //:= Local col ID (max: max_ts_col/max_wpt_col)
        const int offset_max_row = max_ts_row * get_group_id( 0 ); // := Work-group offset
        const int offset_max_col = max_ts_col * get_group_id( 1 ); // := Work-group offset

        // GROUP MEMORY :
        //~~~~~~~~~~~~~~~
        // Local memory to fit a tile of src1 and src2
        __local float loc_tile_src1[ max_ts_com ][ max_ts_row     ];
        __local float loc_tile_src2[ max_ts_col ][ max_ts_com + 2 ];

        // REGISTER MEMORY :
        //~~~~~~~~~~~~~~~~~~
        // Allocate register space
        float reg_tile_src1;
        float reg_tile_src2[ max_wpt_col ];
        float reg_tile_drn[ max_wpt_row ][ max_wpt_col ];
     
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
            
            // Synchronise to make sure the tile is loaded
            barrier(CLK_LOCAL_MEM_FENCE);
     
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
            // Synchronise before loading the next tile
            barrier(CLK_LOCAL_MEM_FENCE);
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