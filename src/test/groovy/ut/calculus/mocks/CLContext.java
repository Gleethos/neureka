package ut.calculus.mocks;

public class CLContext {

    int[] _local;
    int[] _global;
    int _lws;
    int _gws;
    int _wgs;

    public CLContext(
            int lws,
            int gws,
            int max_ts_row,//  = 128, // ts := tile size
            //const u
            int max_ts_col,//  = 128,
            //const u
            int max_wpt_row,// = 8,   // wpt := work per thread
            //const u
            int max_wpt_col, // = 8,
            int row_sze,
            int col_sze
    ) {
        assertInt((double)gws / (double)lws);
        int wgs = gws / lws;
        _lws = lws;
        _gws = gws;
        _wgs = wgs;

        _local =  new int[]{ max_ts_row/max_wpt_row, max_ts_col/max_wpt_col }; // Or { RTSM, RTSN };
        _global = new int[]{ row_sze/max_wpt_row, col_sze/max_wpt_col };

    }

    private int _counter;

    int get_local_id( int dimension ) {
        int i = _counter % _lws;

        if ( dimension==0 ) return i % _local[0];
        else return i / _local[1];
    }

    int get_group_id( int dimension ) {
        return _counter / _lws;
    }

    private void increment(){
        _counter++;
    }

    private void assertInt(double num){
        assert ((num == Math.floor(num)) && !Double.isInfinite(num));
    }

}
