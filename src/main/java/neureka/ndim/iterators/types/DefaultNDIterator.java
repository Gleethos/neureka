package neureka.ndim.iterators.types;

import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterators.NDIterator;

public class DefaultNDIterator implements NDIterator
{
    private final int[] _idx;
    private final int[] _shape;
    private final NDConfiguration _conf;

    public DefaultNDIterator( NDConfiguration ndc ) {
        _shape = ndc.shape();
        _idx = new int[ _shape.length ];
        _conf = ndc;
    }

    @Override
    public int shape( int i ) {
        return _shape[i];
    }

    @Override
    public int[] shape() {
        return _shape;
    }

    @Override
    public void increment() {
        NDConfiguration.Utility.increment( _idx, _shape );
    }

    @Override
    public void decrement() {
        //NDConfiguration.Utility.
    }

    @Override
    public int i() {
        return _conf.i_of_idx( _idx );
    }

    @Override
    public int get( int axis ) {
        return _idx[ axis ];
    }

    @Override
    public int[] get() {
        return _idx;
    }

    @Override
    public void set( int axis, int position ) {
        _idx[ axis ] = position;
    }

    @Override
    public void set( int[] idx ) {
        System.arraycopy( idx, 0, _idx, 0, idx.length );
    }

    @Override
    public int rank() {
        return _shape.length;
    }



}
