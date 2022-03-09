package neureka.ndim.iterator.types.sliced;

import neureka.ndim.config.NDConfiguration;
import neureka.ndim.iterator.NDIterator;

public final class SlicedNDIterator implements NDIterator
{
    private final int[] _indices;
    private final int[] _shape;
    private final NDConfiguration _conf;

    public SlicedNDIterator(NDConfiguration ndc ) {
        _shape = ndc.shape();
        _indices = new int[ _shape.length ];
        _conf = ndc;
    }

    @Override
    public final int shape( int i ) {
        return _shape[i];
    }

    @Override
    public final int[] shape() {
        return _shape;
    }

    @Override
    public final void increment() {
        NDConfiguration.Utility.increment( _indices, _shape );
    }

    @Override
    public final void decrement() {
        NDConfiguration.Utility.decrement( _indices, _shape );
    }

    @Override
    public final int i() {
        return _conf.indexOfIndices(_indices);
    }

    @Override
    public final int get( int axis ) {
        return _indices[ axis ];
    }

    @Override
    public final int[] get() {
        return _indices;
    }

    @Override
    public final void set( int axis, int position ) {
        _indices[ axis ] = position;
    }

    @Override
    public final void set( int[] indices ) {
        System.arraycopy( indices, 0, _indices, 0, indices.length );
    }

    @Override
    public final int rank() {
        return _shape.length;
    }



}
