package neureka.ndim.iterators.types.sliced;

import neureka.ndim.configs.NDConfiguration;
import neureka.ndim.iterators.NDIterator;

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
    public int shape( int i ) {
        return _shape[i];
    }

    @Override
    public int[] shape() {
        return _shape;
    }

    @Override
    public void increment() {
        NDConfiguration.Utility.increment( _indices, _shape );
    }

    @Override
    public void decrement() {
        NDConfiguration.Utility.decrement( _indices, _shape );
    }

    @Override
    public int i() {
        return _conf.indexOfIndices(_indices);
    }

    @Override
    public int get( int axis ) {
        return _indices[ axis ];
    }

    @Override
    public int[] get() {
        return _indices;
    }

    @Override
    public void set( int axis, int position ) {
        _indices[ axis ] = position;
    }

    @Override
    public void set( int[] indices ) {
        System.arraycopy( indices, 0, _indices, 0, indices.length );
    }

    @Override
    public int rank() {
        return _shape.length;
    }



}
