package neureka.ndim.iterators;

import neureka.Tsr;
import neureka.ndim.config.types.D1C;

public class D1Iterator extends AbstractNDIterator
{
    private int _idx;
    private final int _shape;
    private final D1C _conf;

    public D1Iterator( D1C ndc ) {
        _shape = ndc.shape(0);
        _idx = 0;
        _conf = ndc;
    }

    @Override
    public int[] shape() {
        return new int[]{_shape};
    }

    @Override
    public void increment() {
        _idx++;
    }

    @Override
    public void decrement() {
        _idx--;
    }

    @Override
    public int i() {
        return _conf.i_of_idx( _idx );
    }

    @Override
    public int get( int axis ) {
        return _idx;//_idx[ axis ];
    }

    @Override
    public int[] get() {
        return new int[]{_idx};
    }

    @Override
    public void set( int axis, int position ) {
        _idx = position;
    }

    @Override
    public void set( int[] idx ) {
        _idx = idx[0];
    }

    @Override
    public int rank() {
        return 1;
    }



}
