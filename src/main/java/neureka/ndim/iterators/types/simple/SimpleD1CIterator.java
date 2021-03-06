package neureka.ndim.iterators.types.simple;


import neureka.ndim.config.types.simple.SimpleD1Configuration;
import neureka.ndim.iterators.NDIterator;

public final class SimpleD1CIterator extends SimpleD1Configuration implements NDIterator
{
    private int _d1;

    public SimpleD1CIterator( SimpleD1Configuration ndc ) {
        super( ndc.shape( 0 ), ndc.translation( 0 ));
    }


    @Override
    public void increment() {
        _d1++;
    }

    @Override
    public void decrement() {
        _d1--;
    }


    @Override
    public int i() {
        return this.indexOfIndices(_d1);
    }

    @Override
    public int get( int axis ) {
        return _d1;
    }

    @Override
    public int[] get() {
        return new int[]{_d1};
    }

    @Override
    public void set( int axis, int position ) {
        _d1 = position;
    }

    @Override
    public void set( int[] indices) {
        _d1 = indices[0];
    }

    @Override
    public int rank() {
        return 1;
    }



}
