package neureka.ndim.iterators.types.complex;

import neureka.ndim.iterators.NDIterator;
import neureka.ndim.config.types.complex.ComplexD1Configuration;

public class ComplexD1CIterator extends ComplexD1Configuration implements NDIterator
{
    private int _d1;

    public ComplexD1CIterator(ComplexD1Configuration ndc) {
        super(ndc.shape(0), ndc.translation(0), ndc.idxmap(0), ndc.spread(0), ndc.offset(0));
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
        return this.i_of_idx(_d1);
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
    public void set( int[] idx ) {
        _d1 = idx[0];
    }

    @Override
    public int rank() {
        return 1;
    }



}
