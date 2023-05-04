package neureka.ndim.iterator.types.permuted;

import neureka.ndim.config.types.permuted.Permuted2DConfiguration;
import neureka.ndim.iterator.NDIterator;

public final class Permuted2DCIterator extends Permuted2DConfiguration implements NDIterator
{
    private int _d1 = 0;
    private int _d2 = 0;

    public Permuted2DCIterator(Permuted2DConfiguration ndc) {
        super( ndc.shape(), ndc.strides(), ndc.indicesMap() );
    }

    /** {@inheritDoc} */
    @Override public final void increment() {
        _d2++;
        if ( _d2 == _shape2 ) {
            _d2 = 0;
            _d1++;
        }
    }

    /** {@inheritDoc} */
    @Override public final void decrement() {
        _d2--;
        if ( _d2 == -1 ) {
            _d2 = _shape2 - 1;
            _d1--;
        }
    }

    /** {@inheritDoc} */
    @Override public final int i() { return this.indexOfIndices( _d1, _d2 ); }

    /** {@inheritDoc} */
    @Override public final int get( int axis ) { return ( axis == 0 ? _d1 : _d2 ); }

    /** {@inheritDoc} */
    @Override public final int[] get() { return new int[]{ _d1, _d2 }; }

    /** {@inheritDoc} */
    @Override public final void set( int axis, int position ) {
        if ( axis == 0 ) _d1 = position;
        else _d2 = position;
    }

    /** {@inheritDoc} */
    @Override public final void set( int[] indices ) {
        _d1 = indices[ 0 ];
        _d2 = indices[ 1 ];
    }

}
