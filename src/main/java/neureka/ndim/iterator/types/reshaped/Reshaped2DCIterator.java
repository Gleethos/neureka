package neureka.ndim.iterator.types.reshaped;

import neureka.ndim.config.types.reshaped.Reshaped2DConfiguration;
import neureka.ndim.iterator.NDIterator;

public class Reshaped2DCIterator extends Reshaped2DConfiguration implements NDIterator
{
    private int _d1 = 0;
    private int _d2 = 0;

    public Reshaped2DCIterator(Reshaped2DConfiguration ndc) {
        super( ndc.shape(), ndc.translation(), ndc.indicesMap() );
    }

    @Override
    public final void increment() {
        _d2++;
        if ( _d2 == _shape2 ) {
            _d2 = 0;
            _d1++;
        }
    }

    @Override
    public final void decrement() {
        _d2--;
        if ( _d2 == -1 ) {
            _d2 = _shape2 - 1;
            _d1--;
        }
    }

    @Override
    public final int i() {
        return this.indexOfIndices( _d1, _d2 );
    }

    @Override
    public final int get( int axis ) {
        return ( axis == 0 ) ? _d1 : _d2;
    }

    @Override
    public final int[] get() {
        return new int[]{ _d1, _d2 };
    }

    @Override
    public final void set( int axis, int position ) {
        if ( axis == 0 ) _d1 = position;
        else _d2 = position;
    }

    @Override
    public final void set( int[] indices ) {
        _d1 = indices[ 0 ];
        _d2 = indices[ 1 ];
    }

}
