package neureka.ndim.iterators.types.sliced;

import neureka.ndim.configs.types.sliced.Sliced2DConfiguration;
import neureka.ndim.iterators.NDIterator;

public final class Sliced2DCIterator extends Sliced2DConfiguration implements NDIterator
{
    private int _d1 = 0;
    private int _d2 = 0;

    public Sliced2DCIterator(Sliced2DConfiguration ndc) {
        super( ndc.shape(), ndc.translation(), ndc.indicesMap(), ndc.spread(), ndc.offset() );
    }


    @Override
    public void increment() {
        _d2++;
        if ( _d2 == _shape2 ) {
            _d2 = 0;
            _d1++;
        }
    }

    @Override
    public void decrement() {
        _d2--;
        if ( _d2 == -1 ) {
            _d2 = _shape2 - 1;
            _d1--;
        }
    }

    @Override
    public int i() {
        return this.indexOfIndices( _d1, _d2 );
    }

    @Override
    public int get( int axis ) {
        return ( axis == 0 ) ? _d1 : _d2;
    }

    @Override
    public int[] get() {
        return new int[]{ _d1, _d2 };
    }

    @Override
    public void set( int axis, int position ) {
        if ( axis == 0 ) _d1 = position;
        else _d2 = position;
    }

    @Override
    public void set( int[] indices ) {
        _d1 = indices[ 0 ];
        _d2 = indices[ 1 ];
    }

}