package neureka.ndim.iterators.types.complex.main;

import neureka.ndim.config.types.complex.ComplexD2Configuration;
import neureka.ndim.iterators.NDIterator;

public final class ComplexD2CIterator extends ComplexD2Configuration implements NDIterator
{
    protected int _d1 = 0;
    protected int _d2 = 0;

    public ComplexD2CIterator(ComplexD2Configuration ndc) {
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
        if ( _d2 == 0 ) {
            _d1--;
            _d2 = _shape2 - 1;
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