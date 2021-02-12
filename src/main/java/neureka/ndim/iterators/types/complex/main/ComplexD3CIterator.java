package neureka.ndim.iterators.types.complex.main;

import neureka.ndim.iterators.NDIterator;
import neureka.ndim.config.types.complex.ComplexD3Configuration;

public final class ComplexD3CIterator extends ComplexD3Configuration implements NDIterator
{
    protected int _d1 = 0;
    protected int _d2 = 0;
    protected int _d3 = 0;

    public ComplexD3CIterator(ComplexD3Configuration ndc) {
        super( ndc.shape(), ndc.translation(), ndc.indicesMap(), ndc.spread(), ndc.offset() );
    }


    @Override
    public void increment() {
        _d3++;
        if ( _d3 == _shape3 ) {
            _d3 = 0;
            _d2++;
            if ( _d2 == _shape2 ) {
                _d2 = 0;
                _d1++;
            }
        }
    }

    @Override
    public void decrement() {
        if ( _d3 == 0 ) {
            _d3 = _shape3 - 1;
            if ( _d2 == 0 ) {
                _d1--;
            } else _d2--;
        }
    }

    @Override
    public int i() {
        return this.indexOfIndices( _d1, _d2, _d3 );
    }

    @Override
    public int get( int axis ) {
        return ( axis == 0 ) ? _d1 : ( axis == 1 ) ? _d2 : _d3;
    }

    @Override
    public int[] get() {
        return new int[]{ _d1, _d2, _d3 };
    }

    @Override
    public void set( int axis, int position ) {
        if ( axis == 0 ) _d1 = position;
        else if ( axis == 1 ) _d2 = position;
        else _d3 = position;
    }

    @Override
    public void set( int[] indices ) {
        _d1 = indices[0];
        _d2 = indices[1];
        _d3 = indices[2];
    }

}