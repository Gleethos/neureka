package neureka.ndim.iterators.types.complex.legacy;

import neureka.ndim.iterators.NDIterator;
import neureka.ndim.config.types.complex.ComplexD3Configuration;

public class ComplexLegacyD3CIterator extends ComplexD3Configuration implements NDIterator
{
    protected int _d1 = 0;
    protected int _d2 = 0;
    protected int _d3 = 0;

    public ComplexLegacyD3CIterator(ComplexD3Configuration ndc) {
        super( ndc.shape(), ndc.translation(), ndc.idxmap(), ndc.spread(), ndc.offset() );
    }


    @Override
    public void increment() {
        _d1++;
        if ( _d1 == _shape1 ) {
            _d1 = 0;
            _d2++;
            if ( _d2 == _shape2 ) {
                _d2 = 0;
                _d3++;
            }
        }
    }

    @Override
    public void decrement() {
        if ( _d1 == 0 ) {
            _d1 = _shape1 - 1;
            if ( _d2 == 0 ) {
                _d3--;
            } else _d2--;
        }
    }

    @Override
    public int i() {
        return this.i_of_idx( _d1, _d2, _d3 );
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
    public void set( int[] idx ) {
        _d1 = idx[0];
        _d2 = idx[1];
        _d3 = idx[2];
    }

}
