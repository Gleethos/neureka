package neureka.ndim.iterators.types.simple.main;

import neureka.ndim.config.NDIterator;
import neureka.ndim.config.types.simple.SimpleD3Configuration;

public class SimpleD3CIterator extends SimpleD3Configuration implements NDIterator
{
    protected int _d1 = 0;
    protected int _d2 = 0;
    protected int _d3 = 0;

    public SimpleD3CIterator(SimpleD3Configuration ndc) {
        super( ndc.shape(), ndc.translation() );
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