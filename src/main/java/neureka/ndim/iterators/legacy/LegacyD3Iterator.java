package neureka.ndim.iterators.legacy;

import neureka.ndim.config.types.D3C;
import neureka.ndim.iterators.AbstractNDIterator;

public class LegacyD3Iterator extends AbstractNDIterator
{
    private int _d1;
    private int _d2;
    private int _d3;
    private final int _s1;
    private final int _s2;
    private final int _s3;
    private final D3C _conf;

    public LegacyD3Iterator( D3C ndc ) {
        _s1 = ndc.shape( 0 );
        _s2 = ndc.shape( 1 );
        _s3 = ndc.shape( 2 );
        _d1 = 0;
        _d2 = 0;
        _d3 = 0;
        _conf = ndc;
    }

    @Override
    public int[] shape() {
        return new int[]{ _s1, _s2, _s3 };
    }

    @Override
    public void increment() {
        _d1++;
        if ( _d1 == _s1 ) {
            _d1 = 0;
            _d2++;
            if ( _d2 == _s2 ) {
                _d2 = 0;
                _d3++;
            }
        }
    }

    @Override
    public void decrement() {
        if ( _d1 == 0 ) {
            _d1 = _s1 - 1;
            if ( _d2 == 0 ) {
                _d3--;
            } else _d2--;
        }
    }

    @Override
    public int i() {
        return _conf.i_of_idx( _d1, _d2, _d3 );
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

    @Override
    public int rank() {
        return 3;
    }


}