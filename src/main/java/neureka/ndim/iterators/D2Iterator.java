package neureka.ndim.iterators;

import neureka.ndim.config.types.D2C;

public class D2Iterator extends AbstractNDIterator
{
    private int _d1;
    private int _d2;
    private final int _s1;
    private final int _s2;
    private final D2C _conf;

    public D2Iterator( D2C ndc ) {
        _s1 = ndc.shape( 0 );
        _s2 = ndc.shape( 1 );
        _d1 = 0;
        _d2 = 0;
        _conf = ndc;
    }

    @Override
    public int[] shape() {
        return new int[]{ _s1, _s2 };
    }

    @Override
    public void increment() {
        _d2++;
        if ( _d2 == _s2 ) {
            _d2 = 0;
            _d1++;
        }
    }

    @Override
    public void decrement() {
        if ( _d2 == 0 ) {
            _d1--;
            _d2 = _s2 - 1;
        }
    }

    @Override
    public int i() {
        return _conf.i_of_idx( _d1, _d2 );
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
    public void set( int[] idx ) {
        _d1 = idx[0];
        _d2 = idx[1];
    }

    @Override
    public int rank() {
        return 2;
    }



}
