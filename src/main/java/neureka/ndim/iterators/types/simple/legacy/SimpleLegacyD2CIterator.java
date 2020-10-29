package neureka.ndim.iterators.types.simple.legacy;

import neureka.ndim.config.NDIterator;
import neureka.ndim.config.types.simple.SimpleD2Configuration;

public class SimpleLegacyD2CIterator extends SimpleD2Configuration implements NDIterator
{
    protected int _d1 = 0;
    protected int _d2 = 0;

    public SimpleLegacyD2CIterator( SimpleD2Configuration ndc ) {
        super( ndc.shape(), ndc.translation() );
    }

    @Override
    public void increment() {
        _d1++;
        if ( _d1 == _shape1 ) {
            _d1 = 0;
            _d2++;
        }
    }

    @Override
    public void decrement() {
        if ( _d1 == 0 ) {
            _d2--;
            _d1 = _shape1 - 1;
        }
    }

    @Override
    public int i() {
        return this.i_of_idx( _d1, _d2 );
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

}
