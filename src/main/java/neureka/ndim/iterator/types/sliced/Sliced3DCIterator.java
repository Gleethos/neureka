package neureka.ndim.iterator.types.sliced;

import neureka.ndim.config.types.sliced.Sliced3DConfiguration;
import neureka.ndim.iterator.NDIterator;

public final class Sliced3DCIterator extends Sliced3DConfiguration implements NDIterator
{
    private int _d1 = 0;
    private int _d2 = 0;
    private int _d3 = 0;

    public Sliced3DCIterator(Sliced3DConfiguration ndc) {
        super( ndc.shape(), ndc.translation(), ndc.indicesMap(), ndc.spread(), ndc.offset() );
    }


    /** {@inheritDoc} */
    @Override public final void increment() {
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

    /** {@inheritDoc} */
    @Override public final void decrement() {
        _d3--;
        if ( _d3 == -1 ) {
            _d3 = _shape3 - 1;
            _d2--;
            if ( _d2 == -1 ) {
                _d2 = _shape2 - 1;
                _d1--;
            }
        }
    }

    /** {@inheritDoc} */
    @Override public final int i() { return this.indexOfIndices( _d1, _d2, _d3 ); }

    /** {@inheritDoc} */
    @Override public final int get( int axis ) { return ( axis == 0 ? _d1 : ( axis == 1 ? _d2 : _d3 ) ); }

    /** {@inheritDoc} */
    @Override public final int[] get() { return new int[]{ _d1, _d2, _d3 }; }

    /** {@inheritDoc} */
    @Override public final void set( int axis, int position ) {
        if ( axis == 0 ) _d1 = position;
        else if ( axis == 1 ) _d2 = position;
        else _d3 = position;
    }

    /** {@inheritDoc} */
    @Override public final void set( int[] indices ) {
        _d1 = indices[0];
        _d2 = indices[1];
        _d3 = indices[2];
    }

}