package neureka.ndim.config.types.simple;

import neureka.ndim.config.AbstractNDC;

import java.util.Arrays;

public final class SimpleNDConfiguration extends AbstractNDC //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    protected final int[] _shape;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int[] _translation_and_indicesMap;


    protected SimpleNDConfiguration(
            int[] shape, int[] translation
    ) {
        _shape = _cacheArray( shape );
        _translation_and_indicesMap = _cacheArray( translation );
    }

    public static SimpleNDConfiguration construct(
            int[] shape,
            int[] translation
    ) {
        return _cached( new SimpleNDConfiguration( shape, translation ) );
    }

    /** {@inheritDoc} */
    @Override public int rank() { return _shape.length; }

    /** {@inheritDoc} */
    @Override public int[] shape() { return _shape; }

    /** {@inheritDoc} */
    @Override public int shape( int i ) { return _shape[ i ]; }

    /** {@inheritDoc} */
    @Override public int[] indicesMap() { return _translation_and_indicesMap; }

    /** {@inheritDoc} */
    @Override public int indicesMap( int i ) { return _translation_and_indicesMap[ i ]; }

    /** {@inheritDoc} */
    @Override public int[] translation() { return _translation_and_indicesMap; }

    /** {@inheritDoc} */
    @Override public int translation( int i ) { return _translation_and_indicesMap[ i ]; }

    /** {@inheritDoc} */
    @Override public int[] spread() {
        int[] newSpread = new int[ _shape.length ];
        Arrays.fill( newSpread, 1 );
        return newSpread;
    }

    /** {@inheritDoc} */
    @Override public int spread( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public int[] offset() { return new int[ _shape.length ]; }

    /** {@inheritDoc} */
    @Override public int offset( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public int indexOfIndex( int index ) { return indexOfIndices( indicesOfIndex( index ) ); }

    /** {@inheritDoc} */
    @Override
    public int[] indicesOfIndex( int index ) { int[] indices = new int[ _shape.length ];
        for ( int ii=0; ii<rank(); ii++ ) {
            indices[ ii ] += index / _translation_and_indicesMap[ ii ];
            index %= _translation_and_indicesMap[ ii ];
        }
        return indices;
    }

    /** {@inheritDoc} */
    @Override
    public int indexOfIndices( int[] indices ) {
        int i = 0;
        for ( int ii = 0; ii < _shape.length; ii++ ) i += indices[ ii ] * _translation_and_indicesMap[ ii ];
        return i;
    }


}
