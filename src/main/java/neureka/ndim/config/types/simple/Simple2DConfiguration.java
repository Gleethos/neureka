package neureka.ndim.config.types.simple;

import neureka.ndim.config.types.D2C;


public class Simple2DConfiguration extends D2C //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape1;
    protected final int _shape2;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int _translation1;
    private final int _translation2;


    protected Simple2DConfiguration(
            int[] shape,
            int[] translation
    ) {
        _shape1 = shape[ 0 ];
        _shape2 = shape[ 1 ];
        _translation1 = translation[ 0 ];
        _translation2 = translation[ 1 ];
    }

    public static Simple2DConfiguration construct(
            int[] shape,
            int[] translation
    ) {
        return _cached( new Simple2DConfiguration(shape, translation) );
    }


    /** {@inheritDoc} */
    @Override public final int rank() { return 2; }

    /** {@inheritDoc} */
    @Override public final int[] shape() { return new int[]{_shape1, _shape2}; }

    /** {@inheritDoc} */
    @Override public final int shape( int i ) { return ( i==0 ? _shape1 : _shape2 ); }

    /** {@inheritDoc} */
    @Override public final int[] indicesMap() { return new int[]{_translation1, _translation2}; }

    /** {@inheritDoc} */
    @Override public final int indicesMap( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public final int[] translation() { return new int[]{_translation1, _translation2}; }

    /** {@inheritDoc} */
    @Override public final int translation( int i ) { return ( i==0 ? _translation1 : _translation2 ); }

    /** {@inheritDoc} */
    @Override public final int[] spread() { return new int[]{1, 1}; }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public final int[] offset() { return new int[]{0,0}; }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) {
        return (index / _translation1) * _translation1 +
                ((index %_translation1) / _translation2) * _translation2;
    }

    /** {@inheritDoc} */
    @Override public final int[] indicesOfIndex( int index ) {
        int[] indices = new int[ 2 ];
        indices[ 0 ] += index / _translation1;
        index %= _translation1;
        indices[ 1 ] += index / _translation2;
        return indices;
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int[] indices ) {
        int i = 0;
        i += indices[ 0 ] * _translation1;
        i += indices[ 1 ] * _translation2;
        return i;
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int d1, int d2 ) {
        int i = 0;
        i += d1 * _translation1;
        i += d2 * _translation2;
        return i;
    }
}