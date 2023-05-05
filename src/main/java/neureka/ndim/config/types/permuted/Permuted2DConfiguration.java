package neureka.ndim.config.types.permuted;

import neureka.ndim.config.types.D2C;

public class Permuted2DConfiguration extends D2C
{
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape1;
    protected final int _shape2;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int _stride1;
    private final int _stride2;
    /**
     *  The mapping for the indices array.
     */
    private final int _indicesMap1;
    private final int _indicesMap2; // Maps index integer to array like translation. Used to avoid distortion when slicing!


    protected Permuted2DConfiguration(
            int[] shape,
            int[] strides,
            int[] indicesMap
    ) {
        _shape1 = shape[ 0 ];
        _shape2 = shape[ 1 ];
        _stride1 = strides[ 0 ];
        _stride2 = strides[ 1 ];
        _indicesMap1 = indicesMap[ 0 ];
        _indicesMap2 = indicesMap[ 1 ];
    }

    public static Permuted2DConfiguration construct(
            int[] shape,
            int[] strides,
            int[] indicesMap
    ) {
        return _cached( new Permuted2DConfiguration(shape, strides, indicesMap) );
    }

    /** {@inheritDoc} */
    @Override public final int rank() { return 2; }

    /** {@inheritDoc} */
    @Override public final int[] shape() { return new int[]{_shape1, _shape2}; }

    /** {@inheritDoc} */
    @Override public final int shape( int i ) { return ( i==0 ? _shape1 : _shape2 ); }

    /** {@inheritDoc} */
    @Override public final int[] indicesMap() { return new int[]{_indicesMap1, _indicesMap2}; }

    /** {@inheritDoc} */
    @Override public final int indicesMap( int i ) { return ( i==0 ? _indicesMap1 : _indicesMap2 ); }

    /** {@inheritDoc} */
    @Override public final int[] strides() { return new int[]{_stride1, _stride2}; }

    /** {@inheritDoc} */
    @Override public final int strides(int i ) { return ( i==0 ? _stride1 : _stride2); }

    /** {@inheritDoc} */
    @Override public final int[] spread() { return new int[]{1, 1}; }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public final int[] offset() { return new int[]{0, 0}; }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) {
        return (index / _indicesMap1) * _stride1 +
                ((index %_indicesMap1) / _indicesMap2) * _stride2;
    }

    /** {@inheritDoc} */
    @Override public final int[] indicesOfIndex( int index ) {
        int[] indices = new int[ 2 ];
        indices[ 0 ] += index / _indicesMap1;
        index %= _indicesMap1;
        indices[ 1 ] += index / _indicesMap2;
        return indices;
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int[] indices ) {
        int i = 0;
        i += indices[ 0 ]* _stride1;
        i += indices[ 1 ]* _stride2;
        return i;
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int d1, int d2 ) {
        int i = 0;
        i += d1 * _stride1;
        i += d2 * _stride2;
        return i;
    }
}
