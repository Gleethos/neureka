package neureka.ndim.config.types.permuted;

import neureka.ndim.config.types.D1C;

public class Permuted1DConfiguration extends D1C {
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int _stride;
    /**
     *  The mapping of the indices array.
     */
    private final int _indicesMap; // Maps index integer to array like translation. Used to avoid distortion when slicing!


    public static Permuted1DConfiguration construct(
            int[] shape,
            int[] strides,
            int[] indicesMap
    ) {
        return _cached( new Permuted1DConfiguration(shape[ 0 ], strides[ 0 ],  indicesMap[ 0 ]) );
    }

    protected Permuted1DConfiguration(
            int shape,
            int stride,
            int indicesMap
    ) {
        _shape      = shape;
        _stride     = stride;
        _indicesMap = indicesMap;
        assert stride != 0;
        assert indicesMap != 0;
    }

    /** {@inheritDoc} */
    @Override public final int rank() { return 1; }

    /** {@inheritDoc} */
    @Override public final int[] shape() { return new int[]{_shape}; }

    /** {@inheritDoc} */
    @Override public final int shape( int i ) { return _shape; }

    /** {@inheritDoc} */
    @Override public final int[] indicesMap() { return new int[]{_indicesMap}; }

    /** {@inheritDoc} */
    @Override public final int indicesMap( int i ) { return _indicesMap; }

    /** {@inheritDoc} */
    @Override public final int[] strides() { return new int[]{_stride}; }

    /** {@inheritDoc} */
    @Override public final int strides(int i ) { return _stride; }

    /** {@inheritDoc} */
    @Override public final int[] spread() { return new int[]{1}; }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public final int[] offset() { return new int[]{0}; }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) { return ( index / _indicesMap ) * _stride; }

    /** {@inheritDoc} */
    @Override public final int[] indicesOfIndex( int index ) { return new int[]{index / _indicesMap}; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int[] indices ) { return indices[ 0 ] * _stride; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int d1 ) { return d1 * _stride; }

}
