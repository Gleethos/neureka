package neureka.ndim.config.types.simple;

import neureka.ndim.config.types.D1C;


public class Simple1DConfiguration extends D1C //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int _stride_and_indicesMap;


    public static Simple1DConfiguration construct(
        int[] shape,
        int[] strides
    ) {
        return _cached( new Simple1DConfiguration(shape[ 0 ], strides[ 0 ]) );
    }

    protected Simple1DConfiguration(
        int shape,
        int strides
    ) {
        _shape = shape;
        _stride_and_indicesMap = strides;
    }

    /** {@inheritDoc} */
    @Override public final int rank() { return 1; }

    /** {@inheritDoc} */
    @Override public final int[] shape() { return new int[]{_shape}; }

    /** {@inheritDoc} */
    @Override public final int shape( int i ) { return _shape; }

    /** {@inheritDoc} */
    @Override public final int[] indicesMap() { return new int[]{_stride_and_indicesMap}; }

    /** {@inheritDoc} */
    @Override public final int indicesMap( int i ) { return _stride_and_indicesMap; }

    /** {@inheritDoc} */
    @Override public final int[] strides() { return new int[]{_stride_and_indicesMap}; }

    /** {@inheritDoc} */
    @Override public final int strides(int i ) { return _stride_and_indicesMap; }

    /** {@inheritDoc} */
    @Override public final int[] spread() { return new int[]{1}; }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public final int[] offset() { return new int[]{0}; }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) { return index; }

    /** {@inheritDoc} */
    @Override public final int[] indicesOfIndex( int index ) { return new int[]{index / _stride_and_indicesMap}; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int[] indices ) { return indices[ 0 ] * _stride_and_indicesMap; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int d1 ) { return d1 * _stride_and_indicesMap; }

}
