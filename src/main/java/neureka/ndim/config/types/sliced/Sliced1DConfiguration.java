package neureka.ndim.config.types.sliced;

import neureka.ndim.config.types.D1C;

public class Sliced1DConfiguration extends D1C //:= IMMUTABLE
{
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
    /**
     *  Produces the steps of a tensor subset / slice
     */
    private final int _spread;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int _offset;


    public static Sliced1DConfiguration construct(
        int[] shape,
        int[] strides,
        int[] indicesMap,
        int[] spread,
        int[] offset
    ) {
        return _cached( new Sliced1DConfiguration(shape[ 0 ], strides[ 0 ],  indicesMap[ 0 ], spread[ 0 ], offset[ 0 ]) );
    }

    protected Sliced1DConfiguration(
        int shape,
        int stride,
        int indicesMap,
        int spread,
        int offset
    ) {
        _shape      = shape;
        _stride     = stride;
        _indicesMap = indicesMap;
        _spread     = spread;
        _offset     = offset;
        assert stride != 0;
        assert indicesMap != 0;
        assert spread != 0;
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
    @Override public final int indicesMap(int i ) { return _indicesMap; }

    /** {@inheritDoc} */
    @Override public final int[] strides() { return new int[]{_stride}; }

    /** {@inheritDoc} */
    @Override public final int strides(int i ) { return _stride; }

    /** {@inheritDoc} */
    @Override public final int[] spread() { return new int[]{_spread}; }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return _spread; }

    /** {@inheritDoc} */
    @Override public final int[] offset() { return new int[]{_offset}; }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return _offset; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) { return ( ( index / _indicesMap ) * _spread + _offset ) * _stride; }

    /** {@inheritDoc} */
    @Override public final int[] indicesOfIndex( int index ) { return new int[]{index / _indicesMap}; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int[] indices ) { return ( indices[ 0 ] * _spread + _offset ) * _stride; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int d1 ) { return ( d1 * _spread + _offset ) * _stride; }

}
