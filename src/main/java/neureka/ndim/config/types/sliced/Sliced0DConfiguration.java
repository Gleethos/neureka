package neureka.ndim.config.types.sliced;

import neureka.ndim.config.AbstractNDC;

public final class Sliced0DConfiguration extends AbstractNDC //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    private final int _shape;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int _offset;


    public static Sliced0DConfiguration construct(
            int[] shape,
            // translation, will always be 1
            // indicesMap, will also always be 1
            // spread, does not matter!
            int[] offset
    ) {
        return _cached( new Sliced0DConfiguration(shape[ 0 ], offset[ 0 ]) );
    }

    protected Sliced0DConfiguration(
            int shape,
            int offset
    ) {
        _shape = shape;
        _offset = offset;
    }

    /** {@inheritDoc} */
    @Override public int rank() { return 1; }

    /** {@inheritDoc} */
    @Override public int[] shape() { return new int[]{ _shape }; }

    /** {@inheritDoc} */
    @Override public int shape( int i ) { return _shape; }

    /** {@inheritDoc} */
    @Override public int[] indicesMap() { return new int[]{1}; }

    @Override
    public int indicesMap(int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public int[] translation() { return new int[]{1}; }

    /** {@inheritDoc} */
    @Override public int translation( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public int[] spread() { return new int[]{ 1 }; }

    /** {@inheritDoc} */
    @Override public int spread( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public int[] offset() { return new int[]{ _offset }; }

    /** {@inheritDoc} */
    @Override public int offset( int i ) { return _offset; }

    /** {@inheritDoc} */
    @Override public int indexOfIndex(int index) {
        return index + _offset;
    }

    /** {@inheritDoc} */
    @Override public int[] indicesOfIndex(int index) { return new int[]{ index }; }

    /** {@inheritDoc} */
    @Override public int indexOfIndices(int[] indices) { return indices[ 0 ] + _offset; }


}
