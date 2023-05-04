package neureka.ndim.config.types.simple;

import neureka.ndim.config.AbstractNDC;

public final class Simple0DConfiguration extends AbstractNDC //:= IMMUTABLE
{
    public static Simple0DConfiguration construct() {
        return _cached( new Simple0DConfiguration() );
    }

    private Simple0DConfiguration() {}


    /** {@inheritDoc} */
    @Override public int rank() { return 1; }

    /** {@inheritDoc} */
    @Override public int[] shape() { return new int[]{1}; }

    /** {@inheritDoc} */
    @Override public int shape( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public int[] indicesMap() { return new int[]{1}; }

    /** {@inheritDoc} */
    @Override public int indicesMap( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public int[] strides() { return new int[]{1}; }

    /** {@inheritDoc} */
    @Override public int strides(int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public int[] spread() { return new int[]{1}; }

    /** {@inheritDoc} */
    @Override public int spread( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public int[] offset() { return new int[]{0}; }

    /** {@inheritDoc} */
    @Override public int offset( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public int indexOfIndex( int index ) { return index; }

    /** {@inheritDoc} */
    @Override public int[] indicesOfIndex( int index ) { return new int[]{index}; }

    /** {@inheritDoc} */
    @Override public int indexOfIndices(int[] indices) { return indices[ 0 ]; }

}