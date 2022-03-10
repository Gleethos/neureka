package neureka.ndim.config.types.simple;

import neureka.ndim.config.AbstractNDC;

public final class Simple0DConfiguration extends AbstractNDC //:= IMMUTABLE
{
    public static Simple0DConfiguration construct() {
        return _cached( new Simple0DConfiguration() );
    }

    private Simple0DConfiguration() {}

    @Override
    public final int rank() {
        return 1;
    }

    @Override
    public final int[] shape() {
        return new int[]{1};
    }

    @Override
    public final int shape( int i ) {
        return 1;
    }

    @Override
    public final int[] indicesMap() {
        return new int[]{1};
    }

    @Override
    public final int indicesMap(int i ) {
        return 1;
    }

    @Override
    public final int[] translation() {
        return new int[]{1};
    }

    @Override
    public final int translation( int i ) {
        return 1;
    }

    @Override
    public final int[] spread() {
        return new int[]{1};
    }

    @Override
    public final int spread( int i ) {
        return 1;
    }

    @Override
    public final int[] offset() {
        return new int[]{0};
    }

    @Override
    public final int offset( int i ) {
        return 0;
    }

    @Override
    public final int indexOfIndex(int index) {
        return index;
    }

    @Override
    public final int[] indicesOfIndex(int index) {
        return new int[]{index};
    }

    @Override
    public final int indexOfIndices(int[] indices) {
        return indices[ 0 ];
    }

}