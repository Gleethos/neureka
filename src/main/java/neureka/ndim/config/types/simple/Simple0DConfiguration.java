package neureka.ndim.config.types.simple;

import neureka.ndim.config.AbstractNDC;

public final class Simple0DConfiguration extends AbstractNDC //:= IMMUTABLE
{
    public static Simple0DConfiguration construct() {
        return _cached( new Simple0DConfiguration() );
    }

    private Simple0DConfiguration() {}

    @Override
    public int rank() {
        return 1;
    }

    @Override
    public int[] shape() {
        return new int[]{1};
    }

    @Override
    public int shape( int i ) {
        return 1;
    }

    @Override
    public int[] indicesMap() {
        return new int[]{1};
    }

    @Override
    public int indicesMap(int i ) {
        return 1;
    }

    @Override
    public int[] translation() {
        return new int[]{1};
    }

    @Override
    public int translation( int i ) {
        return 1;
    }

    @Override
    public int[] spread() {
        return new int[]{1};
    }

    @Override
    public int spread( int i ) {
        return 1;
    }

    @Override
    public int[] offset() {
        return new int[]{0};
    }

    @Override
    public int offset( int i ) {
        return 0;
    }

    @Override
    public int indexOfIndex(int index) {
        return index;
    }

    @Override
    public int[] indicesOfIndex(int index) {
        return new int[]{index};
    }

    @Override
    public int indexOfIndices(int[] indices) {
        return indices[ 0 ];
    }

}