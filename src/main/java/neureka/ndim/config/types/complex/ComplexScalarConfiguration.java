package neureka.ndim.config.types.complex;

import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;

public final class ComplexScalarConfiguration extends AbstractNDC //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int _offset;


    public static NDConfiguration construct(
            int[] shape,
            // translation, will always be 1
            // indicesMap, will also always be 1
            // spread, does not matter!
            int[] offset
    ) {
        return _cached(new ComplexScalarConfiguration(shape[ 0 ], offset[ 0 ]));
    }

    protected ComplexScalarConfiguration(
            int shape,
            int offset
    ) {
        _shape = shape;
        _offset = offset;
    }

    @Override
    public int rank() {
        return 1;
    }

    @Override
    public int[] shape() {
        return new int[]{ _shape };
    }

    @Override
    public int shape( int i ) {
        return _shape;
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
        return new int[]{ 1 };
    }

    @Override
    public int spread( int i ) {
        return 1;
    }

    @Override
    public int[] offset() {
        return new int[]{ _offset };
    }

    @Override
    public int offset( int i ) {
        return _offset;
    }


    @Override
    public int indexOfIndex(int index) {
        return index + _offset;
    }

    @Override
    public int[] indicesOfIndex(int index) {
        return new int[]{ index };
    }

    @Override
    public int indexOfIndices(int[] indices) {
        return indices[ 0 ] + _offset;
    }


}
