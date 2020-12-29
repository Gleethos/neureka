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
            // idxmap, will also always be 1
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
        return new int[]{_shape};
    }

    @Override
    public int shape( int i ) {
        return _shape;
    }

    @Override
    public int[] idxmap() {
        return new int[]{1};
    }

    @Override
    public int idxmap( int i ) {
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
        return new int[]{_offset};
    }

    @Override
    public int offset( int i ) {
        return _offset;
    }


    @Override
    public int i_of_i( int i ) {
        return i + _offset;
    }

    @Override
    public int[] idx_of_i( int i ) {
        return new int[]{i};
    }

    @Override
    public int i_of_idx( int[] idx ) {
        return idx[ 0 ] + _offset;
    }


}
