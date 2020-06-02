package neureka.ndim.config.complex;

import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;

public class ScalarConfiguration extends AbstractNDC
{
    /**
     *  The shape of the NDArray.
     */
    private final int _shape;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int _offset;
    /**
     *  The value of this tensor. Usually a array of type double[] or float[].
     */

    public static NDConfiguration construct(
            int[] shape,
            //int[] translation, will always be 1
            //int[] idxmap, will also always be 1
            //int[] spread, does not matter!
            int[] offset
    ) {
        return _cached(new ScalarConfiguration(shape[0], offset[0]));
    }

    private ScalarConfiguration(
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
    public int shape(int i) {
        return _shape;
    }

    @Override
    public int[] idxmap() {
        return new int[]{1};
    }

    @Override
    public int idxmap(int i) {
        return 1;
    }

    @Override
    public int[] translation() {
        return new int[]{1};
    }

    @Override
    public int translation(int i) {
        return 1;
    }

    @Override
    public int[] spread() {
        return new int[]{1};
    }

    @Override
    public int spread(int i) {
        return 1;
    }

    @Override
    public int[] offset() {
        return new int[]{_offset};
    }

    @Override
    public int offset(int i) {
        return _offset;
    }


    @Override
    public int i_of_i(int i) {
        return i + _offset;
    }

    @Override
    public int[] idx_of_i(int i) {
        return new int[]{i};
    }

    @Override
    public int i_of_idx(int[] idx) {
        return idx[0] + _offset;
    }


}
