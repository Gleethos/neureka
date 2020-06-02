package neureka.ndim.config.simple;

import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;

public class SimpleScalarConfiguration extends AbstractNDC
{
    public static NDConfiguration construct(){
        return _cached(new SimpleScalarConfiguration());
    }

    private SimpleScalarConfiguration() {}

    @Override
    public int rank() {
        return 1;
    }

    @Override
    public int[] shape() {
        return new int[]{1};
    }

    @Override
    public int shape(int i) {
        return 1;
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
        return new int[]{0};
    }

    @Override
    public int offset(int i) {
        return 0;
    }

    @Override
    public int i_of_i(int i) {
        return i;
    }

    @Override
    public int[] idx_of_i(int i) {
        return new int[]{i};
    }

    @Override
    public int i_of_idx(int[] idx) {
        return idx[0];
    }

}