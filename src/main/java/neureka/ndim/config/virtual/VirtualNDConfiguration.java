package neureka.ndim.config.virtual;

import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.simple.SimpleD2Configuration;

public class VirtualNDConfiguration extends AbstractNDC
{

    private final int[] _shape;

    private VirtualNDConfiguration(int[] shape){
        _shape = _cacheArray( shape );
    }

    public static NDConfiguration construct(
            int[] shape
    ){
        return _cached( new VirtualNDConfiguration( shape ) );
    }

    @Override
    public int rank() {
        return _shape.length;
    }

    @Override
    public int[] shape() {
        return _shape;
    }

    @Override
    public int shape(int i) {
        return _shape[0];
    }

    @Override
    public int[] idxmap() {
        return new int[rank()];
    }

    @Override
    public int idxmap(int i) {
        return 0;
    }

    @Override
    public int[] translation() {
        return new int[rank()];
    }

    @Override
    public int translation(int i) {
        return 0;
    }

    @Override
    public int[] spread() {
        return new int[rank()];
    }

    @Override
    public int spread(int i) {
        return 0;
    }

    @Override
    public int[] offset() {
        return new int[rank()];
    }

    @Override
    public int offset(int i) {
        return 0;
    }

    @Override
    public int i_of_i(int i) {
        return 0;
    }

    @Override
    public int[] idx_of_i(int i) {
        return new int[rank()];
    }

    @Override
    public int i_of_idx(int[] idx) {
        return 0;
    }

}
