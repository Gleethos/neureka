package neureka.ndim.config;

import neureka.ndim.NDConfiguration;

public class D1Configuration implements NDConfiguration
{
    /**
     *  The shape of the NDArray.
     */
    private int _shape;
    /**
     *  The translation from a shape index (idx) to the index of the underlying data array.
     */
    private int _translation;
    /**
     *  The mapping of idx array.
     */
    private int _idxmap; // Used to avoid distortion when reshaping!
    /**
     *  Produces the strides of a tensor subset / slice
     */
    private int _spread;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private int _offset;
    /**
     *  The value of this tensor. Usually a array of type double[] or float[].
     */

    public static D1Configuration construct(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ){
        return new D1Configuration(shape[0], translation[0],  idxmap[0], spread[0], offset[0]);
    }

    private D1Configuration(
            int shape,
            int translation,
            int idxmap,
            int spread,
            int offset
    ) {
        _shape = shape;
        _translation = translation;
        _idxmap = idxmap;
        _spread = spread;
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
        return new int[]{_idxmap};
    }

    @Override
    public int idxmap(int i) {
        return _idxmap;
    }

    @Override
    public int[] translation() {
        return new int[]{_translation};
    }

    @Override
    public int translation(int i) {
        return _translation;
    }

    @Override
    public int[] spread() {
        return new int[]{_spread};
    }

    @Override
    public int spread(int i) {
        return _spread;
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
        return ((i / _idxmap) * _spread + _offset) * _translation;
    }

    @Override
    public int[] idx_of_i(int i) {
        return new int[]{i / _idxmap};
    }
    @Override
    public int i_of_idx(int[] idx) {
        return (idx[0] * _spread + _offset) * _translation;
    }


}
