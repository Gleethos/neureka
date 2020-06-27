package neureka.ndim.config.complex;

import neureka.Neureka;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;

public final class D3Configuration extends AbstractNDC //:= IMMUTABLE
{
    private D3Configuration(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ) {
        _shape1 = shape[0];
        _shape2 = shape[1];
        _shape3 = shape[2];
        _translation1 = translation[0];
        _translation2 = translation[1];
        _translation3 = translation[2];
        _idxmap1 = idxmap[0];
        _idxmap2 = idxmap[1];
        _idxmap3 = idxmap[2];
        _spread1 = spread[0];
        _spread2 = spread[1];
        _spread3 = spread[2];
        _offset1 = offset[0];
        _offset2 = offset[1];
        _offset3 = offset[2];
    }

    public static NDConfiguration construct(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ){
        return _cached(new D3Configuration(shape, translation, idxmap, spread, offset));
    }

    /**
     *  The shape of the NDArray.
     */
    private final int _shape1;
    private final int _shape2;
    private final int _shape3;
    /**
     *  The translation from a shape index (idx) to the index of the underlying data array.
     */
    private final int _translation1;
    private final int _translation2;
    private final int _translation3;
    /**
     *  The mapping of idx array.
     */
    private final int _idxmap1;
    private final int _idxmap2; // Used to avoid distortion when reshaping!
    private final int _idxmap3;
    /**
     *  Produces the strides of a tensor subset / slice
     */
    private final int _spread1;
    private final int _spread2;
    private final int _spread3;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int _offset1;
    private final int _offset2;
    private final int _offset3;
    /**
     *  The value of this tensor. Usually a array of type double[] or float[].
     */

    @Override
    public int rank() {
        return 3;
    }

    @Override
    public int[] shape() {
        return new int[]{_shape1, _shape2, _shape3};
    }

    @Override
    public int shape(int i) {
        return (i==0)?_shape1:(i==1)?_shape2:_shape3;
    }

    @Override
    public int[] idxmap() {
        return new int[]{_idxmap1, _idxmap2, _idxmap3};
    }

    @Override
    public int idxmap(int i) {
        return (i==0)?_idxmap1:(i==1)?_idxmap2:_idxmap3;
    }

    @Override
    public int[] translation() {
        return new int[]{_translation1, _translation2, _translation3};
    }

    @Override
    public int translation(int i) {
        return (i==0)?_translation1:(i==1)?_translation2:_translation3;
    }

    @Override
    public int[] spread() {
        return new int[]{_spread1, _spread2, _spread3};
    }

    @Override
    public int spread(int i) {
        return (i==0)?_spread1:(i==1)?_spread2:_spread3;
    }

    @Override
    public int[] offset() {
        return new int[]{_offset1, _offset2, _offset3};
    }

    @Override
    public int offset(int i) {
        return (i==0)?_offset1:(i==1)?_offset2:_offset3;
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public int i_of_i(int i){
        int idx1, idx2, idx3;
        if (Neureka.instance().settings().indexing().isUsingLegacyIndexing()){
            idx3 = i / _idxmap3;
            i %= _idxmap3;
            idx2 = i / _idxmap2;
            i %= _idxmap2;
            idx1 = i / _idxmap1;
        } else {
            idx1 = i / _idxmap1;
            i %= _idxmap1;
            idx2 = i / _idxmap2;
            i %= _idxmap2;
            idx3 = i / _idxmap3;
        }
        return (idx1 * _spread1 + _offset1) * _translation1 +
                (idx2 * _spread2 + _offset2) * _translation2 +
                (idx3 * _spread3 + _offset3) * _translation3;
    }

    @Override
    public int[] idx_of_i(int i) {
        int idx1, idx2, idx3;
        if (Neureka.instance().settings().indexing().isUsingLegacyIndexing()){
            idx3 = i / _idxmap3;
            i %= _idxmap3;
            idx2 = i / _idxmap2;
            i %= _idxmap2;
            idx1 = i / _idxmap1;
        } else {
            idx1 = i / _idxmap1;
            i %= _idxmap1;
            idx2 = i / _idxmap2;
            i %= _idxmap2;
            idx3 = i / _idxmap3;
        }
        return new int[]{idx1, idx2, idx3};
    }

    @Override
    public int i_of_idx(int[] idx) {
        return (idx[0] * _spread1 + _offset1) * _translation1 +
                    (idx[1] * _spread2 + _offset2) * _translation2 +
                        (idx[2] * _spread3 + _offset3) * _translation3;
    }

}