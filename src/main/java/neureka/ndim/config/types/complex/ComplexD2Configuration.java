package neureka.ndim.config.types.complex;

import neureka.Neureka;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.D2C;

public class ComplexD2Configuration extends D2C //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape1;
    protected final int _shape2;
    /**
     *  The translation from a shape index (idx) to the index of the underlying data array.
     */
    private final int _translation1;
    private final int _translation2;
    /**
     *  The mapping of idx array.
     */
    private final int _idxmap1;
    private final int _idxmap2; // Maps index integer to array like translation. Used to avoid distortion when slicing!
    /**
     *  Produces the strides of a tensor subset / slice
     */
    private final int _spread1;
    private final int _spread2;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int _offset1;
    private final int _offset2;
    /**
     *  The value of this tensor. Usually a array of type double[] or float[].
     */

    protected ComplexD2Configuration(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ) {
        _shape1 = shape[ 0 ];
        _shape2 = shape[ 1 ];
        _translation1 = translation[ 0 ];
        _translation2 = translation[ 1 ];
        _idxmap1 = idxmap[ 0 ];
        _idxmap2 = idxmap[ 1 ];
        _spread1 = spread[ 0 ];
        _spread2 = spread[ 1 ];
        _offset1 = offset[ 0 ];
        _offset2 = offset[ 1 ];
    }

    public static NDConfiguration construct(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ){
        return _cached(new ComplexD2Configuration(shape, translation, idxmap, spread, offset));
    }

    @Override
    public int rank() {
        return 2;
    }

    @Override
    public int[] shape() {
        return new int[]{_shape1, _shape2};
    }

    @Override
    public int shape(int i) {
        return (i==0)?_shape1:_shape2;
    }

    @Override
    public int[] idxmap() {
        return new int[]{_idxmap1, _idxmap2};
    }

    @Override
    public int idxmap(int i) {
        return (i==0)?_idxmap1:_idxmap2;
    }

    @Override
    public int[] translation() {
        return new int[]{_translation1, _translation2};
    }

    @Override
    public int translation(int i) {
        return (i==0)?_translation1:_translation2;
    }

    @Override
    public int[] spread() {
        return new int[]{_spread1, _spread2};
    }

    @Override
    public int spread(int i) {
        return (i==0)?_spread1:_spread2;
    }

    @Override
    public int[] offset() {
        return new int[]{_offset1, _offset2};
    }

    @Override
    public int offset(int i) {
        return (i==0)?_offset1:_offset2;
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public int i_of_i(int i) {
        if (Neureka.instance().settings().indexing().isUsingLegacyIndexing()){
            return (((i%_idxmap2) / _idxmap1) * _spread1 + _offset1) * _translation1 +
                    ((i / _idxmap2) * _spread2 + _offset2) * _translation2;
        } else {
            return ((i / _idxmap1) * _spread1 + _offset1) * _translation1 +
                    (((i%_idxmap1) / _idxmap2) * _spread2 + _offset2) * _translation2;
        }
    }

    @Override
    public int[] idx_of_i(int i) {
        int[] idx = new int[ 2 ];
        if (Neureka.instance().settings().indexing().isUsingLegacyIndexing()){
            idx[ 1 ] += i / _idxmap2;
            i %= _idxmap2;
            idx[ 0 ] += i / _idxmap1;
        } else {
            idx[ 0 ] += i / _idxmap1;
            i %= _idxmap1;
            idx[ 1 ] += i / _idxmap2;
        }
        return idx;
    }

    @Override
    public int i_of_idx(int[] idx) {
        int i = 0;
        i += (idx[ 0 ] * _spread1 + _offset1) * _translation1;
        i += (idx[ 1 ] * _spread2 + _offset2) * _translation2;
        return i;
    }

    @Override
    public int i_of_idx(int d1, int d2) {
        int i = 0;
        i += (d1 * _spread1 + _offset1) * _translation1;
        i += (d2 * _spread2 + _offset2) * _translation2;
        return i;
    }
}