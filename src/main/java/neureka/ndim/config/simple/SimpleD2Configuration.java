package neureka.ndim.config.simple;

import neureka.Neureka;
import neureka.ndim.config.AbstractArrayBasedNDC;
import neureka.ndim.config.NDConfiguration;


public final class SimpleD2Configuration extends AbstractArrayBasedNDC //:= IMMUTABLE
{
    private SimpleD2Configuration(
            int[] shape,
            int[] translation
    ) {
        _shape1 = shape[0];
        _shape2 = shape[1];
        _translation1 = translation[0];
        _translation2 = translation[1];
    }

    public static NDConfiguration construct(
            int[] shape,
            int[] translation
    ){
        return _cached(new SimpleD2Configuration(shape, translation));
    }

    /**
     *  The shape of the NDArray.
     */
    private final int _shape1;
    private final int _shape2;
    /**
     *  The translation from a shape index (idx) to the index of the underlying data array.
     */
    private final int _translation1;
    private final int _translation2;


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
        return new int[]{_translation1, _translation2};
    }

    @Override
    public int idxmap(int i) {
        return 1;
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
        return new int[]{1, 1};
    }

    @Override
    public int spread(int i) {
        return 1;
    }

    @Override
    public int[] offset() {
        return new int[]{0,0};
    }

    @Override
    public int offset(int i) {
        return 0;
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public int i_of_i(int i) {
        if (Neureka.instance().settings().indexing().isUsingLegacyIndexing()){
            return ((i%_translation2) / _translation1) * _translation1 +
                    (i / _translation2) * _translation2;
        } else {
            return (i / _translation1) * _translation1 +
                    ((i%_translation1) / _translation2) * _translation2;
        }
    }

    @Override
    public int[] idx_of_i(int i) {
        int[] idx = new int[2];
        if (Neureka.instance().settings().indexing().isUsingLegacyIndexing()){
            idx[1] += i / _translation2;
            i %= _translation2;
            idx[0] += i / _translation1;
        } else {
            idx[0] += i / _translation1;
            i %= _translation1;
            idx[1] += i / _translation2;
        }
        return idx;
    }

    @Override
    public int i_of_idx(int[] idx) {
        int i = 0;
        i += idx[0] * _translation1;
        i += idx[1] * _translation2;
        return i;
    }

}