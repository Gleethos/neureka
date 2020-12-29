package neureka.ndim.config.types.simple;

import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.D1C;


public class SimpleD1Configuration extends D1C //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape;
    /**
     *  The translation from a shape index (idx) to the index of the underlying data array.
     */
    private final int _translation_and_idxmap;


    public static NDConfiguration construct(
            int[] shape,
            int[] translation
    ) {
        return _cached(new SimpleD1Configuration(shape[ 0 ], translation[ 0 ]));
    }

    protected SimpleD1Configuration(
            int shape,
            int translation
    ) {
        _shape = shape;
        _translation_and_idxmap = translation;
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
        return new int[]{_translation_and_idxmap};
    }

    @Override
    public int idxmap( int i ) {
        return _translation_and_idxmap;
    }

    @Override
    public int[] translation() {
        return new int[]{_translation_and_idxmap};
    }

    @Override
    public int translation( int i ) {
        return _translation_and_idxmap;
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
    public int i_of_i( int i ) {
        return i;
    }

    @Override
    public int[] idx_of_i( int i ) {
        return new int[]{i / _translation_and_idxmap};
    }

    @Override
    public int i_of_idx( int[] idx ) {
        return idx[ 0 ] * _translation_and_idxmap;
    }

    @Override
    public int i_of_idx( int d1 ) {
        return d1 * _translation_and_idxmap;
    }

}
