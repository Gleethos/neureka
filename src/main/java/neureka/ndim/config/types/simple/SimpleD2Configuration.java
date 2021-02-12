package neureka.ndim.config.types.simple;

import neureka.Neureka;
import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.types.D2C;


public class SimpleD2Configuration extends D2C //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape1;
    protected final int _shape2;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int _translation1;
    private final int _translation2;

    protected SimpleD2Configuration(
            int[] shape,
            int[] translation
    ) {
        _shape1 = shape[ 0 ];
        _shape2 = shape[ 1 ];
        _translation1 = translation[ 0 ];
        _translation2 = translation[ 1 ];
    }

    public static NDConfiguration construct(
            int[] shape,
            int[] translation
    ) {
        return _cached(new SimpleD2Configuration(shape, translation));
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
    public int shape( int i ) {
        return (i==0)?_shape1:_shape2;
    }

    @Override
    public int[] indicesMap() {
        return new int[]{_translation1, _translation2};
    }

    @Override
    public int indicesMap(int i ) {
        return 1;
    }

    @Override
    public int[] translation() {
        return new int[]{_translation1, _translation2};
    }

    @Override
    public int translation( int i ) {
        return (i==0)?_translation1:_translation2;
    }

    @Override
    public int[] spread() {
        return new int[]{1, 1};
    }

    @Override
    public int spread( int i ) {
        return 1;
    }

    @Override
    public int[] offset() {
        return new int[]{0,0};
    }

    @Override
    public int offset( int i ) {
        return 0;
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public int indexOfIndex(int index) {
        return (index / _translation1) * _translation1 +
                ((index %_translation1) / _translation2) * _translation2;
    }

    @Override
    public int[] indicesOfIndex(int index) {
        int[] indices = new int[ 2 ];
        indices[ 0 ] += index / _translation1;
        index %= _translation1;
        indices[ 1 ] += index / _translation2;
        return indices;
    }

    @Override
    public int indexOfIndices(int[] indices) {
        int i = 0;
        i += indices[ 0 ] * _translation1;
        i += indices[ 1 ] * _translation2;
        return i;
    }

    @Override
    public int indexOfIndices(int d1, int d2 ) {
        int i = 0;
        i += d1 * _translation1;
        i += d2 * _translation2;
        return i;
    }
}