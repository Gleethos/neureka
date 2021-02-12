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
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int _translation_and_indicesMap;


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
        _translation_and_indicesMap = translation;
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
    public int[] indicesMap() {
        return new int[]{_translation_and_indicesMap};
    }

    @Override
    public int indicesMap(int i ) {
        return _translation_and_indicesMap;
    }

    @Override
    public int[] translation() {
        return new int[]{_translation_and_indicesMap};
    }

    @Override
    public int translation( int i ) {
        return _translation_and_indicesMap;
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
    public int indexOfIndex(int index) {
        return index;
    }

    @Override
    public int[] indicesOfIndex(int index) {
        return new int[]{index / _translation_and_indicesMap};
    }

    @Override
    public int indexOfIndices(int[] indices) {
        return indices[ 0 ] * _translation_and_indicesMap;
    }

    @Override
    public int indexOfIndices(int d1 ) {
        return d1 * _translation_and_indicesMap;
    }

}
