package neureka.ndim.config.types.sliced;

import neureka.ndim.config.types.D1C;

public class Sliced1DConfiguration extends D1C //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int _translation;
    /**
     *  The mapping of the indices array.
     */
    private final int _indicesMap; // Maps index integer to array like translation. Used to avoid distortion when slicing!
    /**
     *  Produces the strides of a tensor subset / slice
     */
    private final int _spread;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int _offset;


    public static Sliced1DConfiguration construct(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        return _cached( new Sliced1DConfiguration(shape[ 0 ], translation[ 0 ],  indicesMap[ 0 ], spread[ 0 ], offset[ 0 ]) );
    }

    protected Sliced1DConfiguration(
            int shape,
            int translation,
            int indicesMap,
            int spread,
            int offset
    ) {
        _shape = shape;
        _translation = translation;
        _indicesMap = indicesMap;
        _spread = spread;
        _offset = offset;
        assert translation != 0;
        assert indicesMap != 0;
        assert spread != 0;
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
        return new int[]{_indicesMap};
    }

    @Override
    public int indicesMap(int i ) {
        return _indicesMap;
    }

    @Override
    public int[] translation() {
        return new int[]{_translation};
    }

    @Override
    public int translation( int i ) {
        return _translation;
    }

    @Override
    public int[] spread() {
        return new int[]{_spread};
    }

    @Override
    public int spread( int i ) {
        return _spread;
    }

    @Override
    public int[] offset() {
        return new int[]{_offset};
    }

    @Override
    public int offset( int i ) {
        return _offset;
    }

    @Override
    public int indexOfIndex(int index) {
        return ((index / _indicesMap) * _spread + _offset) * _translation;
    }

    @Override
    public int[] indicesOfIndex(int index) {
        return new int[]{index / _indicesMap};
    }

    @Override
    public int indexOfIndices(int[] indices) {
        return (indices[ 0 ] * _spread + _offset) * _translation;
    }

    @Override
    public int indexOfIndices(int d1 ) {
        return (d1 * _spread + _offset) * _translation;
    }

}
