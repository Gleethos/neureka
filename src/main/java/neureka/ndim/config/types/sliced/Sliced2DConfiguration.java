package neureka.ndim.config.types.sliced;

import neureka.ndim.config.types.D2C;

public class Sliced2DConfiguration extends D2C //:= IMMUTABLE
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
    /**
     *  The mapping for the indices array.
     */
    private final int _indicesMap1;
    private final int _indicesMap2; // Maps index integer to array like translation. Used to avoid distortion when slicing!
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


    protected Sliced2DConfiguration(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        _shape1 = shape[ 0 ];
        _shape2 = shape[ 1 ];
        _translation1 = translation[ 0 ];
        _translation2 = translation[ 1 ];
        _indicesMap1 = indicesMap[ 0 ];
        _indicesMap2 = indicesMap[ 1 ];
        _spread1 = spread[ 0 ];
        _spread2 = spread[ 1 ];
        _offset1 = offset[ 0 ];
        _offset2 = offset[ 1 ];
    }

    public static Sliced2DConfiguration construct(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        return _cached( new Sliced2DConfiguration(shape, translation, indicesMap, spread, offset) );
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
        return new int[]{_indicesMap1, _indicesMap2};
    }

    @Override
    public int indicesMap(int i ) {
        return (i==0)?_indicesMap1:_indicesMap2;
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
        return new int[]{_spread1, _spread2};
    }

    @Override
    public int spread( int i ) {
        return (i==0)?_spread1:_spread2;
    }

    @Override
    public int[] offset() {
        return new int[]{_offset1, _offset2};
    }

    @Override
    public int offset( int i ) {
        return (i==0)?_offset1:_offset2;
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public int indexOfIndex(int index) {
            return ((index / _indicesMap1) * _spread1 + _offset1) * _translation1 +
                    (((index %_indicesMap1) / _indicesMap2) * _spread2 + _offset2) * _translation2;
    }

    @Override
    public int[] indicesOfIndex(int index) {
        int[] indices = new int[ 2 ];
        indices[ 0 ] += index / _indicesMap1;
        index %= _indicesMap1;
        indices[ 1 ] += index / _indicesMap2;
        return indices;
    }

    @Override
    public int indexOfIndices(int[] indices) {
        int i = 0;
        i += (indices[ 0 ] * _spread1 + _offset1) * _translation1;
        i += (indices[ 1 ] * _spread2 + _offset2) * _translation2;
        return i;
    }

    @Override
    public int indexOfIndices(int d1, int d2) {
        int i = 0;
        i += (d1 * _spread1 + _offset1) * _translation1;
        i += (d2 * _spread2 + _offset2) * _translation2;
        return i;
    }
}