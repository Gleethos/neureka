package neureka.ndim.config.types.reshaped;

import neureka.ndim.config.types.D3C;

public class Reshaped3DConfiguration extends D3C {
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape1, _shape2, _shape3;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int _translation1, _translation2, _translation3;
    /**
     *  The mapping of idx array.
     */
    private final int _indicesMap1, _indicesMap2, _indicesMap3; // Maps index integer to array like translation. Used to avoid distortion when slicing!


    protected Reshaped3DConfiguration(
            int[] shape,
            int[] translation,
            int[] indicesMap
    ) {
        _shape1 = shape[ 0 ];
        _shape2 = shape[ 1 ];
        _shape3 = shape[ 2 ];
        _translation1 = translation[ 0 ];
        _translation2 = translation[ 1 ];
        _translation3 = translation[ 2 ];
        _indicesMap1 = indicesMap[ 0 ];
        _indicesMap2 = indicesMap[ 1 ];
        _indicesMap3 = indicesMap[ 2 ];
    }

    public static Reshaped3DConfiguration construct(
            int[] shape,
            int[] translation,
            int[] indicesMap
    ) {
        return _cached( new Reshaped3DConfiguration(shape, translation, indicesMap) );
    }

    /** {@inheritDoc} */
    @Override public final int rank() { return 3; }

    /** {@inheritDoc} */
    @Override public final int[] shape() { return new int[]{_shape1, _shape2, _shape3}; }

    /** {@inheritDoc} */
    @Override public final int shape( int i ) { return (i==0?_shape1:(i==1?_shape2:_shape3)); }

    /** {@inheritDoc} */
    @Override public final int[] indicesMap() { return new int[]{_indicesMap1, _indicesMap2, _indicesMap3}; }

    /** {@inheritDoc} */
    @Override public final int indicesMap(int i ) { return (i==0?_indicesMap1:(i==1?_indicesMap2:_indicesMap3)); }

    /** {@inheritDoc} */
    @Override public final int[] translation() { return new int[]{_translation1, _translation2, _translation3}; }

    /** {@inheritDoc} */
    @Override public final int translation( int i ) { return (i==0?_translation1:(i==1?_translation2:_translation3)); }

    /** {@inheritDoc} */
    @Override public final int[] spread() { return new int[]{1, 1, 1}; }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public final int[] offset() { return new int[]{0, 0, 0}; }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) {
        int indices1, indices2, indices3;
        indices1 = index / _indicesMap1;
        index %= _indicesMap1;
        indices2 = index / _indicesMap2;
        index %= _indicesMap2;
        indices3 = index / _indicesMap3;
        return indices1 * _translation1 +
                indices2 * _translation2 +
                indices3 * _translation3;
    }

    /** {@inheritDoc} */
    @Override public final int[] indicesOfIndex( int index ) {
        int indices1, indices2, indices3;
        indices1 = index / _indicesMap1;
        index %= _indicesMap1;
        indices2 = index / _indicesMap2;
        index %= _indicesMap2;
        indices3 = index / _indicesMap3;
        return new int[]{indices1, indices2, indices3};
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices(int[] indices) {
        return indices[ 0 ] * _translation1 +
                indices[ 1 ] * _translation2 +
                indices[ 2 ] * _translation3;
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices(int d1, int d2, int d3 ) {
        return d1 * _translation1 +
               d2 * _translation2 +
               d3 * _translation3;
    }

}
