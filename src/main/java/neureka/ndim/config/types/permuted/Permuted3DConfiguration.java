package neureka.ndim.config.types.permuted;

import neureka.ndim.config.types.D3C;

public class Permuted3DConfiguration extends D3C {
    /**
     *  The shape of the NDArray.
     */
    protected final int _shape1;
    protected final int _shape2;
    protected final int _shape3;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int _stride1;
    private final int _stride2;
    private final int _stride3;
    /**
     *  The mapping of idx array.
     */
    private final int _indicesMap1;
    private final int _indicesMap2;
    private final int _indicesMap3; // Maps index integer to array like translation. Used to avoid distortion when slicing!


    protected Permuted3DConfiguration(
        int[] shape,
        int[] strides,
        int[] indicesMap
    ) {
        _shape1      = shape[ 0 ];
        _shape2      = shape[ 1 ];
        _shape3      = shape[ 2 ];
        _stride1     = strides[ 0 ];
        _stride2     = strides[ 1 ];
        _stride3     = strides[ 2 ];
        _indicesMap1 = indicesMap[ 0 ];
        _indicesMap2 = indicesMap[ 1 ];
        _indicesMap3 = indicesMap[ 2 ];
    }

    public static Permuted3DConfiguration construct(
        int[] shape,
        int[] strides,
        int[] indicesMap
    ) {
        return _cached( new Permuted3DConfiguration(shape, strides, indicesMap) );
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
    @Override public final int[] strides() { return new int[]{_stride1, _stride2, _stride3}; }

    /** {@inheritDoc} */
    @Override public final int strides(int i ) { return (i==0? _stride1 :(i==1? _stride2 : _stride3)); }

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
        return indices1 * _stride1 +
                indices2 * _stride2 +
                indices3 * _stride3;
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
        return indices[ 0 ] * _stride1 +
                indices[ 1 ] * _stride2 +
                indices[ 2 ] * _stride3;
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices(int d1, int d2, int d3 ) {
        return d1 * _stride1 +
               d2 * _stride2 +
               d3 * _stride3;
    }

}
