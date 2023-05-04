package neureka.ndim.config.types.simple;

import neureka.ndim.config.types.D3C;

public class Simple3DConfiguration extends D3C //:= IMMUTABLE
{
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

    protected Simple3DConfiguration(
            int[] shape,
            int[] strides
    ) {
        _shape1  = shape[ 0 ];
        _shape2  = shape[ 1 ];
        _shape3  = shape[ 2 ];
        _stride1 = strides[ 0 ];
        _stride2 = strides[ 1 ];
        _stride3 = strides[ 2 ];
    }

    public static Simple3DConfiguration construct(
            int[] shape,
            int[] strides
    ) {
        return _cached( new Simple3DConfiguration(shape, strides) );
    }

    /** {@inheritDoc} */
    @Override public final int rank() { return 3; }

    /** {@inheritDoc} */
    @Override public final int[] shape() { return new int[]{_shape1, _shape2, _shape3}; }

    /** {@inheritDoc} */
    @Override public final int shape( int i ) { return ( i == 0 ?_shape1 : ( i == 1 ? _shape2 : _shape3 ) ); }

    /** {@inheritDoc} */
    @Override public final int[] indicesMap() { return new int[]{_stride1, _stride2, _stride3}; }

    /** {@inheritDoc} */
    @Override public final int indicesMap(int i ) { return ( i == 0 ? _stride1 : ( i == 1 ? _stride2 : _stride3 ) ); }

    /** {@inheritDoc} */
    @Override public final int[] strides() { return new int[]{_stride1, _stride2, _stride3}; }

    /** {@inheritDoc} */
    @Override public final int strides(int i ) { return ( i == 0 ? _stride1 :( i == 1 ? _stride2 : _stride3 ) ); }

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
        indices1 = index / _stride1;
        index %= _stride1;
        indices2 = index / _stride2;
        index %= _stride2;
        indices3 = index / _stride3;
        return indices1 * _stride1 +
                indices2 * _stride2 +
                indices3 * _stride3;
    }

    /** {@inheritDoc} */
    @Override public final int[] indicesOfIndex( int index ) {
        int indices1, indices2, indices3;
        indices1 = index / _stride1;
        index %= _stride1;
        indices2 = index / _stride2;
        index %= _stride2;
        indices3 = index / _stride3;
        return new int[]{ indices1, indices2, indices3 };
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices(int[] indices) {
        return indices[ 0 ] * _stride1 +
               indices[ 1 ] * _stride2 +
               indices[ 2 ] * _stride3;
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int d1, int d2, int d3 ) {
        return d1 * _stride1 +
               d2 * _stride2 +
               d3 * _stride3;
    }
}