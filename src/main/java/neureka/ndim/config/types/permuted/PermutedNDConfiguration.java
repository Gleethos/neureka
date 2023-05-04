package neureka.ndim.config.types.permuted;

import neureka.ndim.config.AbstractNDC;

import java.util.Arrays;

public class PermutedNDConfiguration extends AbstractNDC
{
    /**
     *  The shape of the NDArray.
     */
    private final int[] _shape;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int[] _translation;
    /**
     *  The mapping of an index to an index array.
     *  The index array is created and filled
     *  during iteration and passed to this configuration for element access...
     *  However, it is also possible to creat an index array from an index integer.
     *  This is what the following property does :
     */
    private final int[] _indicesMap; // Maps index integer to array like translation. Used to avoid distortion when slicing!


    protected PermutedNDConfiguration(
            int[] shape,
            int[] translation,
            int[] indicesMap
    ) {
        _shape       = _cacheArray(shape);
        _translation = _cacheArray(translation);
        _indicesMap  = _cacheArray(indicesMap);
    }

    public static PermutedNDConfiguration construct(
            int[] shape,
            int[] translation,
            int[] indicesMap
    ) {
        return _cached( new PermutedNDConfiguration(shape, translation, indicesMap) );
    }


    /** {@inheritDoc} */
    @Override public final int rank() { return _shape.length; }

    /** {@inheritDoc} */
    @Override public final int[] shape() { return _shape.clone(); }

    /** {@inheritDoc} */
    @Override public final int shape( int i ) { return _shape[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] indicesMap() { return _indicesMap.clone(); }

    /** {@inheritDoc} */
    @Override public final int indicesMap(int i ) { return _indicesMap[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] strides() { return _translation.clone(); }

    /** {@inheritDoc} */
    @Override public final int strides(int i ) { return _translation[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] spread() {
        int[] spread = new int[rank()];
        Arrays.fill(spread, 1);
        return spread;
    }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return 1; }

    /** {@inheritDoc} */
    @Override public final int[] offset() {
        int[] offset = new int[rank()];
        Arrays.fill(offset, 0);
        return offset;
    }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) { return indexOfIndices( indicesOfIndex( index ) ); }

    /** {@inheritDoc} */
    @Override public final int[] indicesOfIndex( int index ) {
        int[] indices = new int[ _shape.length ];
        for ( int ii = 0; ii < rank(); ii++ ) {
            indices[ ii ] += index / _indicesMap[ ii ];
            index %= _indicesMap[ ii ];
        }
        return indices;
    }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices( int[] indices ) {
        int i = 0;
        for ( int ii = 0; ii < _shape.length; ii++ )
            i += indices[ ii ] * _translation[ ii ];
        return i;
    }

}
