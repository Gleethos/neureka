package neureka.ndim.config.types.sliced;

import neureka.ndim.config.AbstractNDC;

public class SlicedNDConfiguration extends AbstractNDC //:= IMMUTABLE
{
    /**
     *  The shape of the NDArray.
     */
    private final int[] _shape;
    /**
     *  The translation from a shape index (indices) to the index of the underlying data array.
     */
    private final int[] _strides;
    /**
     *  The mapping of an index to an index array.
     *  The index array is created and filled
     *  during iteration and passed to this configuration for element access...
     *  However, it is also possible to create an index array from an index integer.
     *  This is what the following property does :
     */
    private final int[] _indicesMap; // Maps index integer to array similar as translation. Used to avoid distortion when slicing!
    /**
     *  Produces the steps of a tensor subset / slice
     */
    private final int[] _spread;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int[] _offset;


    protected SlicedNDConfiguration(
            int[] shape,
            int[] strides,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        _shape       = _cacheArray(shape);
        _strides     = _cacheArray(strides);
        _indicesMap  = _cacheArray(indicesMap);
        _spread      = _cacheArray(spread);
        _offset      = _cacheArray(offset);
    }

    public static SlicedNDConfiguration construct(
            int[] shape,
            int[] strides,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        return _cached( new SlicedNDConfiguration(shape, strides, indicesMap, spread, offset) );
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
    @Override public final int indicesMap( int i ) { return _indicesMap[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] strides() { return _strides.clone(); }

    /** {@inheritDoc} */
    @Override public final int strides(int i ) { return _strides[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] spread() { return _spread.clone(); }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return _spread[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] offset() { return _offset.clone(); }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return _offset[ i ]; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) { return indexOfIndices( indicesOfIndex( index ) ); }

    /** {@inheritDoc} */
    @Override
    public final int[] indicesOfIndex( int index ) {
        int[] indices = new int[ _shape.length ];
        for ( int i = 0; i < rank(); i++ ) {
            indices[ i ] += index / _indicesMap[ i ];
            index %= _indicesMap[ i ];
        }
        return indices;
    }

    /** {@inheritDoc} */
    @Override
    public final int indexOfIndices( int[] indices ) {
        int index = 0;
        for ( int i = 0; i < _shape.length; i++ )
            index += ( indices[ i ] * _spread[ i ] + _offset[ i ] ) * _strides[ i ];
        return index;
    }

    
}
