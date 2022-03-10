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
    private final int[] _translation;
    /**
     *  The mapping of an index to an index array.
     *  The index array is created and filled
     *  during iteration and passed to this configuration for element access...
     *  However, it is also possible to creat an index array from an index integer.
     *  This is what the following property does :
     */
    private final int[] _indicesMap; // Maps index integer to array like translation. Used to avoid distortion when slicing!
    /**
     *  Produces the strides of a tensor subset / slice
     */
    private final int[] _spread;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int[] _offset;


    protected SlicedNDConfiguration(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        _shape       = _cacheArray(shape);
        _translation = _cacheArray(translation);
        _indicesMap  = _cacheArray(indicesMap);
        _spread      = _cacheArray(spread);
        _offset      = _cacheArray(offset);
    }

    public static SlicedNDConfiguration construct(
            int[] shape,
            int[] translation,
            int[] indicesMap,
            int[] spread,
            int[] offset
    ) {
        return _cached( new SlicedNDConfiguration(shape, translation, indicesMap, spread, offset) );
    }

    @Override
    public final int rank() {
        return _shape.length;
    }

    @Override
    public final int[] shape() {
        return _shape;
    }

    @Override
    public final int shape( int i ) {
        return _shape[ i ];
    }

    @Override
    public final int[] indicesMap() {
        return _indicesMap;
    }

    @Override
    public final int indicesMap(int i ) {
        return _indicesMap[ i ];
    }

    @Override
    public final int[] translation() {
        return _translation;
    }

    @Override
    public final int translation( int i ) {
        return _translation[ i ];
    }

    @Override
    public final int[] spread() {
        return _spread;
    }

    @Override
    public final int spread( int i ) {
        return _spread[ i ];
    }

    @Override
    public final int[] offset() {
        return _offset;
    }

    @Override
    public final int offset( int i ) {
        return _offset[ i ];
    }

    @Override
    public final int indexOfIndex( int index ) {
        return indexOfIndices( indicesOfIndex( index ) );
    }

    @Override
    public final int[] indicesOfIndex( int index ) {
        int[] indices = new int[ _shape.length ];
        for ( int ii = 0; ii < rank(); ii++ ) {
            indices[ ii ] += index / _indicesMap[ ii ];
            index %= _indicesMap[ ii ];
        }
        return indices;
    }

    @Override
    public final int indexOfIndices( int[] indices ) {
        int i = 0;
        for ( int ii = 0; ii < _shape.length; ii++ )
            i += ( indices[ ii ] * _spread[ ii ] + _offset[ ii ] ) * _translation[ ii ];
        return i;
    }

    
}
