package neureka.ndim.config.types.complex;

import neureka.Neureka;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.AbstractNDC;

public final class ComplexDefaultNDConfiguration extends AbstractNDC //:= IMMUTABLE
{

    /**
     *  The shape of the NDArray.
     */
    protected final int[] _shape;
    /**
     *  The translation from a shape index (idx) to the index of the underlying data array.
     */
    private final int[] _translation;
    /**
     *  The mapping of an index to an index array.
     *  The index array is created and filled
     *  during iteration and passed to this configuration for element access...
     *  However it is also possible to creat an index array from an index integer.
     *  This is what the following property does :
     */
    private final int[] _idxmap; // Maps index integer to array like translation. Used to avoid distortion when slicing!
    /**
     *  Produces the strides of a tensor subset / slice
     */
    private final int[] _spread;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private final int[] _offset;
    /**
     *  The value of this tensor. Usually a array of type double[] or float[].
     */

    protected ComplexDefaultNDConfiguration(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ) {
        _shape = _cacheArray(shape);
        _translation = _cacheArray(translation);
        _idxmap = _cacheArray(idxmap);
        _spread = _cacheArray(spread);
        _offset = _cacheArray(offset);
    }

    public static NDConfiguration construct(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ) {
        return _cached(new ComplexDefaultNDConfiguration(shape, translation, idxmap, spread, offset));
    }

    @Override
    public int rank() {
        return _shape.length;
    }

    @Override
    public int[] shape() {
        return _shape;
    }

    @Override
    public int shape( int i ) {
        return _shape[ i ];
    }

    @Override
    public int[] idxmap() {
        return _idxmap;
    }

    @Override
    public int idxmap( int i ) {
        return _idxmap[ i ];
    }

    @Override
    public int[] translation() {
        return _translation;
    }

    @Override
    public int translation( int i ) {
        return _translation[ i ];
    }

    @Override
    public int[] spread() {
        return _spread;
    }

    @Override
    public int spread( int i ) {
        return _spread[ i ];
    }

    @Override
    public int[] offset() {
        return _offset;
    }

    @Override
    public int offset( int i ) {
        return _offset[ i ];
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public int i_of_i( int i ) {
        return i_of_idx(idx_of_i( i ));
    }

    @Override
    public int[] idx_of_i( int i ) {
        int[] idx = new int[_shape.length];
        for ( int ii = 0; ii < rank(); ii++ ) {
            idx[ ii ] += i / _idxmap[ ii ];
            i %= _idxmap[ ii ];
        }
        return idx;
    }

    @Override
    public int i_of_idx( int[] idx ) {
        int i = 0;
        for ( int ii = 0; ii < _shape.length; ii++ ) i += (idx[ ii ] * _spread[ ii ] + _offset[ ii ]) * _translation[ ii ];
        return i;
    }

    
}
