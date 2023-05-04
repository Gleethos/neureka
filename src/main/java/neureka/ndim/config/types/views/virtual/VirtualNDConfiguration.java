package neureka.ndim.config.types.views.virtual;

import neureka.ndim.config.AbstractNDC;

/**
 *  {@link VirtualNDConfiguration}s represent tensors which
 *  are filled homogeneously with a single value exclusively,
 *  like for example a tensor filled with only zeros.
 *  such tensors have very simple data access patterns
 *  which are modeled by this class!
 */
public class VirtualNDConfiguration extends AbstractNDC
{
    private final int[] _shape;

    private VirtualNDConfiguration( int[] shape ) { _shape = _cacheArray( shape ); }

    public static VirtualNDConfiguration construct(
            int[] shape
    ) {
        return _cached( new VirtualNDConfiguration( shape ) );
    }

    /** {@inheritDoc} */
    @Override public final int rank() { return _shape.length; }

    /** {@inheritDoc} */
    @Override public final int[] shape() { return _shape; }

    /** {@inheritDoc} */
    @Override public final int shape( int i ) { return _shape[ i ]; }

    /** {@inheritDoc} */
    @Override public final int[] indicesMap() { return new int[rank()]; }

    /** {@inheritDoc} */
    @Override public final int indicesMap( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int[] strides() { return new int[rank()]; }

    /** {@inheritDoc} */
    @Override public final int strides(int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int[] spread() { return new int[rank()]; }

    /** {@inheritDoc} */
    @Override public final int spread( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int[] offset() { return new int[rank()]; }

    /** {@inheritDoc} */
    @Override public final int offset( int i ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndex( int index ) { return 0; }

    /** {@inheritDoc} */
    @Override public final int[] indicesOfIndex( int index ) { return new int[rank()]; }

    /** {@inheritDoc} */
    @Override public final int indexOfIndices(int[] indices) { return 0; }

    /** {@inheritDoc} */
    @Override public final boolean isVirtual() { return true; }


}
