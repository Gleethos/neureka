package neureka.ndim.configs.types.virtual;

import neureka.ndim.configs.AbstractNDC;

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

    private VirtualNDConfiguration( int[] shape ) {
        _shape = _cacheArray( shape );
    }

    public static VirtualNDConfiguration construct(
            int[] shape
    ) {
        return _cached( new VirtualNDConfiguration( shape ) );
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
        return _shape[ 0 ];
    }

    @Override
    public int[] indicesMap() {
        return new int[rank()];
    }

    @Override
    public int indicesMap( int i ) {
        return 0;
    }

    @Override
    public int[] translation() {
        return new int[rank()];
    }

    @Override
    public int translation( int i ) {
        return 0;
    }

    @Override
    public int[] spread() {
        return new int[rank()];
    }

    @Override
    public int spread( int i ) {
        return 0;
    }

    @Override
    public int[] offset() {
        return new int[rank()];
    }

    @Override
    public int offset( int i ) {
        return 0;
    }

    @Override
    public int indexOfIndex(int index) {
        return 0;
    }

    @Override
    public int[] indicesOfIndex(int index) {
        return new int[rank()];
    }

    @Override
    public int indexOfIndices(int[] indices) {
        return 0;
    }

}
