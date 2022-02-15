package neureka.ndim.config.types.virtual;

import neureka.ndim.config.AbstractNDC;
import neureka.ndim.config.NDConfiguration;

import java.util.Map;
import java.util.WeakHashMap;

public class VirtualNDConfiguration extends AbstractNDC
{
    private static Map<int[], VirtualNDConfiguration> _Virtual_Cache = new WeakHashMap<>();

    private final int[] _shape;

    private VirtualNDConfiguration( int[] shape ) {
        _shape = _cacheArray( shape );
    }

    public static NDConfiguration construct(
            int[] shape
    ) {
        shape = _cacheArray( shape );
        VirtualNDConfiguration found = _Virtual_Cache.get( shape );
        if ( found != null ) {
            return _Virtual_Cache.get( shape );
        }
        found = new VirtualNDConfiguration( shape );
        _Virtual_Cache.put( shape, found );
        return found;
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
    public int indicesMap(int i ) {
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
