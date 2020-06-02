package neureka.ndim.config;

import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;

public abstract class AbstractArrayBasedNDC extends AbstractNDC
{
    /**
     *  Cached configuration
     */
    private static final Map<Long, int[]> _CACHED_INT_ARRAYS;

    static
    {
        _CACHED_INT_ARRAYS = Collections.synchronizedMap(new WeakHashMap<>()) ;
    }

    protected static int[] _cached(int[] data)
    {
        long key = 0;
        for ( int e : data ) {
            if ( e <= 10 ) key *= 10;
            else if ( e <= 100 ) key *= 100;
            else if ( e <= 1000 ) key *= 1000;
            else if ( e <= 10000 ) key *= 10000;
            else if ( e <= 100000 ) key *= 100000;
            else if ( e <= 1000000 ) key *= 1000000;
            else if ( e <= 10000000 ) key *= 10000000;
            else if ( e <= 100000000 ) key *= 100000000;
            else if ( e <= 1000000000 ) key *= 1000000000;
            key += Math.abs(e) + 1;
        }
        int rank = data.length;
        while ( rank != 0 ) {
            rank /= 10;
            key *= 10;
        }
        key += data.length;
        int[] found = _CACHED_INT_ARRAYS.get(key);
        if ( found != null ) return found;
        else {
            _CACHED_INT_ARRAYS.put(key, data);
            return data;
        }
    }


}
