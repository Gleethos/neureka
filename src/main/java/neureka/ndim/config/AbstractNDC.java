package neureka.ndim.config;

import neureka.Neureka;
import neureka.ndim.AbstractNDArray;
import neureka.ndim.config.complex.*;
import neureka.ndim.config.simple.*;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;

public abstract class AbstractNDC implements NDConfiguration
{
    /**
     *  Cached configuration
     */
    private static final Map<Long, NDConfiguration> _CACHED_NDCS;
    static
    {
        _CACHED_NDCS = Collections.synchronizedMap(new WeakHashMap<>()) ;
    }

    /**
     *  Cached configuration
     */
    private static final Map<Long, int[]> _CACHED_INT_ARRAYS;

    static
    {
        _CACHED_INT_ARRAYS = Collections.synchronizedMap(new WeakHashMap<>()) ;
    }

    protected static int[] _cacheArray(int[] data)
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

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public long keyCode()
    {
        return Arrays.hashCode( shape() ) +
               Arrays.hashCode( translation() ) * 2 +
               Arrays.hashCode( idxmap() ) * 3 +
               Arrays.hashCode( spread() ) * 4 +
               Arrays.hashCode( offset() ) * 5;
    }

    @Override
    public boolean equals(NDConfiguration ndc)
    {
        return  Arrays.equals(shape(), ndc.shape()) &&
                Arrays.equals(translation(), ndc.translation()) &&
                Arrays.equals(idxmap(), ndc.idxmap()) &&
                Arrays.equals(spread(), ndc.spread()) &&
                Arrays.equals(offset(), ndc.offset());
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public static NDConfiguration construct (
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ) {
        if(Neureka.instance().settings().ndim().isOnlyUsingDefaultNDConfiguration()){
            return DefaultNDConfiguration.construct(shape, translation, idxmap, spread, offset);
        }
        boolean isSimple = _isSimpleConfiguration(shape, translation, idxmap, spread, offset);
        NDConfiguration ndc = null;
        if ( isSimple ) {
            if (shape.length == 1) {
                if(shape[0]==1) ndc = SimpleScalarConfiguration.construct();
                else ndc = SimpleD1Configuration.construct(shape, translation);
            } else if (shape.length == 2) {
                ndc = SimpleD2Configuration.construct(shape, translation);
            } else if(shape.length == 3) {
                ndc = SimpleD3Configuration.construct(shape, translation);
            } else ndc = SimpleDefaultNDConfiguration.construct(shape, translation);
        } else {
            if (shape.length == 1) {
                if(shape[0]==1) ndc = ScalarConfiguration.construct(shape, offset);
                else ndc = D1Configuration.construct(shape, translation, idxmap, spread, offset);
            } else if (shape.length == 2) {
                ndc = D2Configuration.construct(shape, translation, idxmap, spread, offset);
            } else if(shape.length == 3) {
                ndc = D3Configuration.construct(shape, translation, idxmap, spread, offset);
            } else ndc = DefaultNDConfiguration.construct(shape, translation, idxmap, spread, offset);
        }
        return ndc;
    }

    protected static <T extends NDConfiguration> NDConfiguration _cached(T ndc)
    {
        long key = ndc.keyCode();
        NDConfiguration found = _CACHED_NDCS.get(key);
        if (
                found != null && ndc.equals(found)
        ) {
            return found;
        } else {
            _CACHED_NDCS.put(key, ndc);
            return ndc;
        }
    }

    private static boolean _isSimpleConfiguration(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ) {
        int[] newTranslation = AbstractNDArray.Utility.Indexing.newTlnOf(shape);
        int[] newSpread = new int[shape.length];
        Arrays.fill(newSpread, 1);
        return  Arrays.equals(translation, newTranslation) &&
                Arrays.equals(idxmap, newTranslation) &&
                Arrays.equals(offset, new int[shape.length]) &&
                Arrays.equals(spread, newSpread);
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public String toString(){
        return "(NDConfiguration|@"+Integer.toHexString(hashCode())+"#"+Long.toHexString(keyCode())+"):{ " +
                    "shape : "+Arrays.toString(shape())+", "+
                    "translation : "+Arrays.toString(translation())+", "+
                    "idxmap : "+Arrays.toString(idxmap())+", "+
                    "spread : "+Arrays.toString(spread())+", "+
                    "offset : "+Arrays.toString(offset())+" "+
                "}";
    }


}
