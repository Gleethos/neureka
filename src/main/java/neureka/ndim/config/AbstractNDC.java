package neureka.ndim.config;

import neureka.ndim.AbstractNDArray;
import neureka.ndim.config.complex.D1Configuration;
import neureka.ndim.config.complex.DefaultNDConfiguration;
import neureka.ndim.config.complex.ScalarConfiguration;
import neureka.ndim.config.simple.SimpleD1Configuration;
import neureka.ndim.config.simple.SimpleScalarConfiguration;
import neureka.ndim.config.simple.SimpleDefaultNDConfiguration;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;

public abstract class AbstractNDC implements NDConfiguration
{
    /**
     *  Cached configuration
     */
    private static final Map<Long, NDConfiguration> _CONFS;
    static
    {
        _CONFS = Collections.synchronizedMap(new WeakHashMap<>()) ;
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
        boolean isSimple = _isSimpleConfiguration(shape, translation, idxmap, spread, offset);
        NDConfiguration ndc = null;
        if ( isSimple ) {
            if (shape.length == 1) {
                if(shape[0]==1) ndc = SimpleScalarConfiguration.construct();
                else ndc = SimpleD1Configuration.construct(shape, translation);
            } else ndc = SimpleDefaultNDConfiguration.construct(shape, translation);
        } else {
            if (shape.length == 1) {
                if(shape[0]==1) ndc = ScalarConfiguration.construct(shape, offset);
                else ndc = D1Configuration.construct(shape, translation, idxmap, spread, offset);
            } else ndc = DefaultNDConfiguration.construct(shape, translation, idxmap, spread, offset);
        }
        return ndc;
    }

    protected static <T extends NDConfiguration> NDConfiguration _cached(T ndc)
    {
        long key = ndc.keyCode();
        NDConfiguration found = _CONFS.get(key);
        if (
                found != null && ndc.equals(found)
        ) {
            return found;
        } else {
            _CONFS.put(key, ndc);
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
