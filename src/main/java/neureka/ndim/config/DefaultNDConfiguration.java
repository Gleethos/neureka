package neureka.ndim.config;

import neureka.Neureka;
import neureka.ndim.NDConfiguration;

import java.util.Arrays;
import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;

public class DefaultNDConfiguration implements NDConfiguration
{
    /**
     *  Cached configuration
     */
    private static final Map<Long, int[]> _CONFIGS;
    private static final Map<Long, DefaultNDConfiguration> _CONFS;
    static
    {
        _CONFIGS = Collections.synchronizedMap(new WeakHashMap<>()) ;
        _CONFS = Collections.synchronizedMap(new WeakHashMap<>()) ;
    }

    private DefaultNDConfiguration(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ) {
        _shape = _cached(shape);
        _translation = _cached(translation);
        _idxmap = _cached(idxmap);
        _spread = _cached(spread);
        _offset = _cached(offset);
    }

    public static DefaultNDConfiguration construct(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ){
        DefaultNDConfiguration newNDC = new DefaultNDConfiguration(shape, translation, idxmap, spread, offset);
        long key = newNDC._key();
        DefaultNDConfiguration found = _CONFS.get(key);
        if(
                found != null &&
                Arrays.hashCode(newNDC.shape())==Arrays.hashCode(found.shape()) &&
                Arrays.hashCode(newNDC.translation())==Arrays.hashCode(found.translation()) &&
                Arrays.hashCode(newNDC.idxmap())==Arrays.hashCode(found.idxmap()) &&
                Arrays.hashCode(newNDC.spread())==Arrays.hashCode(found.spread()) &&
                Arrays.hashCode(newNDC.offset())==Arrays.hashCode(found.offset())
        ){
            return found;
        } else {
            _CONFS.put(key, newNDC);
            return newNDC;
        }
    }

    /**
     *  The shape of the NDArray.
     */
    private int[] _shape;
    /**
     *  The translation from a shape index (idx) to the index of the underlying data array.
     */
    private int[] _translation;
    /**
     *  The mapping of idx array.
     */
    private int[] _idxmap; // Used to avoid distortion when reshaping!
    /**
     *  Produces the strides of a tensor subset / slice
     */
    private int[] _spread;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    private int[] _offset;
    /**
     *  The value of this tensor. Usually a array of type double[] or float[].
     */

    @Override
    public int rank() {
        return _shape.length;
    }

    @Override
    public int[] shape() {
        return _shape;
    }

    @Override
    public int shape(int i) {
        return _shape[i];
    }

    @Override
    public int[] idxmap() {
        return _idxmap;
    }

    @Override
    public int idxmap(int i) {
        return _idxmap[i];
    }

    @Override
    public int[] translation() {
        return _translation;
    }

    @Override
    public int translation(int i) {
        return _translation[i];
    }

    @Override
    public int[] spread() {
        return _spread;
    }

    @Override
    public int spread(int i) {
        return _spread[i];
    }

    @Override
    public int[] offset() {
        return _offset;
    }

    @Override
    public int offset(int i) {
        return _offset[i];
    }


    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    @Override
    public int i_of_i(int i){
        return i_of_idx(idx_of_i(i));
    }

    @Override
    public int[] idx_of_i(int i) {
        int[] idx = new int[_shape.length];
        if (Neureka.instance().settings().indexing().isUsingLegacyIndexing()){
            for (int ii=rank()-1; ii>=0; ii--){
                idx[ii] += i / _idxmap[ii];
                i %= _idxmap[ii];
            }
        } else {
            for (int ii=0; ii<rank(); ii++) {
                idx[ii] += i / _idxmap[ii];
                i %= _idxmap[ii];
            }
        }
        return idx;
    }
    @Override
    public int i_of_idx(int[] idx) {
        int i = 0;
        for (int ii=0; ii<_shape.length; ii++) i += (idx[ii] * _spread[ii] + _offset[ii]) * _translation[ii];
        return i;
    }


    private static int[] _cached(int[] data)
    {
        long key = 0;
        for (int e : data) {
            if (e <= 10) key *= 10;
            else if (e <= 100) key *= 100;
            else if (e <= 1000) key *= 1000;
            else if (e <= 10000) key *= 10000;
            else if (e <= 100000) key *= 100000;
            else if (e <= 1000000) key *= 1000000;
            else if (e <= 10000000) key *= 10000000;
            else if (e <= 100000000) key *= 100000000;
            else if (e <= 1000000000) key *= 1000000000;
            key += Math.abs(e) + 1;
        }
        int rank = data.length;
        while(rank != 0) {
            rank /= 10;
            key*=10;
        }
        key += data.length;
        int[] found = _CONFIGS.get(key);
        if (found != null) return found;
        else {
            _CONFIGS.put(key, data);
            return data;
        }
    }

    private long _key(){
        return _shape.hashCode() + _translation.hashCode() * 2 + _idxmap.hashCode() * 3 + _spread.hashCode() * 4 + _offset.hashCode() * 5;
    }
    
    
}
