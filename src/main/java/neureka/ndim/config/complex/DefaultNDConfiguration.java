package neureka.ndim.config.complex;

import neureka.Neureka;
import neureka.ndim.config.NDConfiguration;
import neureka.ndim.config.AbstractArrayBasedNDC;

import java.util.Collections;
import java.util.Map;
import java.util.WeakHashMap;

public class DefaultNDConfiguration extends AbstractArrayBasedNDC
{
    /**
     *  Cached configuration
     */
    private static final Map<Long, DefaultNDConfiguration> _CONFS;
    static
    {
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

    public static NDConfiguration construct(
            int[] shape,
            int[] translation,
            int[] idxmap,
            int[] spread,
            int[] offset
    ){
        return _cached(new DefaultNDConfiguration(shape, translation, idxmap, spread, offset));
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
            for ( int ii = rank()-1; ii >= 0; ii-- ) {
                idx[ii] += i / _idxmap[ii];
                i %= _idxmap[ii];
            }
        } else {
            for ( int ii = 0; ii < rank(); ii++ ) {
                idx[ii] += i / _idxmap[ii];
                i %= _idxmap[ii];
            }
        }
        return idx;
    }

    @Override
    public int i_of_idx(int[] idx) {
        int i = 0;
        for ( int ii = 0; ii < _shape.length; ii++ ) i += (idx[ii] * _spread[ii] + _offset[ii]) * _translation[ii];
        return i;
    }

    
}
