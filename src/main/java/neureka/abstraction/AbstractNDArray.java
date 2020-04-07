package neureka.abstraction;

import neureka.Neureka;
import neureka.Tsr;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Collections;
import java.util.Locale;
import java.util.Map;
import java.util.WeakHashMap;


/**
 *  This is the precursor class to the final Tsr class from which
 *  tensor instances can be created.
 *  The inheritance model of a tensor is structured as follows:
 *  Tsr inherits from AbstractNDArray which inherits from AbstractComponentOwner
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 *
 */
public abstract class AbstractNDArray extends AbstractComponentOwner
{
    static
    {
        _CONFIGS = Collections.synchronizedMap(new WeakHashMap<>()) ;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     *  Cached configuration
     */
    private final static Map<Long, int[]> _CONFIGS;

    /**
     *  The shape of the NDArray.
     */
    protected int[] _shape;
    /**
     *  The translation from a shape index (idx) to the index of the underlying data array.
     */
    protected int[] _translation;
    /**
     *  The mapping of idx array.
     */
    protected int[] _idxmap; // Used to avoid distorion when reshaping!
    /**
     *  Produces the strides of a tensor subset / slice
     */
    protected int[] _spread;
    /**
     *  Defines the position of a subset / slice tensor within its parent!
     */
    protected int[] _offset;
    /**
     *  The value of this tensor. Usually a array of type double[] or float[].
     */
    protected Object _value;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public boolean is64(){
        return _value instanceof double[];
    }

    public  boolean is32(){
        return _value instanceof float[];
    }


    protected static int[] _cached(int[] data) {
        if(true) return data;
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
        if (found != null) {
            return found;
        } else {
            _CONFIGS.put(key, data);
            return data;
        }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public int i_of_idx(int[] idx) {
        int i = 0;
        for (int ii=0; ii<_shape.length; ii++) i += (idx[ii] * _spread[ii] + _offset[ii]) * _translation[ii];
        return i;
    }

    public int i_of_i(int i){
        return i_of_idx(idx_of_i(i));
    }

    public int[] idx_of_i(int i) {
        int[] idx = new int[_shape.length];
        if (Neureka.instance().settings().indexing().legacy()){
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

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public int[] shape() {
        return _shape;
    }

    public int shape(int i){
        return _shape[i];
    }

    public int rank(){
        return _shape.length;
    }

    public int[] idxmap(){
        return _idxmap;
    }

    public int[] translation() {
        return _translation;
    }

    public int[] spread(){
        return _spread;
    }

    public int[] offset(){
        return _offset;
    }

    public int size() {
        return Utility.Indexing.szeOfShp(this.shape());
    }



    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    /**
     *  Static methods.
     */
    public static class Utility
    {
        public static class Stringify
        {
            @Contract(pure = true)
            public static String formatFP(double v){
                DecimalFormatSymbols formatSymbols = new DecimalFormatSymbols(Locale.US);
                DecimalFormat Formatter = new DecimalFormat("##0.0##E0", formatSymbols);
                String vStr = String.valueOf(v);
                if (vStr.length()>7){
                    if (vStr.startsWith("0.")){
                        vStr = vStr.substring(0, 7)+"E0";
                    } else if(vStr.startsWith("-0.")){
                        vStr = vStr.substring(0, 8)+"E0";
                    } else {
                        vStr = Formatter.format(v);
                        vStr = (!vStr.contains(".0E0"))?vStr:vStr.replace(".0E0",".0");
                        vStr = (vStr.contains("."))?vStr:vStr.replace("E0",".0");
                    }
                }
                return vStr;
            }

            @Contract(pure = true)
            public static String strConf(int[] conf) {
                StringBuilder str = new StringBuilder();
                for (int i=0; i<conf.length; i++) str.append(conf[i]).append((i != conf.length - 1) ? ", " : "");
                return "[" + str + "]";
            }
        }

        /**
         * Indexing methods.
         */
        public static class Indexing
        {
            @Contract(pure = true)
            public static void increment(@NotNull int[] shpIdx, @NotNull int[] shape) {
                int i;
                if (Neureka.instance().settings().indexing().legacy()) i = 0;
                else i = shape.length-1;
                while (i >= 0 && i < shape.length) i = _incrementAt(i, shpIdx, shape);
            }

            @Contract(pure = true)
            private static int _incrementAt(int i, @NotNull int[] shpIdx, @NotNull int[] shape) {
                if (Neureka.instance().settings().indexing().legacy()) {
                    if (shpIdx[i] < (shape[i])) {
                        shpIdx[i]++;
                        if (shpIdx[i] == (shape[i])) {
                            shpIdx[i] = 0;
                            i++;
                        } else {
                            i = -1;
                        }
                    } else {
                        i++;
                    }
                    return i;
                } else {
                    if (shpIdx[i] < (shape[i])) {
                        shpIdx[i]++;
                        if (shpIdx[i] == (shape[i])) {
                            shpIdx[i] = 0;
                            i--;
                        } else {
                            i = -1;
                        }
                    } else {
                        i--;
                    }
                    return i;
                }
            }

            @Contract(pure = true)
            public static int[] newTlnOf(int[] shape) {
                int[] tln = new int[shape.length];
                int prod = 1;
                if (Neureka.instance().settings().indexing().legacy()) {
                    for (int i = 0; i < tln.length; i++) {
                        tln[i] = prod;
                        prod *= shape[i];
                    }
                } else {
                    for (int i = tln.length-1; i >= 0; i--) {
                        tln[i] = prod;
                        prod *= shape[i];
                    }
                }
                return tln;
            }

            @Contract(pure = true)
            public static int[] rearrange(@NotNull int[] array, @NotNull int[] ptr) {
                int[] newShp = new int[ptr.length];
                for (int i = 0; i < ptr.length; i++) {
                    if (ptr[i] < 0) newShp[i] = Math.abs(ptr[i]);
                    else if (ptr[i] >= 0) newShp[i] = array[ptr[i]];
                }
                return newShp;
            }

            @Contract(pure = true)
            public static int[] shpCheck(int[] newShp, Tsr t) {
                if (szeOfShp(newShp) != t.size()) {
                    throw new IllegalArgumentException(
                            "New shape does not match tensor size!" +
                                    " (" + Utility.Stringify.strConf(newShp) + ((szeOfShp(newShp) < t.size()) ? "<" : ">") + Utility.Stringify.strConf(t.shape()) + ")");
                }
                return newShp;
            }

            @Contract(pure = true)
            public static int[] rearrange(int[] tln, int[] shp, @NotNull int[] newForm) {
                int[] shpTln = newTlnOf(shp);
                int[] newTln = new int[newForm.length];
                for (int i = 0; i < newForm.length; i++) {
                    if (newForm[i] < 0) newTln[i] = shpTln[i];
                    else if (newForm[i] >= 0) newTln[i] = tln[newForm[i]];
                }
                return newTln;
            }

            @Contract(pure = true)
            public static int[][] makeFit(int[] sA, int[] sB){
                int lastIndexOfA = 0;
                for (int i=sA.length-1; i>=0; i--) {
                    if(sA[i]!=1){
                        lastIndexOfA = i;
                        break;
                    }
                }
                int firstIndexOfB = 0;
                for (int i=0; i<sB.length; i++){
                    if(sB[i]!=1){
                        firstIndexOfB = i;
                        break;
                    }
                }
                int newSize = lastIndexOfA + sB.length - firstIndexOfB;
                int[] rsA = new int[newSize];
                int[] rsB = new int[newSize];
                for(int i=0; i<newSize; i++) {
                    if(i<=lastIndexOfA) rsA[i] = i; else rsA[i] = -1;
                    if(i>=lastIndexOfA) rsB[i] = i-lastIndexOfA+firstIndexOfB; else rsB[i] = -1;
                }
                return new int[][]{rsA, rsB};
            }

            @Contract(pure = true)
            public static int[] shpOfCon(int[] shp1, int[] shp2) {
                int[] shape = new int[(shp1.length + shp2.length) / 2];
                for (int i = 0; i < shp1.length && i < shp2.length; i++) shape[i] = Math.abs(shp1[i] - shp2[i]) + 1;
                return shape;
            }

            @Contract(pure = true)
            public static int[] shpOfBrc(int[] shp1, int[] shp2) {
                int[] shape = new int[(shp1.length + shp2.length) / 2];
                for (int i = 0; i < shp1.length && i < shp2.length; i++) {
                    shape[i] = Math.max(shp1[i], shp2[i]);
                    if (Math.min(shp1[i], shp2[i])!=1&&Math.max(shp1[i], shp2[i])!=shape[i]) {
                        throw new IllegalStateException("Broadcast not possible. Shapes do not match!");
                    }
                }
                return shape;
            }

            @Contract(pure = true)
            public static int szeOfShp(int[] shape) {
                int size = 1;
                for (int i : shape) size *= i;
                return size;
            }

        }

    }




}
