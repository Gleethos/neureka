package neureka.abstraction;

import neureka.Neureka;
import neureka.Tsr;
import neureka.acceleration.CPU;
import neureka.acceleration.Device;
import neureka.autograd.GraphNode;
import neureka.framing.Relation;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.Locale;
import java.util.Map;
import java.util.WeakHashMap;


/**
 *  This is the precursor class to the final Tsr class from which
 *  tensor instances can be created.
 *  The inheritance model of a tensor is structured as follows:
 *  Tsr inherits from AbstractTensor which inherits from AbstractComponentOwner
 *  The inheritance model is linear, meaning that all classes involved
 *  are not extended more than once.
 *
 * @param <InstanceType> The final inheritance tip is the Tsr class which is also this type
 */
public abstract class AbstractTensor<InstanceType> extends AbstractComponentOwner
{
    static
    {
        _CONFIGS = new WeakHashMap<>();
        _CPU = new CPU();
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    /**
     *  Default device (host cpu)
     */
    private static Device<Tsr> _CPU;

    /**
     *  Cached configuration
     */
    private static Map<Long, int[]> _CONFIGS;

    /**
     *  Value data fields
     */
    protected int[] _shape, _translation, _idxmap;
    protected Object _value;

    /**
     *  Flag Fields
     */
    private int _flags = 0;//Default

    private final static int RQS_GRADIENT_MASK = 1;
    private final static int IS_OUTSOURCED_MASK = 2;
    private final static int IS_VIRTUAL_MASK = 4;

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public boolean is64(){
        return _value instanceof double[];
    }

    public  boolean is32(){
        return _value instanceof float[];
    }

    /**
     * @param newShape
     * @return
     */
    protected void _configureFromNewShape(int[] newShape) {
        int size = Utility.Indexing.szeOfShp(newShape);
        _value = (_value==null) ? new double[size] : _value;
        int length = (this.is64())?((double[])_value).length:((float[])_value).length;
        if (size != length && !this.isVirtual()) {
            throw new IllegalArgumentException("[Tsr][_iniShape]: Size of shape does not match stored value64!");
        }
        _shape = _cached(newShape);
        _translation = _cached(Utility.Indexing.newTlnOf(newShape));
        _idxmap = _translation;
    }

    protected static int[] _cached(int[] data) {
        long key = 0;
        for (int i = 0; i < data.length; i++) {
            if (data[i] <= 10)  key *= 10;
            else if (data[i] <= 100) key *= 100;
            else if (data[i] <= 1000) key *= 1000;
            else if (data[i] <= 10000) key *= 10000;
            else if (data[i] <= 100000) key *= 100000;
            else if (data[i] <= 1000000) key *= 1000000;
            else if (data[i] <= 10000000) key *= 10000000;
            else if (data[i] <= 100000000) key *= 100000000;
            key += Math.abs(data[i])+1;
        }
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
        int[] sliceCfg = null;
        if (has(int[].class)) sliceCfg = (int[])find(int[].class);
        for (int ii=0; ii<_shape.length; ii++) {
            int scale = (sliceCfg==null || sliceCfg.length==rank()) ? 1 : sliceCfg[rank()+ii];
            i += (idx[ii] * scale + ((sliceCfg==null) ? 0 : sliceCfg[ii])) * _translation[ii];
        }
        return i;
    }

    protected int _i_of_i(int i){
        return i_of_idx(idx_of_i(i));
    }

    public int[] idx_of_i(int i) {
        int[] idx = new int[_shape.length];
        if (Neureka.Settings.Indexing.legacy()){
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

    public int size() {
        if (this.isEmpty()) return 0;
        return Utility.Indexing.szeOfShp(this.shape());
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    protected void _setFlags(int flags){
        if(_flags == -1) return;
        _flags = flags;
    }

    protected int _getFlags(){
        return _flags;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    public abstract InstanceType setRqsGradient(boolean rqsGradient);


    public boolean rqsGradient() {
        return (_flags & RQS_GRADIENT_MASK) == RQS_GRADIENT_MASK;
    }

    protected  void _setRqsGradient(boolean rqsGradient) {
        if (rqsGradient() != rqsGradient) {
            if (rqsGradient) {
                _flags += RQS_GRADIENT_MASK;
            } else {
                _flags -= RQS_GRADIENT_MASK;
            }
        }
    }

    //>>>

    public abstract InstanceType setIsOutsourced(boolean isOutsourced);

    public boolean isOutsourced() {
        return (_flags & IS_OUTSOURCED_MASK) == IS_OUTSOURCED_MASK;
    }

    protected void _setIsOutsourced(boolean isOutsourced) {
        if (isOutsourced() != isOutsourced) {
            if (isOutsourced) _flags += IS_OUTSOURCED_MASK;
            else _flags -= IS_OUTSOURCED_MASK;
        }
    }

    //>>>

    public abstract InstanceType setIsVirtual(boolean isVirtual);

    public boolean isVirtual() {
        return (_flags & IS_VIRTUAL_MASK) == IS_VIRTUAL_MASK;
    }

    protected void _setIsVirtual(boolean isVirtual) {
        if (isVirtual() != isVirtual) {
            if (isVirtual) {
                _flags += IS_VIRTUAL_MASK;
            } else {
                _flags -= IS_VIRTUAL_MASK;
            }
        }
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //

    /**
     * This method is executed when a new Component is added to the tensor.
     * The public add method is implemented in the super class
     * 'AbstractComponentOwner' from which this class inherits.
     * In this super class the component logic is implemented.
     *
     * @param newComponent A component used to access features. (GraphNode, Index, Relation, int[], ...)
     * @return The unchanged object or maybe in future versions: null (component rejected)
     */
    @Override
    protected Object _addOrReject(Object newComponent){
        newComponent = (newComponent instanceof int[]) ? _cached((int[]) newComponent) : newComponent;
        if(newComponent instanceof Device){
            if(!((Device)newComponent).has(this)){
                ((Device)newComponent).add(this);
            }
        }
        return newComponent;
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // HIGH LEVEL PROPERTIES :

    public boolean isEmpty() {
        return _value == null && !this.isOutsourced();
    }

    public boolean isUndefined() {
        return _shape == null;
    }

    public boolean isSlice(){
        Relation child = (Relation)find(Relation.class);
        return (child!=null && child.hasParent());
    }

    public int sliceCount(){
        Relation child = (Relation)find(Relation.class);
        return (child!=null)?child.childCount():0;
    }

    public boolean isSliceParent(){
        Relation parent = (Relation)find(Relation.class);
        return (parent!=null && parent.hasChildren());
    }

    public boolean belongsToGraph() {
        return this.has(GraphNode.class);
    }

    public boolean isLeave() {
        return (!this.has(GraphNode.class)) || ((GraphNode) this.find(GraphNode.class)).isLeave();
    }

    public boolean isBranch() {
        return !this.isLeave();
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    // Direct Access to component (Device)

    /**
     * @return the device on which this tensor is stored or null if it is not outsourced.
     */
    public Device device() {
        if (this.isOutsourced()) return (Device) this.find(Device.class);
        return _CPU;
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
                    if (vStr.substring(0, 2).equals("0.")){
                        vStr = vStr.substring(0, 7)+"E0";
                    } else if(vStr.substring(0, 3).equals("-0.")){
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
            private static String _strShp(int[] shp) {
                String str = "";
                for(int i=0; i<shp.length; i++) str += shp[i] + ((i != shp.length - 1) ? ", " : "");
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
                if (Neureka.Settings.Indexing.legacy()) i = 0;
                else i = shape.length-1;
                while (i >= 0 && i < shape.length) i = _incrementAt(i, shpIdx, shape);
            }

            @Contract(pure = true)
            private static int _incrementAt(int i, @NotNull int[] shpIdx, @NotNull int[] shape) {
                if(Neureka.Settings.Indexing.legacy()){
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
                if(Neureka.Settings.Indexing.legacy()){
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
            public static int[] rearrange(int[] array, @NotNull int[] ptr) {
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
                            "[Tsr][shpCheck(int[] newShp, Tsr t)]: New shape does not match tensor size!" +
                                    " (" + Utility.Stringify._strShp(newShp) + ((szeOfShp(newShp) < t.size()) ? "<" : ">") + Utility.Stringify._strShp(t.shape()) + ")");
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
                    if(Math.min(shp1[i], shp2[i])!=1&&Math.max(shp1[i], shp2[i])!=shape[i]){
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
