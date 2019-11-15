package neureka.core;

import neureka.core.device.Device;
import neureka.core.function.Function;
import neureka.core.function.factory.autograd.GraphNode;
import neureka.core.utility.DataHelper;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class Tsr {

    // DEFAULT DEVICE (HOST CPU)
    //=========================
    private static Device CPU;
    //OpenClDevice!!!!!!!

    //STATIC FUNCTIONS MEMORY:
    //=========================
    private static HashMap<Long, int[]> CONFIGS;

    static {
        CONFIGS = new HashMap<>();//The things we do for memory
        CPU = null;//new AparapiDevice(null);//<= creates CPU-Aparapi-Kernel
    }
    //-----------------------------------------------------------------------

    //MODULE I / O :
    //=========================
    private ArrayList<Object> _components = new ArrayList<Object>();

    //-----------------------------------------------------------------------
    public ArrayList<Object> getComponents() {
        return _components;
    }

    public Tsr setComponents(ArrayList<Object> properties) {
        _components = properties;
        return this;
    }

    public Tsr add(Object newComponent) {
        if (newComponent == null) {
            return this;
        }
        if (_components != null) {
            Object oldCompartment = find(newComponent.getClass());
            if (oldCompartment != null) {
                _components.remove(oldCompartment);
                _components.trimToSize();
            }
            _components.add((newComponent instanceof int[]) ? cached((int[]) newComponent) : newComponent);
        } else {
            _components = new ArrayList<>();
            _components.add(newComponent);
        }
        return this;
    }

    public Object find(Class componentClass) {
        if (_components != null) {
            for (int Pi = 0; Pi < _components.size(); Pi++) {
                if (componentClass.isInstance(_components.get(Pi))) {
                    return _components.get(Pi);
                }
            }
        }
        return null;
    }

    public Tsr remove(Class componentClass) {
        Object oldComponent = find(componentClass);
        if (oldComponent != null) {
            _components.remove(oldComponent);
            _components.trimToSize();
        }
        if (_components.size() == 0) {
            _components = null;
        }
        return this;
    }

    public boolean has(Class componentClass) {
        return find(componentClass) != null;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //DATA FIELDS:
    //=========================
    private int[] _shape, _translation, _idxmap;
    private Object _value, _gradient;
    //-----------------------------------------------------------------------

    public Device device() {
        if (this.isOutsourced()) {
            return (Device) this.find(Device.class);
        }
        return CPU;
    }

    public double[] targetValue64(boolean unique){
        if(!this.isOutsourced() && unique){
            double[] value = new double[this.size()];
            double[] v = (gradientIsTargeted())? gradient64(): value64();
            for(int i=0; i<value.length; i++){
                value[i] = v[i];
            }
            return value;
        } else {
            return (gradientIsTargeted())? gradient64(): value64();
        }
    }

    public float[] targetValue32(boolean unique){
        if(!this.isOutsourced() && unique){
            float[] value = new float[this.size()];
            float[] v = (gradientIsTargeted())? gradient32(): value32();
            for(int i=0; i<value.length; i++){
                value[i] = v[i];
            }
            return value;
        } else {
            return (gradientIsTargeted())? gradient32(): value32();
        }
    }

    public double[] targetValue64(){
        return (gradientIsTargeted())? gradient64(): value64();
    }

    public float[] targetValue32(){
        return (gradientIsTargeted())? gradient32(): value32();
    }

    public Tsr setTargetValue64(double[] value){
        if(this.isOutsourced()){
            ((Device) this.find(Device.class)).overwrite64(this, value);
        } else {
            if(this.gradientIsTargeted()){
                _gradient = value;
            } else {
                _value = value;
            }
        }
        return this;
    }

    public Tsr setTargetValue32(float[] value){
        if(this.isOutsourced()){
            ((Device) this.find(Device.class)).overwrite32(this, value);
        } else {
            if(this.gradientIsTargeted()){
                _gradient = value;
            } else {
                _value = value;
            }
        }
        return this;
    }

    public Tsr setTargetValue(Object value){
        if(value instanceof float[]){
            this.setTargetValue32((float[])value);
        } else if(value instanceof  double[]){
            this.setTargetValue64((double[])value);
        } else if(value instanceof Float){
            if(this.gradientIsTargeted()){
                if(this.is32()){
                    _gradient = Tsr.fcn.newFloatArray(((Float)value).floatValue(), this.size());
                } else {
                    _gradient = Tsr.fcn.newDoubleArray(((Float)value).doubleValue(), this.size());
                }
            } else {
                this.setIsVirtual(true);
                if(this.is32()){
                    ((float[])_value)[0] = ((Float)value).floatValue();
                } else {
                    ((double[])_value)[0] = ((Float)value).doubleValue();
                }
            }
        } else if(value instanceof Double){
            if(this.gradientIsTargeted()){
                if(this.is32()){
                    _gradient = Tsr.fcn.newFloatArray(((Double)value).floatValue(), this.size());
                } else {
                    _gradient = Tsr.fcn.newDoubleArray(((Double)value).doubleValue(), this.size());
                }
            } else {
                this.setIsVirtual(true);
                if(this.is64()){
                    ((double[])_value)[0] = ((Double)value).doubleValue();
                } else {
                    ((float[])_value)[0] = ((Double)value).floatValue();
                }
            }
        }
        return this;
    }

    public double[] gradient64() {
        if (this.rqsGradient() && this.isOutsourced() && this.has(Device.class)) {
            return ((Device) find(Device.class)).value64Of(this, true);
        }
        return (this.is64())?(double[])_gradient:DataHelper.floatToDouble((float[])_gradient);
    }
    public float[] gradient32(){
        return (this.is64())?DataHelper.doubleToFloat((double[])_gradient):(float[])_gradient;
    }

    public Tsr addToGradient(Tsr g) {
        if(this.isOutsourced()){
            Device device = (Device) this.find(Device.class);
            this.setGradientIsTargeted(true);
            device.add(g);
            device.execute(new Tsr[]{this, g}, Function.TYPES.LOOKUP.get("<"), -1);
            device.get(g);
            this.setGradientIsTargeted(false);
        } else {
            if(this.is64()){
                double[] value = g.value64();
                _gradient = (_gradient==null)?new double[value.length]:_gradient;
                for(int i=0; i<value.length; i++){
                    ((double[])_gradient)[i] = value[i];
                }
            } else {
                float[] value = g.value32();
                _gradient = (_gradient==null)?new float[value.length]:_gradient;
                for(int i=0; i<value.length; i++){
                    ((float[])_gradient)[i] = value[i];
                }
            }
        }
        return this;
    }

    public boolean is64(){
        return _value instanceof double[];
    }
    public  boolean is32(){
        return _value instanceof float[];
    }

    public Tsr to32(){
        if(this.is64()){
            Device device = (Device) this.find(Device.class);
            if(device!=null){
                device.get(this);
            }
            _value = DataHelper.doubleToFloat((double[])_value);
            _gradient = DataHelper.doubleToFloat((double[])_gradient);
            if(device!=null){
                device.add(this);
            }
        }
        return this;
    }

    public Tsr to64(){
        if(this.is32()){
            Device device = (Device) this.find(Device.class);
            if(device!=null){
                device.get(this);
            }
            _value = DataHelper.floatToDouble((float[])_value);
            _gradient = DataHelper.floatToDouble((float[])_gradient);
            if(device!=null){
                device.add(this);
            }
        }
        return this;
    }

    public double[] value64() {
        if (_value == null && this.isOutsourced() && this.has(Device.class)) {
            return ((Device) this.find(Device.class)).value64Of(this, false);
        }
        double[] newValue = (this.is64())?(double[])_value: DataHelper.floatToDouble((float[])_value);
        if (this.isVirtual()) {
            newValue = new double[this.size()];
            double[] value = (this.is64())?(double[])_value:DataHelper.floatToDouble((float[])_value);
            for (int i = 0; i < newValue.length; i++) {
                newValue[i] = value[0];
            }
        }
        return newValue;
    }

    public float[] value32(){
        if (_value == null && this.isOutsourced() && this.has(Device.class)) {
            return ((Device) this.find(Device.class)).value32Of(this, false);
        }
        float[] newValue = (this.is64())?DataHelper.doubleToFloat((double[])_value):(float[])_value;
        if (this.isVirtual()) {
            newValue = new float[this.size()];
            for (int i = 0; i < newValue.length; i++) {
                newValue[i] = newValue[0];
            }
        }
        return newValue;
    }


    public Tsr setValue(double[] newValue) {
        _value = newValue;
        if (this.isOutsourced() && newValue != null) {
            ((Device) this.find(Device.class)).add(this);
        }
        return this;
    }

    public int[] shape() {
        //if(this.has(int[].class)){
        //    int[] shape = new int[this.rank()];
        //    int[] idx = (int[])this.find(int[].class);
        //    for(int i=0; i<shape.length; i++){
        //        shape[i] = _shape[i] - idx[i];
        //    }
        //    return shape;
        //}
        return _shape;
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
        if (this.isEmpty()) {
            return 0;
        }
        return fcn.indexing.szeOfShp(this.shape());
    }

    public boolean isEmpty() {
        return _value == null && !this.isOutsourced();
    }

    public boolean isUndefined() {
        return _shape == null;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //FLAG FIELDS:
    //=========================
    private int _flags = 0 + 0 + 0 + 0;//Default
    private final static int RQS_GRADIENT_MASK = 1;
    private final static int IS_OUTSOURCED_MASK = 2;
    private final static int IS_VIRTUAL_MASK = 4;
    private final static int GRADIENT_IS_TARGETED_MASK = 8;
    //-----------------------------------------------------------------------
    public boolean rqsGradient() {
        return (_flags & RQS_GRADIENT_MASK) == RQS_GRADIENT_MASK;
    }

    public Tsr setRqsGradient(boolean rqsGradient) {
        if (rqsGradient() != rqsGradient) {
            if (rqsGradient) {
                _flags += RQS_GRADIENT_MASK;
            } else {
                this.setGradientIsTargeted(false);
                if(this.isOutsourced()){
                    ((Device)find(Device.class)).get(this);
                    _gradient = null;
                    ((Device)find(Device.class)).add(this);
                }
                _flags -= RQS_GRADIENT_MASK;
            }
        }
        return this;
    }
    //---
    public boolean isOutsourced() {
        return (_flags & IS_OUTSOURCED_MASK) == IS_OUTSOURCED_MASK;
    }

    public Tsr setIsOutsourced(boolean isOutsourced) {
        if (isOutsourced() != isOutsourced) {
            if (isOutsourced) {
                _flags += IS_OUTSOURCED_MASK;
            } else {
                _flags -= IS_OUTSOURCED_MASK;
            }
        }
        if (isOutsourced) {
            _value = null;
            _gradient = null;
        } else if (this.has(Device.class)) {
            Device device = (Device) this.find(Device.class);
            if (device.has(this)) {
                device.get(this);
            }
            this.remove(Device.class);
        }
        return this;
    }
    //---
    public boolean isVirtual() {
        return (_flags & IS_VIRTUAL_MASK) == IS_VIRTUAL_MASK;
    }

    public Tsr setIsVirtual(boolean isVirtual) {
        if (isVirtual() != isVirtual) {
            if(this.isOutsourced()){
                if (!isVirtual) {
                    _flags -= IS_VIRTUAL_MASK;
                }
            } else {
                double v = (_value==null)?0:(((this.is64())?((double[])_value)[0]:((float[])_value)[0]));
                if (isVirtual) {
                    _value = new double[]{v};
                    _flags += IS_VIRTUAL_MASK;
                } else {
                    _value = (this.is64())?new double[this.size()]:new float[this.size()];
                    int length = (this.is64())?((double[])_value).length:((float[])_value).length;
                    for (int i = 0; i < length; i++) {
                        if(this.is64()){
                            ((double[])_value)[i] = v;
                        } else {
                            ((float[])_value)[i] = (float)v;
                        }
                    }
                    _flags -= IS_VIRTUAL_MASK;
                }
            }
        }
        return this;
    }
    //---
    public boolean gradientIsTargeted() {
        return (_flags & GRADIENT_IS_TARGETED_MASK) == GRADIENT_IS_TARGETED_MASK;
    }

    public Tsr setGradientIsTargeted(boolean gradientIsTargeted) {
        if (this.gradientIsTargeted() != gradientIsTargeted) {
            if (gradientIsTargeted) {
                _flags += (this.rqsGradient())?GRADIENT_IS_TARGETED_MASK:0;
            } else {
                _flags -= GRADIENT_IS_TARGETED_MASK;
            }
        }
        return this;
    }
    //---
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //GENERIC PROPERTIES :
    //=========================
    public boolean hasDataParent(){
        return this.has(Tsr.class);
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
    //DISPLAY :
    //=========================
    public String toString(String mode){
        return _toString(mode, (mode.contains("f"))?"    ":null);
    }

    private String _toString(String mode, String deep) {
        String base = (deep==null)?"":"\n"+deep;
        String delimiter = (deep==null)?"":"    ";
        String half = (deep==null)?"":"  ";
        String deeper = (deep==null)?deep:deep+delimiter;
        int max = (mode.contains("s"))?3:50;
        if (this.isEmpty()) {
            return "empty";
        } else if (this.isUndefined()) {
            return "undefined";
        }
        String strShape = "";
        int[] shape = shape();
        for (int i = 0; i < _shape.length; i++) {
            strShape += shape[i];
            if (i < shape.length - 1) {
                strShape += "x";
            }
        }
        boolean compact = mode.contains("c");
        strShape = "[" + strShape + "]";
        String asString = "";
        asString += _stringified((value64()), compact, max);//(this.isOutsourced())?this.value64():_value
        asString = strShape + ":(" + asString + ")";
        if(mode.contains("g")){
            if(this.rqsGradient()){
                asString += ":g:";
                double[] gradient = this.gradient64();
                if(gradient!=null){
                    asString += "("+_stringified((gradient64()), compact, max)+")";
                } else {
                    asString += "(null)";
                }
            }
        }
        if (mode.contains("r")) {
            if (this.has(GraphNode.class) && ((GraphNode) this.find(GraphNode.class)).size() > 0) {
                GraphNode node = (GraphNode) this.find(GraphNode.class);
                AtomicReference<String> enclosed = new AtomicReference<>("; ");
                node.forEach((t, d) -> {
                    enclosed.set(enclosed.get() +
                            base+"=>d|[ " +
                            base+delimiter+    d._toString(mode, deeper) + " " +
                            base+half+"]|:t{ " +
                            base+delimiter+    t._toString(mode, deeper) + " " +
                            base+half+"}, ");

                });
                asString += enclosed.get();
            }
        }
        if (mode.contains("d")) {
            if (this.has(GraphNode.class) && ((GraphNode) this.find(GraphNode.class)).size() > 0) {
                GraphNode node = (GraphNode) this.find(GraphNode.class);
                if (node.mode() != 0) {//node.getMap().values().stream().coll
                    AtomicReference<String> enclosed = new AtomicReference<>("; ");
                    node.forEach((t, d) -> {
                        enclosed.set(enclosed.get() +
                                "->d" + d._toString(mode, deeper) + ", ");
                    });
                    asString += enclosed.get();
                }
            }
        }
        return asString;
    }

    private String _stringified(double[] v, boolean format, int max){
        String asString = "";
        int size = this.size();//(this.isVirtual() ? this.size() : v.length);
        int trim = (size-max);
        size = (trim>0)?max:size;
        for (int i = 0; i < size; i++) {
            String vStr;
            if(format){
                vStr = fcn.stringify.formatFP(
                        v[
                                (this.isVirtual())
                                        ? 0
                                        : Tsr.fcn.indexing.i_of_i(i, this)
            ]);
            } else {
                vStr = String.valueOf(v[(this.isVirtual()) ? 0 : Tsr.fcn.indexing.i_of_i(i, this)]);
            }
            asString += vStr;
            if (i < size - 1) {
                asString += ", ";
            } else if(trim>0){
                asString += ", ... + "+trim+" more";
            }
        }
        return asString;
    }

    public String toString() {
        return toString("dgc");
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //CONSTRUCTION :
    //=========================
    //Generic construction: (Groovy, Scala, ...)
    public Tsr(Object arg){
        _construct(new Object[]{arg});
    }
    public Tsr(Object arg1, Object arg2){
            _construct(new Object[]{arg1, arg2});
    }
    public Tsr(Object arg1, Object arg2, Object arg3){
        _construct(new Object[]{arg1, arg2, arg3});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4){
        _construct(new Object[]{arg1, arg2, arg3, arg4});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8, Object arg9){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9});
    }
    public Tsr(Object arg1, Object arg2, Object arg3, Object arg4, Object arg5, Object arg6, Object arg7, Object arg8, Object arg9, Object arg10){
        _construct(new Object[]{arg1, arg2, arg3, arg4, arg5, arg6, arg7, arg8, arg9, arg10});
    }

    private int[] intArray(Object[] arg){
        int length = ((Object[])arg).length;
        int[] array = new int[length];
        for(int i=0; i<length; i++){
            array[i] = ((Integer)((Object[])arg)[i]).intValue();
        }
        return array;
    }

    private double[] doubleArray(Object[] arg){
        int length = ((Object[])arg).length;
        double[] array = new double[length];
        for(int i=0; i<length; i++){
            if(((Object[])arg)[i] instanceof Integer){
                array[i] = ((Integer)((Object[])arg)[i]).intValue();
            } else if(((Object[])arg)[i] instanceof Double) {
                array[i] = ((Double)((Object[])arg)[i]).doubleValue();
            }
        }
        return array;
    }

    public Tsr(Object[] args){
        _construct(args);
    }

    private void _construct(Object[] args) {
        if(args==null || args.length==0)return;
        args[0] = (args[0] instanceof ArrayList)?((ArrayList)args[0]).toArray():args[0];
        args[1] = (args[1] instanceof ArrayList)?((ArrayList)args[1]).toArray():args[1];
        if(args[0] instanceof Object[]){
            if(((Object[])args[0])[0] instanceof Integer){
                args[0] = intArray((Object[])args[0]);//array;
            } else {
                int length = ((Object[])args[0]).length;
                Tsr[] array = new Tsr[length];
                for(int i=0; i<length; i++){
                    array[i] = (Tsr)((Object[])args[0])[i];
                }
                args[0] = array;
            }
        }
        if(args[1] instanceof Object[]){
            if(((Object[])args[1])[0] instanceof Integer){
                args[1] = doubleArray((Object[]) args[1]);
            }
        }

        //CASES:
        if(args[0] instanceof int[] && (args[1] instanceof Double ||args[1] instanceof Integer)){
            args[1] = (args[1] instanceof Integer)?((Integer)args[1]).doubleValue():args[1];
            _construct((int[])args[0], ((Double)args[1]).doubleValue());
            return;
        } else if(args[0] instanceof Tsr[] && args[1] instanceof String){
            _construct((Tsr[])args[0], (String)args[1], true);
            return;
        } else if(args[0] instanceof int[] && args[1] instanceof double[]){
            _construct((int[])args[0], (double[])args[1]);
            return;
        }
        //EQUATION:
        boolean containsString = false;
        int numberOfTensors = 0;
        ArrayList<Tsr> list = new ArrayList<>();
        for(Object o : args){
            containsString = (o instanceof  String)?true:containsString;
            if(o instanceof Tsr){
                if(!list.contains((Tsr)o)){
                    list.add((Tsr)o);
                    numberOfTensors ++;
                }
            }
        }
        boolean doAD = true;
        Tsr[] tsrs = new Tsr[numberOfTensors];
        String f = "";
        int ti=0;
        for(Object o : args){
            if(list.contains(o)){
                tsrs[ti] = ((Tsr)o);
                f+=("I["+ti+"]");
                ti++;
            } else if(o instanceof  String){
                f+=(String)o;
            } else if(o instanceof  Boolean){
                doAD = (Boolean)o;
            }
        }
        _construct(tsrs, f, doAD);

    }

    public Tsr() {}// creates empty tensor;

    public Tsr(double value){
        _construct(new int[]{1}, value);
    }

    public Tsr(int[] shape) {
        _value = new double[fcn.indexing.szeOfShp(shape)];
        this.initialShape(shape);
    }

    public Tsr(int[] shape, double value) {
        _construct(shape, value);
    }

    private void _construct(int[] shape, double value){
        int size = fcn.indexing.szeOfShp(shape);
        _value = new double[1];
        this.setIsVirtual((size > 1));
        this.initialShape(shape);
        ((double[])_value)[0] = value;
    }

    public Tsr(int[] shape, double[] value) {
        _construct(shape, value);
    }

    private void _construct(int[] shape, double[] value){
        _value = value;
        initialShape(shape);
    }

    /**
     * @param tensor which acts as template for this new tensor.
     */
    public Tsr(Tsr tensor) {
        _shape = tensor._shape;
        _translation = tensor._translation;
        _idxmap = tensor._idxmap;
        _value = (this.is64())?new double[tensor.size()]:new float[tensor.size()];
        _components = null;//tensor._components;
        _flags = tensor._flags;
        int length = (this.is64())?((double[])_value).length:((float[])_value).length;
        for (int i = 0; i < length; i++) {
            if(tensor.is64()){
                ((double[])_value)[i] = ((double[])tensor._value)[i];
            } else {
                ((float[])_value)[i] = ((float[])tensor._value)[i];
            }
        }
        this.setIsOutsourced(false);
    }

    /**
     * @param newShape
     * @return
     */
    public Tsr initialShape(int[] newShape) {
        int size = fcn.indexing.szeOfShp(newShape);
        _value = (_value==null)?new double[size]:_value;
        int length = (this.is64())?((double[])_value).length:((float[])_value).length;
        if (size != length && !this.isVirtual()) {
            throw new IllegalArgumentException("Size of shape does not match stored value64!");
        }
        _shape = cached(newShape);
        _translation = cached(fcn.indexing.idxTln(newShape));
        _idxmap = _translation;
        return this;
    }

    private static int[] cached(int[] data) {
        long key = 0;
        for (int i = 0; i < data.length; i++) {
            if (data[i] <= 10) {
                key *= 10;
            } else if (data[i] <= 100) {
                key *= 100;
            } else if (data[i] <= 1000) {
                key *= 1000;
            } else if (data[i] <= 10000) {
                key *= 10000;
            } else if (data[i] <= 100000) {
                key *= 100000;
            } else if (data[i] <= 1000000) {
                key *= 1000000;
            } else if (data[i] <= 10000000) {
                key *= 10000000;
            }
            key += Math.abs(data[i]);
        }
        int[] found = CONFIGS.get(key);
        if (found != null) {
            return found;
        } else {
            CONFIGS.put(key, data);
            return data;
        }
    }

    //TRACKED COMPUTATION :
    //=========================
    public Tsr(Tsr tensor, String operation) {
        if (tensor == null) {
            return;
        }
        _construct(new Tsr[]{tensor}, operation, true);
    }

    public Tsr(Tsr[] tensors, String operation) {
        _construct(tensors, operation, true);
    }

    public Tsr(Tsr[] tensors, String operation, boolean doAD) {
        _construct(tensors, operation, doAD);
    }

    private void _construct(Tsr[] tensors, String operation, boolean doAD) {
        if (tensors == null || tensors.length == 0 || tensors[0] == null) {
            return;
        }
        Tsr result = Function.setup.commit(tensors, operation, doAD);
        boolean resultIsUnique = true;
        for(Tsr t : tensors){
            if(t == result){
                resultIsUnique = false;
                break;
            }
        }
        if(resultIsUnique){
            this.inject(result);
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //MODIFICATION :
    //=========================
    public int[] idx() {
        return (this.has(int[].class) ? (int[]) find(int[].class) : null);
    }
    public Tsr idx(int[] newIdx) {
        if(newIdx==null){
            throw new IllegalArgumentException("[Tsr][idx(int[] newIdx)]: Invalid argument! Index must not be null!");
        } else if(newIdx.length!=this.rank()){
            throw new IllegalArgumentException("[Tsr][idx(int[] newIdx)]: Invalid argument! Index rank must match shape rank!");
        }
        for(int i=0; i<this.rank(); i++){
            if(newIdx[i]<=this.shape()[i]){
                throw new IllegalArgumentException("[Tsr][idx(int[] newIdx)]: Invalid argument! Index at index "+
                        i+" is violates ( newIndex[i] <= shape[i] )!");
            }
        }
        this.add(newIdx);
        return this;
    }

    public Tsr inject(Tsr tensor) {
        _value = tensor._value;
        _shape = tensor._shape;
        _idxmap = tensor._idxmap;
        _translation = tensor._translation;
        _components = tensor._components;
        _flags = tensor._flags;
        if(tensor.isOutsourced()){
            Device device = (Device) tensor.find(Device.class);
            device.swap(tensor, this);
        }
        return this;
    }

    public Tsr backward(Tsr error) {
        if(this.isOutsourced()){
            this.device().add(error);
        }
        if (this.rqsGradient()) {
            this.addToGradient(error);
        }
        if (this.has(GraphNode.class)) {
            ((GraphNode) this.find(GraphNode.class)).backward(error);
        }
        return this;
    }

    public Tsr delete() {
        //if(isUndefined()){
        //    return this;//THIS SHOULD NOT BE CALLED!! TODO RESOLVE!
        //}
        if (this.isOutsourced()) {
            ((Device) this.find(Device.class)).rmv(this);
        }
        GraphNode node =((GraphNode)this.find(GraphNode.class));
        if(node != null){
            if(!node.isVirtual() && node.isUsedAsDerivative()){
                throw new IllegalStateException("Trying to delete a tensor which is part of a function graph and used as derivative!");
            }
            node.deathBy(node);
        }
        _flags = -1;
        _value = null;
        _shape = null;
        _translation = null;
        _idxmap = null;
        _components = null;
        _gradient = null;
        return this;
    }

    //TENSOR OPERATION (OVERLOADABLE):
    //=================================
    public Tsr plus(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0+i1");
    }
    public Tsr minus(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0-i1");
    }
    public Tsr multiply(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0*i1");
    }
    public Tsr div(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0/i1");
    }
    public Tsr mod(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0%i1");
    }
    public Tsr power(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0^i1");
    }
    public Tsr xor(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0^i1");
    }
    public boolean equals(Tsr other) {
        return (this.hashCode()==other.hashCode());
    }
    public Tsr getAt(Object key) {
        key = (key instanceof  ArrayList)?((ArrayList)key).toArray():key;
        key = intArray((Object[]) key);
        Tsr subset = new Tsr();
        if(key instanceof int[]) {
            int[] idx = (int[])key;
            boolean containsInv = false;
            for(int i=0; i<this.rank(); i++){
                if(idx[i]<0) containsInv = true;
            }
            int[] newShape = new int[this.rank()];//_shape;
            if(containsInv) {
                newShape = new int[this.rank()];
                for(int i=0; i<this.rank(); i++) {
                    if(idx[i]>=0) {
                        newShape[i] = _shape[i]-idx[i];
                    } else {
                        newShape[i] = _shape[i]+idx[i];
                        idx[i] = 0;//_shape[i]+idx[i];
                    }
                }
            }
            subset._value = this._value;
            subset._translation = this._translation;
            subset._idxmap = cached(fcn.indexing.idxTln(newShape));
            if(this.isOutsourced()){
                Device device = (Device) this.find(Device.class);
                device.add(subset, this);
            }
            subset._shape = cached(newShape);
            subset.add(idx);

        }
        return subset;
    }

    //ELEMENTARY OPERATIONS:
    //=========================
    public Tsr foreach(Consumer<Integer> action) {
        this.setIsVirtual(false);
        int sze = this.size();
        int[] idx = new int[this.shape().length];
        for (int i = 0; i < sze; i++) {
            fcn.indexing.increment(idx, this.shape());
            action.accept(fcn.indexing.i_of_i(i,this));//fcn.indexing.iOf(idx, this.translation())
        }
        return this;
    }

    public Tsr foreach(BiConsumer<Integer, Double> action) {
        this.setIsVirtual(false);
        int sze = this.size();
        int[] idx = new int[this.shape().length];
        double[] value = this.targetValue64();
        for (int i = 0; i < sze; i++) {
            fcn.indexing.increment(idx, this.shape());
            int index = fcn.indexing.i_of_i(i,this);
            action.accept(index, value[index]);
        }
        return this;
    }

    /**
     * ======================================================================================================
     * STATIC FUNCTIONS:
     */
    public static class fcn
    {
        public static class io
        {
            public static double getFrom(Tsr t, int i) {
                if (t.isEmpty() || t.isUndefined()) {
                    return 0;
                } else if (t.isVirtual()) {
                    return t.targetValue64()[0];
                }
                return t.targetValue64()[indexing.i_of_i(i,t)];
            }

            public static double getFrom(Tsr t, int[] idx) {
                t.setIsVirtual(false);
                return t.targetValue64()[indexing.iOf(idx, t.translation())];
            }

            public static void setInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.targetValue64()[indexing.i_of_i(i,t)] = value;
            }

            public static void setInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.targetValue64()[indexing.iOf(idx, t.translation())] = value;
            }

            public static void addInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.targetValue64()[indexing.i_of_i(i,t)] += value;
            }

            public static void addInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.targetValue64()[indexing.iOf(idx, t.translation())] += value;
            }

            public static Tsr addInto(Tsr t, Tsr source) {
                if (t.isVirtual() && source.isVirtual()) {
                    t.targetValue64()[0] += ((source.gradientIsTargeted())?source.gradient64():source.value64())[0];
                } else {
                    if (t.isVirtual()) {
                        t.setIsVirtual(false);
                    }
                    int[] index = new int[t.shape().length];
                    int size = t.size();
                    for (int i = 0; i < size; i++) {
                        addInto(t, index, getFrom(source, index));
                        indexing.increment(index, t.shape());
                    }
                }
                return source;
            }

            public static void subInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.targetValue64()[indexing.i_of_i(i,t)] -= value;
            }

            public static void subInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.targetValue64()[indexing.iOf(idx, t.translation())] -= value;
            }

            public static void subInto(Tsr t, Tsr source) {
                if (t.isVirtual() && source.isVirtual()) {
                    t.targetValue64()[0] -= ((source.gradientIsTargeted())?source.gradient64():source.value64())[0];
                } else {
                    if (t.isVirtual()) {
                        t.setIsVirtual(false);
                    }
                    int[] index = new int[t.shape().length];
                    int size = t.size();
                    for (int i = 0; i < size; i++) {
                        io.subInto(t, index, io.getFrom(source, index));
                        indexing.increment(index, t.shape());
                    }
                }
            }

            public static void mulInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.targetValue64()[indexing.i_of_i(i,t)] *= value;
            }

            public static void mulInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.targetValue64()[indexing.iOf(idx, t.translation())] *= value;
            }

        }

        public static class exec {

            public static Tsr reshaped(Tsr tensor, int[] newForm, boolean newTsr) {
                tensor = (newTsr)? create.cpyOf(tensor):tensor;
                //tensor._record(tensor.shape(), tensor.translation());
                tensor._shape = cached(indexing.shpCheck(indexing.rearrange(tensor._shape, newForm), tensor));
                tensor._translation = cached(indexing.rearrange(tensor._translation, tensor._shape, newForm));
                tensor._idxmap =  cached(indexing.idxTln(tensor._shape));
                return tensor;
            }
            //OPERATIONS:
            //=========================

        }

        public static class create
        {
            private static Tsr _newEmptyLike(Tsr template){
                Tsr t = new Tsr();
                t._shape = template._shape;
                t._idxmap = template._idxmap;
                t._translation = template.translation();
                return t;
            }

            public static Tsr newTsrLike(Tsr template, double value){
                Tsr t = _newEmptyLike(template);
                if(template.is32()){
                    t.setTargetValue((float)value);
                } else {
                    t.setTargetValue(value);
                }
                if(template.isOutsourced()){
                    ((Device)template.find(Device.class)).add(t);
                }
                return t;
            }

            public static Tsr newTsrLike(Tsr template){//The output tensor will not have gradients!
                Tsr t = _newEmptyLike(template);
                if(template.is32()){
                    t.setTargetValue32(new float[template.size()]);
                } else {
                    t.setTargetValue64(new double[template.size()]);
                }
                if(template.isOutsourced()){
                    ((Device)template.find(Device.class)).add(t);
                }
                return t;
            }


            public static Tsr newTsr(double value, int[] shape) {
                int sze = indexing.szeOfShp(shape);
                Tsr tensor = new Tsr();
                tensor._value = new double[sze];
                tensor.initialShape(shape);
                for (int i = 0; i < sze; i++) {
                    ((double[])tensor._value)[i] = value;
                }
                return tensor;
            }

            public static Tsr newTsr(double[] value, int[] shape) {
                Tsr tensor = new Tsr();
                tensor._value = value;
                tensor.initialShape(shape);
                return tensor;
            }

            public static Tsr newTsr(int[] shape, int[] translation) {
                Tsr tensor = new Tsr();
                tensor._value = new double[indexing.szeOfShp(shape)];
                tensor.initialShape(shape);
                tensor._translation = (translation != null) ? translation : tensor._translation;//FUNCTIONS.put()
                return tensor;
            }

            public static Tsr cpyOf(Tsr tensor) {
                Tsr newTensor = new Tsr();
                newTensor._shape = tensor._shape;
                newTensor._translation = tensor._translation;
                newTensor._idxmap = tensor._idxmap;
                newTensor._value = (tensor.is64())?new double[tensor.size()]:new float[tensor.size()];
                newTensor._components = null;//tensor._components;
                newTensor._flags = tensor._flags;
                if(tensor.is64()){
                    int length = (tensor.is64())?((double[])tensor._value).length:((float[])tensor._value).length;
                    for (int i = 0; i < length; i++) {
                        if(tensor.is64()){
                            double[] value = (double[])tensor._value;
                            ((double[])newTensor._value)[i] = value[i];
                        }else {
                            float[] value = (float[])tensor._value;
                            ((float[])newTensor._value)[i] = value[i];
                        }
                    }
                }
                if (tensor.isOutsourced()) {
                    newTensor.add(tensor.device());
                }
                return newTensor;
            }

            public static Tsr cpyOf(Object[] things) {
                for (int i = 0; i < things.length; i++) {
                    if (things[i] instanceof int[]) {

                    } else if (things[i] instanceof double[]) {

                    }
                }
                return new Tsr();
            }

        }

        public static double[] newDoubleArray(double value, int size){
            double[] array = new double[size];
            for(int i=0; i<size; i++){
                array[i] = value;
            }
            return array;
        }

        public static float[] newFloatArray(float value, int size){
            float[] array = new float[size];
            for(int i=0; i<size; i++){
                array[i] = value;
            }
            return array;
        }

        public static void inject(double[] data, boolean grd, Tsr tensor) {
            if (grd) {
                tensor._gradient = data;
            } else {
                tensor._value = data;
            }
        }

        public static class stringify{
            @Contract(pure = true)
            public static String formatFP(double v){
                DecimalFormatSymbols formatSymbols = new DecimalFormatSymbols(Locale.US);
                DecimalFormat Formatter = new DecimalFormat("##0.0##E0", formatSymbols);
                String vStr = String.valueOf(v);
                if(vStr.length()>7){
                    if(vStr.substring(0, 2).equals("0.")){
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
            public static String str(int[] shp) {
                String str = "";
                int i = 0;
                for (int s : shp) {
                    str += s + ((i != shp.length - 1) ? ", " : "");
                    i++;
                }
                return "[" + str + "]";
            }

        }

        /**
         * ======================================================================================================
         * INDEXING FUNCTIONS:
         */
        public static class indexing
        {
            @Contract(pure = true)
            public static int i_of_i(int i, Tsr t){
                int[] idx = new int[t._shape.length];
                if(t.has(int[].class)){
                    int[] baseIdx = (int[])t.find(int[].class);
                    for(int ii=0; ii<baseIdx.length; ii++){
                        idx[ii] = baseIdx[ii];
                    }
                }
                for(int ii=t.rank()-1; ii>=0; ii--){
                    idx[ii] += i/t._idxmap[ii];
                    i %= t._idxmap[ii];
                }
                for(int ii=0; ii<t._shape.length; ii++){
                    i += idx[ii]*t._translation[ii];
                }
                return i;
            }
            /*
                1, 2, 3, 4, =>* 2, 3, 4,
                5, 6, 7, 8, =>* 6, 7, 8,
                9, 1, 2, 3, =>* 1, 2, 3,
                4, 5, 6, 7, =>* 5, 6, 7,
                8, 9, 1, 2,
                3, 4, 5, 6
             */

            @Contract(pure = true)
            public static void increment(@NotNull int[] shpIdx, @NotNull int[] shape) {
                int i = 0;
                while (i >= 0 && i < shape.length) {//fixed
                    i = incrementAt(i, shpIdx, shape);
                }
            }

            @Contract(pure = true)
            public static int incrementAt(int i, @NotNull int[] shpIdx, @NotNull int[] shape) {
                if (shpIdx[i] < (shape[i])) {//fixed
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
            }

            @Contract(pure = true)
            public static void decrement(@NotNull int[] shpIdx, @NotNull int[] shape) {
                int i = 0;
                while (i >= 0 && i < shape.length) {
                    i = decrementAt(i, shpIdx, shape);
                }
            }

            @Contract(pure = true)
            public static int decrementAt(int i, @NotNull int[] shpIdx, @NotNull int[] shape) {
                if (shpIdx[i] == 0) {
                    i++;
                } else {
                    shpIdx[i]--;
                    i--;
                    while (shpIdx[i] == 0) {
                        shpIdx[i] = shape[i] - 1;
                        i--;
                    }
                    i = -1;
                }
                return i;
            }

            @Contract(pure = true)
            public static void incrementFor(int count, int[] shpIdx, int[] shape) {
                for (int Di = 0; Di < count; Di++) {
                    increment(shpIdx, shape);
                }
            }

            @Contract(pure = true)
            public static void decrementFor(int count, int[] shpIdx, int[] shape) {
                for (int Di = 0; Di < count; Di++) {
                    decrement(shpIdx, shape);
                }
            }

            @Contract(pure = true)
            public static int[] idxTln(int[] shape) {
                int[] tln = new int[shape.length];
                int prod = 1;
                for (int i = 0; i < tln.length; i++) {
                    tln[i] = prod;
                    prod *= shape[i];
                }
                return tln;
            }

            @Contract(pure = true)
            public static int[] idxOf(int i, int[] tln) {
                int[] idx = new int[tln.length];
                for (int ti = tln.length - 1; ti >= 0; ti--) {
                    int r = i % tln[ti];
                    idx[ti] = (i - r) / tln[ti];
                    i = r;
                }
                return idx;
            }

            @Contract(pure = true)
            public static int[] rearrange(int[] array, @NotNull int[] ptr) {
                int[] newShp = new int[ptr.length];
                for (int i = 0; i < ptr.length; i++) {
                    if (ptr[i] < 0) {
                        newShp[i] = Math.abs(ptr[i]);
                    } else if (ptr[i] >= 0) {
                        newShp[i] = array[ptr[i]];
                    }
                }
                return newShp;
            }

            @Contract(pure = true)
            public static int[] shpCheck(int[] newShp, Tsr t) {
                if (szeOfShp(newShp) != t.size()) {
                    throw new IllegalArgumentException("New _shape does not match tensor size!" +
                            " (" + stringify.str(newShp) + ((szeOfShp(newShp) < t.size()) ? "<" : ">") + stringify.str(t.shape()) + ")");
                }
                return newShp;
            }

            @Contract(pure = true)
            public static int[] rearrange(int[] tln, int[] shp, @NotNull int[] newForm) {
                int[] shpTln = idxTln(shp);
                int[] newTln = new int[newForm.length];
                for (int i = 0; i < newForm.length; i++) {
                    if (newForm[i] < 0) {
                        newTln[i] = shpTln[i];
                    } else if (newForm[i] >= 0) {
                        newTln[i] = tln[newForm[i]];
                    }
                }
                return newTln;
            }

            @Contract(pure = true)
            private static String strInt(int[] array) {
                String S = "";
                for (int i = 0; i < array.length; i++) {
                    S += "[" + array[i] + "]";
                }
                return S;
            }

            @Contract(pure = true)
            public static int[] shpOfCon(int[] shp1, int[] shp2) {
                int[] shape = new int[(shp1.length + shp2.length) / 2];
                for (int i = 0; i < shp1.length && i < shp2.length; i++) {
                    shape[i] = Math.abs(shp1[i] - shp2[i]) + 1;
                }
                return shape;
            }

            @Contract(pure = true)
            public static int iOf(int[] idx, int[] tln) {
                int i = 0;
                for (int ii = 0; ii < tln.length; ii++) {
                    i += idx[ii] * tln[ii];
                }
                return i;
            }

            @Contract(pure = true)
            public static int szeOfShp(int[] shape) {
                int size = 1;
                for (int Di = 0; Di < shape.length; Di++) {
                    size *= shape[Di];
                }
                return size;
            }

            /**-----------------------------------------
             *
             * 	[2][1][4][6][8][6]
             * 	[3][-1][2][-4][-5][6]//[0][0][0]=>3 times othr
             * 	is really:
             * 	[4  ][-2][1  ][-6][-8][6  ]
             *	   	   |	    |   |
             * 	[  6][ 7][  2][ 1][ 4][  8] <= then multiplying with this
             * 	[4|6][ 6][1|2][ 6][ 5][6|8]
             *
             *  a > b -> c = (a-b)+1
             *  a < b -> c = (b-a)+1
             *
             * */


        }

    }


}
