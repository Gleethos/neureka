package neureka.core;

import neureka.core.device.IDevice;
import neureka.core.device.aparapi.AparapiDevice;
import neureka.core.function.IFunction;
import neureka.core.function.factory.autograd.GraphNode;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Locale;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class Tsr {

    // DEFAULT DEVICE (HOST CPU)
    //=========================
    private static AparapiDevice CPU;
    //OPenClDevice!!!!!!!

    //STATIC FUNCTIONS MEMORY:
    //=========================
    private static HashMap<Long, int[]> CONFIGS;

    static {
        CONFIGS = new HashMap<>();//The things we do for memory
        CPU = new AparapiDevice(null);//<= creates CPU-Aparapi-Kernel
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
    private Object _value, _gradient;//double[]
    //-----------------------------------------------------------------------

    public AparapiDevice device() {
        if (this.isOutsourced()) {
            return (AparapiDevice) this.find(AparapiDevice.class);
        }
        return CPU;
    }

    public double[] targetValue(boolean unique){
        if(!this.isOutsourced() && unique){
            double[] value = new double[this.size()];
            double[] v = (gradientIsTargeted())?gradient():value();
            for(int i=0; i<value.length; i++){
                value[i] = v[i];
            }
            return value;
        } else {
            return (gradientIsTargeted())?gradient():value();
        }
    }

    public double[] targetValue(){
        return (gradientIsTargeted())?gradient():value();
    }

    public Tsr setTargetValue(double[] value){
        if(this.isOutsourced()){
            ((AparapiDevice) this.find(AparapiDevice.class)).overwrite(this, value);
        } else {
            if(this.gradientIsTargeted()){
                _gradient = value;
            } else {
                _value = value;
            }
        }
        return this;
    }

    public double[] gradient() {
        if (this.rqsGradient() && this.isOutsourced() && this.has(AparapiDevice.class)) {
            return ((AparapiDevice) find(AparapiDevice.class)).valueOf(this, true);
        }
        return (this.isFP64())?(double[])_gradient:fcn.io.floatToDouble((float[])_gradient);
    }
    public float[] gradientFP32(){
        return (this.isFP64())?Tsr.fcn.io.doubleToFloat((double[])_gradient):(float[])_gradient;
    }

    public Tsr addToGradient(Tsr g) {
        if(this.isOutsourced()){
            AparapiDevice device = (AparapiDevice) this.find(AparapiDevice.class);
            this.setGradientIsTargeted(true);
            device.add(g);
            device.execute(new Tsr[]{this, g}, IFunction.TYPES.LOOKUP.get("<"), -1);
            device.get(g);
            this.setGradientIsTargeted(false);
        } else {
            if(this.isFP64()){
                double[] value = g.value();
                _gradient = (_gradient==null)?new double[value.length]:_gradient;
                for(int i=0; i<value.length; i++){
                    ((double[])_gradient)[i] = value[i];
                }
            } else {
                float[] value = g.valueFP32();
                _gradient = (_gradient==null)?new float[value.length]:_gradient;
                for(int i=0; i<value.length; i++){
                    ((float[])_gradient)[i] = value[i];
                }
            }
        }
        return this;
    }

    public boolean isFP64(){
        return _value instanceof double[];
    }

    public double[] value() {
        if (_value == null && this.isOutsourced() && this.has(IDevice.class)) {
            return ((IDevice) this.find(IDevice.class)).valueOf(this, false);
        }
        double[] newValue = (this.isFP64())?(double[])_value:fcn.io.floatToDouble((float[])_value);
        if (this.isVirtual()) {
            newValue = new double[this.size()];
            double[] value = (this.isFP64())?(double[])_value:fcn.io.floatToDouble((float[])_value);
            for (int i = 0; i < newValue.length; i++) {
                newValue[i] = value[0];
            }
        }
        return newValue;
    }

    public float[] valueFP32(){
        if (_value == null && this.isOutsourced() && this.has(IDevice.class)) {
            return ((IDevice) this.find(IDevice.class)).floatValueOf(this, false);
        }
        float[] newValue = (this.isFP64())?Tsr.fcn.io.doubleToFloat((double[])_value):(float[])_value;
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
            ((AparapiDevice) this.find(AparapiDevice.class)).add(this);
        }
        return this;
    }

    public int[] shape() {
        return _shape;
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
        return (this.isOutsourced())
                ? fcn.indexing.szeOfShp(this.shape())
                : (this.isVirtual()
                ? fcn.indexing.szeOfShp(this.shape())
                : ((this.isFP64())?((double[])_value).length:((float[])_value).length));
    }

    public int[] shpIdx(int idx) {
        return fcn.indexing.idxOf(idx, _translation);
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
                    ((AparapiDevice)find(AparapiDevice.class)).get(this);
                    _gradient = null;
                    ((AparapiDevice)find(AparapiDevice.class)).add(this);
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
        } else if (this.has(AparapiDevice.class)) {
            AparapiDevice device = (AparapiDevice) this.find(AparapiDevice.class);
            if (device.has(this)) {
                device.get(this);
            }
            this.remove(AparapiDevice.class);
        }
        return this;
    }
    //---
    public boolean isVirtual() {
        return (_flags & IS_VIRTUAL_MASK) == IS_VIRTUAL_MASK;
    }

    public Tsr setIsVirtual(boolean isVirtual) {
        if (isVirtual() != isVirtual) {
            double v = (((this.isFP64())?((double[])_value)[0]:((float[])_value)[0]));
            if (isVirtual) {
                _value = new double[]{v};
                _flags += IS_VIRTUAL_MASK;
            } else {
                _value = (this.isFP64())?new double[this.size()]:new float[this.size()];
                int length = (this.isFP64())?((double[])_value).length:((float[])_value).length;
                for (int i = 0; i < length; i++) {
                    if(this.isFP64()){
                        ((double[])_value)[i] = v;
                    } else {
                        ((float[])_value)[i] = (float)v;
                    }
                }
                _flags -= IS_VIRTUAL_MASK;
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
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //GENERIC PROPERTIES :
    //=========================
    public boolean belongsToGraph() {
        return this.has(GraphNode.class);
    }

    public boolean isLeave() {
        return (!this.has(GraphNode.class)) || ((GraphNode) this.find(GraphNode.class)).isOrigin();
    }

    public boolean isBranch() {
        return !this.isLeave();
    }

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //DISPLAY :
    //=========================
    public String toString(String mode) {
        if (this.isEmpty()) {
            return "empty";
        } else if (this.isUndefined()) {
            return "undefined";
        }
        String strShape = "";
        for (int i = 0; i < _shape.length; i++) {
            strShape += _shape[i];
            if (i < _shape.length - 1) {
                strShape += "x";
            }
        }
        boolean compact = mode.contains("c");
        strShape = "[" + strShape + "]";
        String asString = "";
        asString += _stringified((value()), compact);//(this.isOutsourced())?this.value():_value
        asString = strShape + ":(" + asString + ")";
        if(mode.contains("g")){
            if(this.rqsGradient()){
                asString += ":g:";
                double[] gradient = this.gradient();
                if(gradient!=null){
                    asString += "("+_stringified((gradient()), compact)+")";
                } else {
                    asString += "(null)";
                }
            }
        }
        if (mode.contains("r")) {
            if (this.has(GraphNode.class) && ((GraphNode) this.find(GraphNode.class)).size() > 0) {
                GraphNode node = (GraphNode) this.find(GraphNode.class);
                String[] relatives = {"; "};
                node.forEach((t, g) -> relatives[0] += "=>d|[ " + g.toString(mode) + " ]|:t{ " + t.toString(mode) + " }, ");
                asString += relatives[0];
            }
        }
        if (mode.contains("d")) {
            if (this.has(GraphNode.class) && ((GraphNode) this.find(GraphNode.class)).size() > 0) {
                GraphNode node = (GraphNode) this.find(GraphNode.class);
                if (node.mode() != 0) {
                    String[] relatives = {"; "};
                    node.forEach((target, derivative) -> relatives[0] += "->d" + derivative.toString(mode) + ", ");
                    asString += relatives[0];
                }
            }
        }
        return asString;
    }

    private String _stringified(double[] v, boolean format){
        String asString = "";
        int size = (this.isVirtual() ? this.size() : v.length);
        int trim = (size-50);
        size = (trim>0)?50:size;
        for (int i = 0; i < size; i++) {
            String vStr;
            if(format){
                vStr = fcn.stringify.formatFP(v[(this.isVirtual()) ? 0 : i]);
            } else {
                vStr = String.valueOf(v[(this.isVirtual()) ? 0 : i]);
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

    public Tsr(Object[] args){
        _construct(args);
    }

    private void _construct(Object[] array){
        boolean containsString = false;
        int numberOfTensors = 0;
        ArrayList<Tsr> list = new ArrayList<>();
        for(Object o : array){
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
        for(Object o : array){
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
        _value = value;
        initialShape(shape);
    }

    /**
     * @param tensor
     */
    public Tsr(Tsr tensor) {
        _shape = tensor._shape;
        _translation = tensor._translation;
        _idxmap = tensor._idxmap;
        _value = (this.isFP64())?new double[tensor.size()]:new float[tensor.size()];
        _components = null;//tensor._components;
        _flags = tensor._flags;
        int length = (this.isFP64())?((double[])_value).length:((float[])_value).length;
        for (int i = 0; i < length; i++) {
            if(tensor.isFP64()){
                ((double[])_value)[i] = ((double[])tensor._value)[i];
            } else {
                ((float[])_value)[i] = ((float[])tensor._value)[i];
            }
        }
    }

    /**
     * @param newShape
     * @return
     */
    public Tsr initialShape(int[] newShape) {
        int size = fcn.indexing.szeOfShp(newShape);
        _value = (_value==null)?new double[size]:_value;
        int length = (this.isFP64())?((double[])_value).length:((float[])_value).length;
        if (size != length && !this.isVirtual()) {
            throw new IllegalArgumentException("Size of shape does not match stored value!");
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
        Tsr result = IFunction.setup.commit(tensors, operation, doAD);
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
    public int[][] config() {
        return (this.has(int[][].class) ? (int[][]) find(int[][].class) : new int[][]{this.shape(), this.translation()});
    }

    public int[] shape(int i) {
        int[][] conf = this.config();
        i = Math.abs(i) * 2 + 0;
        if (i < conf.length) {
            return conf[i];
        }
        return null;
    }

    public int[] translation(int i) {
        int[][] conf = this.config();
        i = Math.abs(i) * 2 + 1;
        return (i < conf.length) ? conf[i] : null;
    }

    private void _record(int[] shp, int[] tln) {
        int[][] conf = this.config();
        int sze = (conf == null) ? 0 : conf.length;
        int[][] newConf = new int[sze + 2][];
        newConf[0] = shp;
        newConf[1] = tln;
        for (int i = 2; i < newConf.length; i++) {
            newConf[i] = conf[i - 2];
        }
        this.add(newConf);
    }

    public Tsr inject(Tsr tensor) {
        _value = tensor._value;
        _shape = tensor._shape;
        _translation = tensor._translation;
        _components = tensor._components;
        _flags = tensor._flags;
        if(tensor.isOutsourced()){
            AparapiDevice device = (AparapiDevice) tensor.find(AparapiDevice.class);
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
        if (this.isOutsourced()) {
            ((AparapiDevice) this.find(AparapiDevice.class)).rmv(this);
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

    //ELEMENTARY OPERATIONS:
    //=========================
    public Tsr foreach(Consumer<Integer> action) {
        this.setIsVirtual(false);
        int sze = this.size();
        int[] idx = new int[this.shape().length];
        for (int i = 0; i < sze; i++) {
            fcn.indexing.increment(idx, this.shape());
            action.accept(fcn.indexing.i_of_i(i,_shape, _translation, _idxmap));//fcn.indexing.iOf(idx, this.translation())
        }
        return this;
    }

    public Tsr foreach(BiConsumer<Integer, Double> action) {
        this.setIsVirtual(false);
        int sze = this.size();
        int[] idx = new int[this.shape().length];
        double[] value = this.targetValue();
        for (int i = 0; i < sze; i++) {
            fcn.indexing.increment(idx, this.shape());
            int index = fcn.indexing.i_of_i(i,_shape, _translation, _idxmap);
            action.accept(index, value[index]);
        }
        return this;
    }

    /**
     * ======================================================================================================
     * STATIC FUNCTIONS:
     */
    public static class fcn {

        public static class io
        {
            public static float[] doubleToFloat(double[] data){
                float[] newData = new float[data.length];
                for(int i=0; i<data.length; i++){
                    newData[i] = (float)data[i];
                }
                return newData;
            }

            public static double[] floatToDouble(float[] data){
                if(data==null){
                    return null;
                }
                double[] newData = new double[data.length];
                for(int i=0; i<data.length; i++){
                    newData[i] = (double)data[i];
                }
                return newData;
            }


            public static double getFrom(Tsr t, int i) {
                if (t.isEmpty() || t.isUndefined()) {
                    return 0;
                } else if (t.isVirtual()) {
                    return t.targetValue()[0];
                }
                return t.targetValue()[indexing.i_of_i(i,t._shape, t._translation, t._idxmap)];
            }

            public static double getFrom(Tsr t, int[] idx) {
                t.setIsVirtual(false);
                return t.targetValue()[indexing.iOf(idx, t.translation())];
            }

            public static void setInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.targetValue()[indexing.i_of_i(i,t._shape, t._translation, t._idxmap)] = value;
            }

            public static void setInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.targetValue()[indexing.iOf(idx, t.translation())] = value;
            }

            public static void addInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.targetValue()[indexing.i_of_i(i,t._shape, t._translation, t._idxmap)] += value;
            }

            public static void addInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.targetValue()[indexing.iOf(idx, t.translation())] += value;
            }

            public static Tsr addInto(Tsr t, Tsr source) {
                if (t.isVirtual() && source.isVirtual()) {
                    t.targetValue()[0] += ((source.gradientIsTargeted())?source.gradient():source.value())[0];
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
                t.targetValue()[indexing.i_of_i(i,t._shape, t._translation, t._idxmap)] -= value;
            }

            public static void subInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.targetValue()[indexing.iOf(idx, t.translation())] -= value;
            }

            public static void subInto(Tsr t, Tsr source) {
                if (t.isVirtual() && source.isVirtual()) {
                    t.targetValue()[0] -= ((source.gradientIsTargeted())?source.gradient():source.value())[0];
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
                t.targetValue()[indexing.i_of_i(i,t._shape, t._translation, t._idxmap)] *= value;
            }

            public static void mulInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.targetValue()[indexing.iOf(idx, t.translation())] *= value;
            }

        }

        public static class exec {

            public static Tsr reshaped(Tsr tensor, int[] newForm, boolean newTsr) {
                tensor = (newTsr)? cpyOf(tensor):tensor;
                tensor._record(tensor.shape(), tensor.translation());
                tensor._shape = cached(indexing.shpCheck(indexing.rearrange(tensor._shape, newForm), tensor));
                tensor._translation = cached(indexing.rearrange(tensor._translation, tensor._shape, newForm));
                tensor._idxmap =  cached(indexing.idxTln(tensor._shape));
                return tensor;
            }
            //OPERATIONS:
            //=========================

        }

        public static void inject(double[] data, boolean grd, Tsr tensor) {
            if (grd) {
                tensor._gradient = data;
            } else {
                tensor._value = data;
            }
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

        public static Tsr newTsr(double[] value, int[] shape, int[] translation) {
            Tsr tensor = new Tsr();
            tensor._value = value;
            tensor.initialShape(shape);
            tensor._translation = translation;
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
            newTensor._value = (tensor.isFP64())?new double[tensor.size()]:new float[tensor.size()];
            newTensor._components = null;//tensor._components;
            newTensor._flags = tensor._flags;
            if(tensor.isFP64()){
                int length = (tensor.isFP64())?((double[])tensor._value).length:((float[])tensor._value).length;
                for (int i = 0; i < length; i++) {
                    if(tensor.isFP64()){
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
            public static int i_of_i(int i, int[] shape, int[] translation, int[] idxmap){
                int[] idx = new int[shape.length];
                for(int ii=0; ii<shape.length; ii++){
                    idx[ii] = i/idxmap[ii];
                    i %= idxmap[ii];
                }
                for(int ii=0; ii<shape.length; ii++){
                    i += idx[ii]*translation[ii];
                }
                return i;
            }

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
