package neureka;

import neureka.acceleration.CPU;
import neureka.acceleration.Device;
import neureka.framing.Index;
import neureka.framing.Relation;
import neureka.calculus.Function;
import neureka.calculus.factory.assembly.FunctionBuilder;
import neureka.autograd.GraphNode;
import neureka.autograd.JITProp;
import neureka.optimization.Optimizer;
import neureka.utility.AbstractComponentOwner;
import neureka.utility.DataHelper;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.math.BigDecimal;
import java.text.DecimalFormat;
import java.text.DecimalFormatSymbols;
import java.util.*;
import java.util.concurrent.atomic.AtomicReference;

public class Tsr extends AbstractComponentOwner
{
    static {
        CONFIGS = new WeakHashMap<>();
        CPU = new CPU();
    }

    /**
     *  Default device (host cpu)
     */
    private static Device CPU;

    /**
     *  Cached configuration
     */
    private static Map<Long, int[]> CONFIGS;

    /**
     *  Value data fields
     */
    private int[] _shape, _translation, _idxmap;
    private Object _value;

    /**
     *  Flag Fields
     */
    private int _flags = 0;//Default
    private final static int RQS_GRADIENT_MASK = 1;
    private final static int IS_OUTSOURCED_MASK = 2;
    private final static int IS_VIRTUAL_MASK = 4;

    //-----------------------------------------------------------------------

    public boolean rqsGradient() {
        return (_flags & RQS_GRADIENT_MASK) == RQS_GRADIENT_MASK;
    }

    public Tsr setRqsGradient(boolean rqsGradient) {
        if (rqsGradient() != rqsGradient) {
            if (rqsGradient) {
                _flags += RQS_GRADIENT_MASK;
            } else {
                this.remove(Tsr.class);
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
            if (isOutsourced) _flags += IS_OUTSOURCED_MASK;
            else _flags -= IS_OUTSOURCED_MASK;
        }
        if (isOutsourced) {
            _value = null;
        } else if (
                !forComponent( Device.class, (d)->{
                    if (((Device)d).has(this)) ((Device)d).get(this);
                    this.remove(Device.class);
                    forComponent(Tsr.class, (gradient)->
                            ((Tsr) gradient).forComponent(Device.class, (gd)->{
                                if (((Device)gd).has((Tsr)gradient)) ((Device)gd).get((Tsr) gradient);
                                ((Tsr) gradient).remove(Device.class);
                            })
                    );
                }) && _value==null
        ){
            setIsVirtual(true);
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
                if (!isVirtual) _flags -= IS_VIRTUAL_MASK;
            } else {
                double v = (_value==null)?0:(((this.is64())?((double[])_value)[0]:((float[])_value)[0]));
                if (isVirtual) {
                    _value = new double[]{v};
                    _flags += IS_VIRTUAL_MASK;
                    Relation parent = (Relation)find(Relation.class);
                    if(parent!=null) parent.foreachChild((c)->c._value=_value);
                } else {
                    _value = (this.is64())?new double[this.size()]:new float[this.size()];
                    int length = (this.is64())?((double[])_value).length:((float[])_value).length;
                    for (int i = 0; i < length; i++) {
                        if(this.is64()) ((double[])_value)[i] = v;
                        else ((float[])_value)[i] = (float)v;
                    }
                    _flags -= IS_VIRTUAL_MASK;
                }
            }
        } else if(isVirtual && _value==null){
            _value = new double[]{0};
        }
        return this;
    }


    //-----------------------------------------------------------------------

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
    protected Object _add(Object newComponent){
        newComponent = (newComponent instanceof int[]) ? _cached((int[]) newComponent) : newComponent;
        if(newComponent instanceof Device){
            if(!((Device)newComponent).has(this)){
                ((Device)newComponent).add(this);
            }
        }
        return newComponent;
    }

    /**
     * @return the device on which this tensor is stored or null if it is not outsourced.
     */
    public Device device() {
        if (this.isOutsourced()) return (Device) this.find(Device.class);
        return CPU;
    }

    public Tsr setValue64(double[] value){
        if(this.isOutsourced()) ((Device) this.find(Device.class)).overwrite64(this, value);
        else _value = value;
        return this;
    }

    public Tsr setValue32(float[] value){
        if(this.isOutsourced()) ((Device) this.find(Device.class)).overwrite32(this, value);
        else _value = value;
        return this;
    }

    public Tsr setValue(Object value){
        if(value instanceof float[]) this.setValue32((float[])value);
        else if(value instanceof  double[]) this.setValue64((double[])value);
        else if(value instanceof Float) {
            this.setIsVirtual(true);
            if(this.is32()){
                ((float[])_value)[0] = ((Float)value).floatValue();
            } else {
                ((double[])_value)[0] = ((Float)value).doubleValue();
            }
        } else if(value instanceof Double){
            this.setIsVirtual(true);
            if(this.is64()){
                ((double[])_value)[0] = ((Double)value).doubleValue();
            } else {
                ((float[])_value)[0] = ((Double)value).floatValue();
            }
        }
        return this;
    }

    public Object getValue(){
        if(this.isOutsourced()){
            Device device = ((Device)find(Device.class));
            return (this.is32())?device.value32Of(this):device.value64Of(this);
        }
        return _value;
    }

    public double[] gradient64() {
        Tsr gradient = (Tsr)this.find(Tsr.class);
        if(gradient==null) return null;
        return (this.is32())?DataHelper.floatToDouble(gradient.value32()):gradient.value64();
    }

    public float[] gradient32(){
        Tsr gradient = (Tsr)this.find(Tsr.class);
        if(gradient==null) return null;
        return (this.is64())?DataHelper.doubleToFloat((double[])gradient.value64()):(float[])gradient.value32();
    }

    public Tsr addToGradient(Tsr error) {
        if(!forComponent(Tsr.class, (g)->{
            this.add(FunctionBuilder.build("I[0]<-(I[0]+I[1])", false).activate(new Tsr[]{(Tsr)g, error}));
        })){
            this.add(error).forComponent(Device.class, (d)->((Device)d).add(error));
        }
        return this;
    }

    public boolean is64(){
        return _value instanceof double[];
    }

    public  boolean is32(){
        return _value instanceof float[];
    }

    public Tsr to32() {
        if(this.is64()){
            Device device = (Device) this.find(Device.class);
            if(device!=null) device.get(this);
            _value = DataHelper.doubleToFloat((double[])_value);
            forComponent(Tsr.class, (g)->((Tsr)g).to32());
            if(device!=null) device.add(this);
        }
        return this;
    }

    public Tsr to64() {
        if(this.is32()){
            Device device = (Device) this.find(Device.class);
            if(device!=null) device.get(this);
            _value = DataHelper.floatToDouble((float[])_value);
            forComponent(Tsr.class, (g)->((Tsr)g).to64());
            if(device!=null) device.add(this);
        }
        return this;
    }

    public double value64(int i) {
        if(this.isVirtual()){
            if(this.is64()) return ((double[])_value)[0];
            else return ((float[])_value)[0];
        } else {
            if(this.is64()) return ((double[])_value)[i];
            else return ((float[])_value)[i];
        }
    }

    public double[] value64() {
        if (_value == null && this.isOutsourced() && this.has(Device.class)) {
            return ((Device) this.find(Device.class)).value64Of(this);
        }
        double[] newValue = (this.is64())?(double[])_value: DataHelper.floatToDouble((float[])_value);
        if (this.isVirtual() && newValue!=null) {
            newValue = new double[this.size()];
            double[] value = (this.is64())?(double[])_value:DataHelper.floatToDouble((float[])_value);
            for (int i = 0; i < newValue.length; i++) newValue[i] = value[0];
        }
        return newValue;
    }

    public float value32(int i) {
        if(this.isVirtual()){
            if(this.is64()) return (float) ((double[])_value)[0];
            else return ((float[])_value)[0];
        } else {
            if(this.is64()) return (float) ((double[])_value)[i];
            else return ((float[])_value)[i];
        }
    }

    public float[] value32() {
        if (_value == null && this.isOutsourced() && this.has(Device.class)) {
            return ((Device) this.find(Device.class)).value32Of(this);
        }
        float[] newValue = (this.is64())?DataHelper.doubleToFloat((double[])_value):(float[])_value;
        if (this.isVirtual() && newValue!=null) {
            newValue = new float[this.size()];
            for (int i = 0; i < newValue.length; i++) newValue[i] = newValue[0];
        }
        return newValue;
    }

    public Tsr setValue(double[] newValue) {
        _value = newValue;
        if (this.isOutsourced() && newValue != null) ((Device) this.find(Device.class)).add(this);
        return this;
    }

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
        return Util.Indexing.szeOfShp(this.shape());
    }

    public boolean isEmpty() {
        return _value == null && !this.isOutsourced();
    }

    public boolean isUndefined() {
        return _shape == null;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //GENERIC PROPERTIES :
    //=========================
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
            if (i < shape.length - 1) strShape += "x";
        }
        boolean compact = mode.contains("c");
        strShape = "[" + strShape + "]";
        if(mode.contains("shape")||mode.contains("shp")) return strShape;
        String asString = "";
        asString += _stringified((value64()), compact, max);//(this.isOutsourced())?this.value64():_value
        asString = strShape + ":(" + asString + ")";
        if(mode.contains("g")){
            if(this.rqsGradient()){
                asString += ":g:";
                Tsr gradient = (Tsr)this.find(Tsr.class);
                if(gradient!=null){
                    asString += gradient.toString("c").replace(strShape+":","");
                } else {
                    asString+="(null)";
                }
            }
        }
        if (mode.contains("r")) {
            if (this.has(GraphNode.class) && ((GraphNode) this.find(GraphNode.class)).size() > 0) {
                GraphNode node = (GraphNode) this.find(GraphNode.class);
                AtomicReference<String> enclosed = new AtomicReference<>("; ");
                node.forEachDerivative((t, d) -> {
                    if(d.derivative()==null){
                        enclosed.set(enclosed.get() +
                                "->d(null), ");
                    } else {
                        enclosed.set(enclosed.get() +
                                base+"=>d|[ " +
                                base+delimiter+    d.derivative()._toString(mode, deeper) + " " +
                                base+half+"]|:t{ " +
                                base+delimiter+    ((t.getPayload()!=null)?t.getPayload()._toString(mode, deeper):t.toString("")) + " " +
                                base+half+"}, ");
                    }
                });
                asString += enclosed.get();
            }
        }
        if (mode.contains("d")) {
            if (this.has(GraphNode.class) && ((GraphNode) this.find(GraphNode.class)).size() > 0) {
                GraphNode node = (GraphNode) this.find(GraphNode.class);
                if (node.mode() != 0) {//node.getMap().values().stream().coll
                    AtomicReference<String> enclosed = new AtomicReference<>("; ");
                    node.forEachDerivative((t, d) -> {
                        if(d.derivative()==null){
                            enclosed.set(enclosed.get() + "->d(null), ");
                        } else {
                            enclosed.set(enclosed.get() +
                                    "->d" + d.derivative()._toString(mode, deeper) + ", ");
                        }

                    });
                    asString += enclosed.get();
                }
            }
        }
        return asString;
    }

    private String _stringified(double[] v, boolean format, int max){
        String asString = "";
        int size = this.size();
        int trim = (size-max);
        size = (trim>0)?max:size;
        for (int i = 0; i < size; i++) {
            String vStr;
            if(format){
                vStr = Util.Stringify.formatFP(v[(this.isVirtual()) ? 0 : _i_of_i(i)]);
            } else {
                vStr = String.valueOf(v[(this.isVirtual()) ? 0 : _i_of_i(i)]);
            }
            asString += vStr;
            if (i < size - 1) asString += ", ";
            else if (trim > 0) asString += ", ... + "+trim+" more";
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
        if(arg1 instanceof List){
            if(arg2 instanceof String){
                if(((List)arg1).get(0) instanceof Integer){
                    List<Integer> shape = ((List)arg1);
                    int[] shp = new int[shape.size()];
                    for(int i=0; i<shp.length; i++) shp[i] = shape.get(i);
                    _construct(shp, (String)arg2);
                } else if(((List)arg1).get(0) instanceof Tsr){
                    _construct(((List<Tsr>)arg1).toArray(new Tsr[((List<Tsr>)arg1).size()]), (String)arg2, true);
                }
            } else if (arg2 instanceof List && ((List)arg2).get(0) instanceof Integer){
                List range = (List)arg2;
                List<Integer> shape = ((List)arg1);
                int[] shp = new int[shape.size()];
                for(int i=0; i<shp.length; i++) shp[i] = shape.get(i);
                double[] value = new double[Util.Indexing.szeOfShp(shp)];//TODO: type evaluation
                for(int i=0; i<value.length; i++) value[i] = (Integer)range.get(i%range.size());
                _construct(shp, value);
                return;
            }
        }
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

    public Tsr(int[] shape, String seed){
        _construct(shape, seed);
    }

    private void _construct(int[] shape, String seed){
        _construct(shape);
        //for(int i=0; i<this.size(); i++) ((double[])_value)[i] = DataHelper.getDoubleOf(DataHelper.longHash(seed+i));
        _value = DataHelper.seededDoubleArray((double[])_value, seed);
    }

    private int[] intArray(Object[] arg){
        int length = arg.length;
        int[] array = new int[length];
        for(int i=0; i<length; i++) array[i] = (Integer) arg[i];
        return array;
    }

    private double[] doubleArray(Object[] arg){
        int length = arg.length;
        double[] array = new double[length];
        for(int i=0; i<length; i++){
            if(arg[i] instanceof Integer) array[i] = (Integer) arg[i];
            else if(arg[i] instanceof Double) array[i] = (Double) arg[i];
            else if(arg[i] instanceof BigDecimal) array[i] = ((BigDecimal)arg[i]).doubleValue();
        }
        return array;
    }

    public Tsr(Object[] args){
        _construct(args);
    }

    private void _construct(Object[] args) {
        if(args==null || args.length==0)return;
        if(args[0] instanceof  Tsr && args.length==1){
            inject(Util.Create.newTsrLike((Tsr)args[0]));
            return;
        }
        args[0] = (args[0] instanceof ArrayList)?((ArrayList)args[0]).toArray():args[0];
        args[1] = (args[1] instanceof ArrayList)?((ArrayList)args[1]).toArray():args[1];
        if(args[0] instanceof Object[]){
            if(((Object[])args[0])[0] instanceof Integer){
                args[0] = intArray((Object[])args[0]);//array;
            } else {
                int length = ((Object[])args[0]).length;
                Tsr[] array = new Tsr[length];
                for(int i=0; i<length; i++) array[i] = (Tsr)((Object[])args[0])[i];
                args[0] = array;
            }
        }
        if(args[1] instanceof Object[]){
            if(((Object[])args[1])[0] instanceof Integer) args[1] = doubleArray((Object[]) args[1]);
            else if(((Object[])args[1])[0] instanceof BigDecimal) args[1] = doubleArray((Object[]) args[1]);
        }
        //CASES:
        if(args[0] instanceof int[] && (args[1] instanceof Double || args[1] instanceof Integer)){
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
                if(!list.contains(o)){
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
        _construct(shape);
    }

    private void _construct(int[] shape){
        _value = new double[Util.Indexing.szeOfShp(shape)];
        this.initialShape(shape);
    }

    public Tsr(int[] shape, double value) {
        _construct(shape, value);
    }

    private void _construct(int[] shape, double value){
        int size = Util.Indexing.szeOfShp(shape);
        _value = new double[1];
        this.setIsVirtual((size > 1));
        this.initialShape(shape);
        ((double[])_value)[0] = value;
    }

    public Tsr(int[] shape, double[] value) {
        _construct(shape, value);
    }

    private void _construct(int[] shape, double[] value) {
        _value = value;
        initialShape(shape);
    }

    /**
     * @param tensor which acts as template for this new tensor.
     */
    public Tsr(Tsr tensor, boolean cpy) {
        _value = (tensor.is64())?new double[tensor.size()]:new float[tensor.size()];
        _components = null;
        _flags = 0;
        int length = (tensor.is64())?((double[])_value).length:((float[])_value).length;
        if(cpy){
            if(tensor.is64()){
                double[] value = tensor.value64();
                if (length >= 0) System.arraycopy(value, 0, ((double[]) _value), 0, length);
            } else {
                float[] value = tensor.value32();
                if (length >= 0) System.arraycopy(value, 0, ((float[]) _value), 0, length);
            }
        }
        initialShape(tensor.shape());
    }

    /**
     * @param newShape
     * @return
     */
    public Tsr initialShape(int[] newShape) {
        int size = Util.Indexing.szeOfShp(newShape);
        _value = (_value==null)?new double[size]:_value;
        int length = (this.is64())?((double[])_value).length:((float[])_value).length;
        if (size != length && !this.isVirtual()) {
            throw new IllegalArgumentException("[Tsr][_iniShape]: Size of shape does not match stored value64!");
        }
        _shape = _cached(newShape);
        _translation = _cached(Util.Indexing.newTlnOf(newShape));
        _idxmap = _translation;
        return this;
    }

    private static int[] _cached(int[] data) {
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
        if (tensor == null) return;
        _construct(new Tsr[]{tensor}, operation, true);
    }

    public Tsr(Tsr[] tensors, String operation) {
        _construct(tensors, operation, true);
    }

    public Tsr(Tsr[] tensors, String operation, boolean doAD) {
        _construct(tensors, operation, doAD);
    }

    private void _construct(Tsr[] tensors, String operation, boolean doAD) {
        if (tensors == null || tensors.length == 0 || tensors[0] == null) return;
        Tsr result = Function.Setup.commit(this, tensors, operation, doAD);
        this.inject(result);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //MODIFICATION :
    //=========================


    public Tsr inject(Tsr tensor) {
        if(tensor==null) return this;
        _value = tensor._value;
        _shape = tensor._shape;
        _idxmap = tensor._idxmap;
        _translation = tensor._translation;
        _components = tensor._components;
        _flags = tensor._flags;
        if(_components!=null){//Inform components about their new owner:
            _components.forEach((c)->{if(c instanceof Component) ((Component)c).update(tensor, this);});
        }
        tensor._value = null;
        tensor._shape = null;
        tensor._idxmap = null;
        tensor._translation = null;
        tensor._components = null;
        tensor._flags = -1;
        return this;
    }

    /**
     *
     * @param error
     * @return
     */
    public Tsr backward(Tsr error) {
        if(!forComponent(GraphNode.class, (node)->((GraphNode)node).backward(error))){
            if(this.rqsGradient()) addToGradient(error);
        }
        return this;
    }

    public Tsr backward(double value){
        backward(new Tsr(_shape, value));
        return this;
    }

    public void applyGradient() {
        forComponent(JITProp.class, (jit)->((JITProp)jit).execute());
        forComponent(Tsr.class, (g)->{
            forComponent(Optimizer.class, (o)->((Optimizer)o).optimize((Tsr)g));
            remove(Tsr.class);
            FunctionBuilder.build("I[0]<-(I[0]+I[1])", false).activate(new Tsr[]{this, (Tsr)g});
        });
    }

    public Tsr delete() {
        forComponent(Device.class, (d)->((Device)d).rmv(this));
        forComponent(GraphNode.class, (n)->{
            if(((GraphNode)n).isUsedAsDerivative()){//&| !node.isVirtual()
                throw new IllegalStateException("Trying to delete a tensor which is part of a function graph and used as derivative!");
            }//n.extinguishLineageBy(node);
        });
        _flags = -1;
        _value = null;
        _shape = null;
        _translation = null;
        _idxmap = null;
        forComponent(Tsr.class, (g)->((Tsr)g).delete());
        _components = null;
        return this;
    }

    //TENSOR OPERATION (OVERLOADABLE):
    //=================================
    public Tsr T(){//Transposed!
        String operation = "";//TODO: make a static version of this which is always available
        for(int i=rank()-1; i>=0; i--) operation += (i+((i==0)?"":", "));
        operation = "["+operation+"]:(I[0])";
        return new Tsr(this, operation);
    }

    public Tsr plus(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0+i1");
    }
    public Tsr plus(Double value) {
        return plus(new Tsr(this.shape(), value));
    }
    public Tsr minus(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0-i1");
    }
    public Tsr negative(){
        return FunctionBuilder.build("(-1*I[0])", false).activate(new Tsr[]{this});
    }
    public Tsr multiply(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0*i1");
    }
    public Tsr multiply(Double value) {
        return multiply(new Tsr(this.shape(), value));
    }
    public Tsr div(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0/i1");
    }
    public Tsr div(Double value) {
        return div(new Tsr(this.shape(), value));
    }
    public Tsr mod(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0%i1");
    }
    public Tsr power(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0^i1");
    }
    public Tsr power(Double value){
        return power(new Tsr(this.shape(), value));
    }
    public Tsr xor(Tsr other) {
        return new Tsr(new Tsr[]{this, other}, "i0^i1");
    }
    public Tsr xor(Double value) {
        return xor(new Tsr(this.shape(), value));
    }
    public boolean equals(Tsr other) {
        return (this.hashCode()==other.hashCode());
    }

    public Tsr label(String[][] labels){
        Index index = (Index)find(Index.class);
        if(index==null){
            index = new Index(this.rank());
            add(index);
        }
        for(int i=0; i<labels.length; i++){
            if(labels[i]!=null){
                for(int ii=0; ii<labels[i].length; ii++){
                    if(labels[i][ii]!=null) index.set(labels[i][ii], i, ii);
                }
            }
        }
        return this;
    }
    public Tsr label(List<List> labels){
        Index index = (Index)find(Index.class);
        if(index==null){
            index = new Index(this.rank());
            add(index);
        }
        for(int i=0; i<labels.size(); i++){
            if(labels.get(i)!=null){
                for(int ii=0; ii<labels.get(i).size(); ii++){
                    if(labels.get(i).get(ii)!=null) index.set(labels.get(i).get(ii), i, ii);
                }
            }
        }
        return this;
    }

    public Tsr putAt(Object key, Tsr value){
        if(value.isEmpty()) throw new IllegalArgumentException("[Tsr][putAt(Object key, Tsr value)]: Value is empty!");
        Tsr slice = (key==null)?this:(Tsr)getAt(key);
        boolean valueIsDeviceVisitor = false;
        if(slice.isOutsourced() && !value.isOutsourced()){
            Device device = (Device)slice.find(Device.class);
            device.add(value);
            valueIsDeviceVisitor = true;
        }
        if(this.isEmpty() && slice.isEmpty() || slice.size()!=value.size()) inject(value);//Rethink this a little
        else new Tsr(new Tsr[]{slice, value}, "I[0]<-I[1]", false);
        if(valueIsDeviceVisitor) ((Device)value.find(Device.class)).get(value);
        return this;
    }
    public double getAt(int[] idx){
        return value64()[i_of_idx(idx)];
    }
    public Object getAt(Object key) {
        if(key==null) return this;
        if(key instanceof List) if(((List)key).size()==0) return this;
        int[] idxbase = null;
        int[] newShape = new int[this.rank()];
        if(key instanceof List) {
            key = ((List)key).toArray();
            boolean allInt = true;
            for(Object o : (Object[])key) allInt = allInt && o instanceof Integer;
            if(allInt) {
                key = intArray((Object[]) key);
                idxbase = (int[])key;
                if(key != null) {
                    for(int i=0; i<this.rank(); i++) idxbase[i] = (idxbase[i]<0)?_shape[i]+idxbase[i]:idxbase[i];
                    return Util.io.getFrom(this, idxbase);
                }
            } else {
                boolean hasScale = false;
                for(Object o : (Object[])key) hasScale = hasScale || o instanceof Map;
                idxbase = new int[((hasScale)?2:1)*this.rank()];
                Object[] ranges = (Object[])key;
                _configureSubsetFromRanges(ranges, idxbase, newShape, 0);
            }
        }//...not simple slice... Advanced:
        else if(key instanceof Map)// ==> i, j, k slicing!
        {
            idxbase = new int[this.rank()*2];
            Object[] ranges = ((Map)key).keySet().toArray();
            _configureSubsetFromRanges(ranges, idxbase, newShape, 0);
            Object[] steps = ((Map)key).values().toArray();
            for(int i=rank(); i<2*this.rank(); i++){
                idxbase[i] = (Integer)steps[i-rank()];
                newShape[i-rank()] /= (Integer)steps[i-rank()];
            }
        }
        Tsr subset = new Tsr();
        subset._value = this._value;
        subset._translation = this._translation;
        subset._idxmap = _cached(Util.Indexing.newTlnOf(newShape));
        subset._shape = _cached(newShape);
        if(idxbase.length==2*rank()){
            for(int i=rank(); i<idxbase.length; i++) idxbase[i] = (idxbase[i]==0)?1:idxbase[i];
        }
        subset.add(idxbase);
        if(this.isOutsourced()){
            Device device = (Device) this.find(Device.class);
            device.add(subset, this);
        }
        if(this.isVirtual()) subset.setIsVirtual(true);
        subset.add(new Relation().addParent(this));
        Relation parent = (Relation) find(Relation.class);
        parent = (parent!=null)?parent:new Relation();
        parent.addChild(subset);
        this.add(parent);
        return subset;
    }

    /**
     *
     * @param ranges Elements of this array might be multiple things:
     *               - A map whose first entry represents a mapping between range and steps.
     *               - A list from which a first and last entry will be interpreted as range.
     *               - Any other object which might bew found in a 'Index' component.
     * @param idxbase Start index for every rank.
     * @param newShape New shape of the new sub-tensor.
     * @param offset Rank offset incremented according to recursive calls.
     * @return A new rank index.
     */
    private int _configureSubsetFromRanges(Object[] ranges, int[] idxbase, int[] newShape, int offset){
        for (int i=0; i<ranges.length; i++) {
            int first = 0;
            int last = 0;
            if (!(ranges[i] instanceof  List)){
                if (ranges[i] instanceof Map){
                    Object[] ks = ((Map)ranges[i]).keySet().toArray();
                    Object[] steps = ((Map)ranges[i]).values().toArray();
                    int new_i = _configureSubsetFromRanges(ks, idxbase, newShape, i+offset);
                    for (int ii=rank(); ii<(rank()+steps.length); ii++) {
                        idxbase[ii+i+offset] = (Integer)steps[ii-rank()];
                        newShape[ii+i+offset-rank()] /= idxbase[ii+i+offset];
                    }
                    i = new_i;
                    continue;
                } else {
                    Index index = (Index)find(Index.class);
                    if (index!=null){
                        Integer position = index.get(ranges[i], i+offset);
                        first = position;
                        last = position;
                    } else {
                        throw new IllegalStateException("[Tsr]: Given index key at axis "+i+offset+" not found!");
                    }
                }
            }else{
                ranges[i] = ((List)ranges[i]).toArray();
                ranges[i] = (((Object[])ranges[i])[0] instanceof List)?((List)((Object[])ranges[i])[0]).toArray():((Object[])ranges[i]);
                if (!(((Object[])(ranges[i]))[0] instanceof Integer) || !(((Object[])(ranges[i]))[((Object[])(ranges[i])).length-1] instanceof Integer)){
                    Index index = (Index)find(Index.class);
                    if (!(((Object[])(ranges[i]))[0] instanceof Integer)){
                        if (index!=null){
                            first = index.get(((Object[])(ranges[i]))[0], i+offset);
                        }
                    }  else {
                        first = (Integer) ((Object[])(ranges[i]))[0];
                    }
                    if (!(((Object[])(ranges[i]))[((Object[])(ranges[i])).length-1] instanceof Integer)){
                        if (index!=null){
                            last = index.get(((Object[])(ranges[i]))[((Object[])(ranges[i])).length-1], i+offset);
                        }
                    } else {
                        last = (Integer) ((Object[])(ranges[i]))[((Object[])(ranges[i])).length-1];
                    }
                } else {
                    first = ((Integer)((Object[])ranges[i])[0]);
                    last = ((Integer)((Object[])ranges[i])[((Object[])ranges[i]).length-1]);
                }
            }
            if (first>last){
                int temp = first;
                first = last;
                last = temp;
            }
            first = (first < 0) ? _shape[i]+first : first;
            last = (last < 0) ? _shape[i]+last : last;
            newShape[i+offset] = (last - first) + 1;
            idxbase[i+offset] = first;
        }
        return ranges.length+offset-1;
    }

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

    private int _i_of_i(int i){
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

    /**
     *  Static methods.
     */
    public static class Util
    {
        public static class io
        {
            public static double getFrom(Tsr t, int i) {
                if (t.isEmpty() || t.isUndefined()) return 0;
                else if (t.isVirtual()) return t.value64()[0];
                return t.value64()[t._i_of_i(i)];
            }

            public static double getFrom(Tsr t, int[] idx) {
                t.setIsVirtual(false);
                return t.value64()[t.i_of_idx(idx)];
            }

            public static void setInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.value64()[t._i_of_i(i)] = value;
            }

            public static void setInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.value64()[t.i_of_idx(idx)] = value;
            }

            public static void addInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.value64()[t._i_of_i(i)] += value;
            }

            public static void addInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.value64()[t.i_of_idx(idx)] += value;
            }

            public static Tsr addInto(Tsr t, Tsr source) {
                if (t.isVirtual() && source.isVirtual()) t.value64()[0] += source.value64()[0];
                else FunctionBuilder.build("I[0]<-(I[0]+I[1])", false).activate(new Tsr[]{t, source});
                return source;
            }

            public static void subInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.value64()[t._i_of_i(i)] -= value;
            }

            public static void subInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.value64()[t.i_of_idx(idx)] -= value;
            }

            public static void subInto(Tsr t, Tsr source) {
                if (t.isVirtual() && source.isVirtual()) {
                    t.value64()[0] -= source.value64()[0];
                } else {
                    if (t.isVirtual()) t.setIsVirtual(false);
                    int[] index = new int[t.shape().length];
                    int size = t.size();
                    for (int i = 0; i < size; i++) {
                        io.subInto(t, index, io.getFrom(source, index));
                        Indexing.increment(index, t.shape());
                    }
                }
            }

            public static void mulInto(Tsr t, int i, double value) {
                t.setIsVirtual(false);
                t.value64()[t._i_of_i(i)] *= value;
            }

            public static void mulInto(Tsr t, int[] idx, double value) {
                t.setIsVirtual(false);
                t.value64()[t.i_of_idx(idx)] *= value;
            }

        }

        public static class Exec
        {
            public static Tsr reshaped(Tsr tensor, int[] newForm, boolean newTsr) {
                tensor = (newTsr) ? new Tsr(tensor, true) : tensor;
                tensor._shape = _cached(Indexing._shpCheck(Indexing.rearrange(tensor._shape, newForm), tensor));
                tensor._translation = _cached(Indexing._rearrange(tensor._translation, tensor._shape, newForm));
                tensor._idxmap =  _cached(Indexing.newTlnOf(tensor._shape));
                int[] sliceCfg = null;
                if (tensor.has(int[].class)) sliceCfg = (int[])tensor.find(int[].class);
                if (sliceCfg!=null){
                    int[] newSliceConfig = new int[sliceCfg.length];
                    for (int i=0; i<newForm.length; i++){
                        newSliceConfig[i] = sliceCfg[newForm[i]];
                    }
                    if (sliceCfg.length!=tensor.rank()){
                        for (int i=0; i<newForm.length; i++){
                            newSliceConfig[tensor.rank()+i] = sliceCfg[tensor.rank()+newForm[i]];
                        }
                    }
                    tensor.add(newSliceConfig);
                }
                return tensor;
            }

        }

        public static class Create
        {
            public  static Tsr E(int[] shape){
                return new Tsr(shape, 2.7182818284590452353602874713527);
            }

            public static Tsr newRandom(int[] shape){
                return newRandom(shape, 8701252152903546L);
            }

            public static Tsr newRandom(int[] shape, long seed){
                int size = Indexing.szeOfShp(shape);
                //DataHelper.newSeededDoubleArray(seed, size);
                //double[] value = new double[size];
                //for (int i=0; i<size; i++) value[i] = DataHelper.getDoubleOf(seed+i);
                return new Tsr(shape, DataHelper.newSeededDoubleArray(seed, size));
            }

            public static Tsr newTsrLike(Tsr template, double value) {
                Tsr t = _newEmptyLike(template);
                if (template.is32()) t.setValue((float)value);
                else t.setValue(value);
                if (template.isOutsourced()) ((Device)template.find(Device.class)).add(t);
                return t;
            }

            public static Tsr newTsrLike(Tsr template) {//The output tensor will not have gradients!
                Tsr t = _newEmptyLike(template);
                if (template.is32()) t.setValue32(new float[template.size()]);
                else t.setValue64(new double[template.size()]);
                if (template.isOutsourced()) ((Device)template.find(Device.class)).add(t);
                return t;
            }

            private static Tsr _newEmptyLike(Tsr template) {
                Tsr t = new Tsr();
                t._shape = template._shape;
                t._idxmap = template._idxmap;
                t._translation = template.translation();
                return t;
            }

        }

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
            private static int[] _shpCheck(int[] newShp, Tsr t) {
                if (szeOfShp(newShp) != t.size()) {
                    throw new IllegalArgumentException(
                            "[Tsr][shpCheck(int[] newShp, Tsr t)]: New shape does not match tensor size!" +
                            " (" + Stringify._strShp(newShp) + ((szeOfShp(newShp) < t.size()) ? "<" : ">") + Stringify._strShp(t.shape()) + ")");
                }
                return newShp;
            }

            @Contract(pure = true)
            private static int[] _rearrange(int[] tln, int[] shp, @NotNull int[] newForm) {
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
