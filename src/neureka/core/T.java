package neureka.core;

import neureka.core.device.Device;
import neureka.core.function.IFunction;
import neureka.core.function.factory.autograd.GraphNode;
import neureka.core.utility.DataHelper;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.BiConsumer;
import java.util.function.Consumer;

public class T {

    // DEFAULT DEVICE (HOST CPU)
    //=========================
    private static Device CPU;

    //STATIC FUNCTIONS MEMORY:
    //=========================
    private static HashMap<Long, int[]> CONFIGS;
    static {
        CONFIGS = new HashMap<>();//The things we do for memory
        CPU = new Device(null);//<= creates CPU-Aparapi-TensorKernel
    }
    //-----------------------------------------------------------------------

    //MODULE I / O :
    //=========================
    private ArrayList<Object> _components = new ArrayList<Object>();
    //-----------------------------------------------------------------------
    public ArrayList<Object> getComponents() {
        return _components;
    }
    public T setComponents(ArrayList<Object> properties) {
        _components = properties;
        return this;
    }
    public T add(Object newComponent) {
        if(newComponent==null){
            return this;
        }
        if (_components != null) {
            Object oldCompartment = find(newComponent.getClass());
            if (oldCompartment != null) {
                _components.remove(oldCompartment);
                _components.trimToSize();
            }
            _components.add((newComponent instanceof  int[])?cached((int[])newComponent):newComponent);
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
    public T remove(Class componentClass) {
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
    private int[] _shape, _translation;
    private double[] _value, _gradient;
    //-----------------------------------------------------------------------

    public Device device(){
        if(this.isOutsourced()){
            return (Device) this.find(Device.class);
        }
        return CPU;
    }

    public double[] gradient(){
        if(this.rqsGradient()&&this.isOutsourced()&&this.has(Device.class)){
            return ((Device)this.find(Device.class)).valueOf(this, true);
        }
        return _gradient;
    }
    public T setGradient(T g){
        _gradient = g.value();
        return this;
    }

    public double[] value() {
        if(_value ==null && this.isOutsourced() && this.has(Device.class)){
            return ((Device)this.find(Device.class)).valueOf(this, false);
        }
        double[] newValue = _value;
        if(this.isVirtual()){
            newValue = new double[this.size()];
            for(int i=0; i<newValue.length; i++){
                newValue[i] = _value[0];
            }
        }
        return newValue;
    }
    public T setValue(double[] newValue){
        _value = newValue;
        if(this.isOutsourced() && newValue!=null){
            ((Device)this.find(Device.class)).add(this);
        }
        return this;
    }

    public int[] shape() {
        return _shape;
    }

    public int[] translation(){
        return _translation;
    }

    public int size() {
        if(this.isEmpty()){
            return 0;
        }
        //_value is not optimal! //TODO GET SIZE FROM KERNEL IF OUTSOURCED
        return (this.isOutsourced())
                ?T.utility.szeOfShp(this.shape())
                :(this.isVirtual()
                    ?T.utility.szeOfShp(this.shape())
                    :_value.length);
    }

    public int[] shpIdx(int idx) {
        return T.utility.IdxToShpIdx(idx, _translation);
    }

    public boolean isEmpty() {
        return _value == null && !this.isOutsourced();
    }

    public boolean isUndefined(){
        return _shape == null;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //FLAG FIELDS:
    //=========================
    private int _flags = 0 + 0 + 0 + 0;//Default
    private final static int RQS_GRADIENT_MASK = 1;
    private final static int IS_OUTSOURCED_MASK = 2;
    private final static int IS_VIRTUAL_MASK = 4;
    //-----------------------------------------------------------------------
    public boolean rqsGradient() {
        return (_flags & RQS_GRADIENT_MASK) == RQS_GRADIENT_MASK;
    }
    public T setRqsGradient(boolean rqsGradient) {
        if (rqsGradient() != rqsGradient) {
            if (rqsGradient) {
                _flags += RQS_GRADIENT_MASK;
            } else {
                _flags -= RQS_GRADIENT_MASK;
            }
        }
        return this;
    }

    public boolean isOutsourced() {
        return (_flags & IS_OUTSOURCED_MASK) == IS_OUTSOURCED_MASK;
    }
    public T setIsOutsourced(boolean isOutsourced) {
        if (isOutsourced() != isOutsourced) {
            if (isOutsourced) {
                _flags += IS_OUTSOURCED_MASK;
            } else {
                _flags -= IS_OUTSOURCED_MASK;
            }
        }
        if(isOutsourced){
            _value = null;
            _gradient = null;
        }else if(this.has(Device.class)){
            Device device = (Device) this.find(Device.class);
            if(device.has(this)){
                device.get(this);
            }
            this.remove(Device.class);
        }
        return this;
    }

    public boolean isVirtual() {
        return (_flags & IS_VIRTUAL_MASK) == IS_VIRTUAL_MASK;
    }

    public T setIsVirtual(boolean isVirtual) {
        if (isVirtual() != isVirtual) {
            double v = _value[0];
            if (isVirtual) {
                _value = new double[]{v};
                _flags += IS_VIRTUAL_MASK;
            } else {
                _value = new double[this.size()];
                for(int i = 0; i<_value.length; i++){
                    _value[i] = v;
                }
                _flags -= IS_VIRTUAL_MASK;
            }
        }
        return this;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //GENERIC PROPERTIES :
    //=========================
    public boolean belongsToGraph(){
      return this.has(GraphNode.class);
    }

    public boolean isLeave(){
        return (!this.has(GraphNode.class)) || ((GraphNode) this.find(GraphNode.class)).isOrigin();
    }

    public boolean isBranch(){
        return !this.isLeave();
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    //DISPLAY :
    //=========================
    public String toString(String mode){
        if(this.isEmpty()){
            return "empty";
        }else if(this.isUndefined()){
            return "undefined";
        }
        String strShape = "";
        for(int i = 0; i<_shape.length; i++){
            strShape+=_shape[i];
            if(i<_shape.length-1){
                strShape+="x";
            }
        }
        strShape = "["+strShape+"]";
        String strValue = "";
        double[] v = _value;
        int size = (this.isVirtual()?this.size():v.length);
        for(int i=0; i<size; i++){
            strValue+= v[(this.isVirtual())?0:i];
            if(i<size-1){
                strValue+=", ";
            }
        }
        strValue = strShape+":("+strValue+")";
        if(mode=="r"){
            if(this.has(GraphNode.class)&&((GraphNode) this.find(GraphNode.class)).size()>0){
                GraphNode d = (GraphNode) this.find(GraphNode.class);
                String[] strDerivatives = {"; "};
                d.forEach((target, derivative)->{
                    strDerivatives[0]+="=>d|[ "+derivative.toString("r")+" ]|:t{ "+target.toString("r")+" }, ";
                });
                strValue += strDerivatives[0];
            }
        }else if(mode == "d"){
            if(this.has(GraphNode.class)&&((GraphNode) this.find(GraphNode.class)).size()>0){
                GraphNode d = (GraphNode) this.find(GraphNode.class);
                if(d.mode()!=0){
                    String[] strDerivatives = {"; "};
                    d.forEach((target, derivative)->strDerivatives[0]+="->d"+derivative.toString()+", ");
                    strValue += strDerivatives[0];
                }
            }
        }
        return strValue;
    }
    public String toString(){
        return toString("d");
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //CONSTRUCTION :
    //=========================
    public T() { }// creates empty tensor;

    public T(int[] shape) {
        _value = new double[T.utility.szeOfShp(shape)];
        this.initialShape(shape);
    }
    public T(int[] shape, double value) {
        int size = T.utility.szeOfShp(shape);
        _value = new double[1];
        this.setIsVirtual((size>1));
        this.initialShape(shape);
        _value[0] = value;
    }

    public T(int[] shape, double[] value){
        _value = value;
        initialShape(shape);
    }

    public T(T tensor) {
        _shape = tensor._shape;
        _translation = tensor._translation;
        _value = new double[tensor.size()];
        _components = null;//tensor._components;
        this._flags = tensor._flags;
        for (int i = 0; i < _value.length; i++) {
            _value[i] = tensor._value[i];
        }
    }

    public T initialShape(int[] newShape) {
        int size = T.utility.szeOfShp(newShape);
        if (_value == null) {
            _value = new double[size];
        }
        if (size != _value.length && !this.isVirtual()) {
            return this;//TODO: Exception!
        }
        _shape = cached(newShape);
        _translation = cached(T.utility.idxTln(newShape));
        return this;
    }

    private int[] cached(int[] data){
        long key = 0;
        for (int i = 0; i < data.length; i++) {
            if(data[i]<=10){
                key *= 10;
            }else if(data[i]<=100){
                key *= 100;
            }else if(data[i]<=1000){
                key *= 1000;
            }else if(data[i]<=10000){
                key *= 10000;
            }else if(data[i]<=100000){
                key *= 100000;
            }else if(data[i]<=1000000){
                key *= 1000000;
            }else if(data[i]<=10000000){
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
    public T(T tensor, String operation) {
        if(tensor==null){
            return;
        }
        _construct(new T[]{tensor}, operation);
    }
    public T(T[] tensors, String operation) {
        _construct(tensors, operation);
    }

    private void _construct(T[] tensors, String operation){
        if(tensors==null||tensors.length==0||tensors[0]==null){
            return;
        }
        IFunction.execute(this, tensors, operation);
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //MODIFICATION :
    //=========================
    public int[][] config(){
        return (this.has(int[][].class)?(int[][])find(int[][].class):new int[][]{this.shape(), this.translation()});
    }

    public int[] shape(int i){
        int[][] conf = this.config();
        i = Math.abs(i)*2+0;
        if(i<conf.length){
            return conf[i];
        }
        return null;
    }

    public int[] translation(int i){
        int[][] conf = this.config();
        i = Math.abs(i)*2+1;
        if(i<conf.length){
            return conf[i];
        }
        return null;
    }

    private void _record(int[] shp, int[] tln){
        int[][] conf = this.config();
        int sze = (conf==null)?0:conf.length;
        int[][] newConf = new int[sze+2][];
        newConf[0] = shp;
        newConf[1] = tln;
        for(int i=2; i<newConf.length; i++){
            newConf[i] = conf[i-2];
        }
        this.add(newConf);
    }

    public T inject(T tensor) {
        _value = tensor._value;
        _shape = tensor._shape;
        _translation = tensor._translation;
        _components = tensor._components;
        this._flags = tensor._flags;
        return this;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public T backward(T error) {
        if(this.rqsGradient()){
            this.setGradient(error);
        }
        if(this.has(GraphNode.class)){
            ((GraphNode)this.find(GraphNode.class)).backward(error);
        }
        return this;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public T delete(){
        if(this.isOutsourced()){
            ((Device)this.find(Device.class)).rmv(this);
        }
        this._flags = -1;
        _value = null;
        _shape = null;
        _translation = null;
        _components = null;
        _gradient = null;
        return this;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //ELEMENTARY OPERATIONS:
    //=========================
    public T foreach(Consumer<Integer> action){
        this.setIsVirtual(false);
        int sze = this.size();
        int[] idx = new int[this.shape().length];
        for(int i=0; i<sze; i++){
            T.utility.increment(idx, this.shape());
            action.accept(T.utility.idxOfShpIdxAndShp(idx, this.shape()));
        }
        return this;
    }
    public T foreach(BiConsumer<Integer, Double> action){
        this.setIsVirtual(false);
        int sze = this.size();
        int[] idx = new int[this.shape().length];
        double[] value = _value;
        for(int i=0; i<sze; i++){
            T.utility.increment(idx, this.shape());
            action.accept(i, value[T.utility.idxOfShpIdxAndShp(idx, this.shape())]);
        }
        return this;
    }

    /**
     *    ======================================================================================================
     *    FACTORY FUNCTIONS:
     * */
    public static class factory{

        public static class io {
            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
            public static double getFrom(T t, int i) {
                if(t.isEmpty()||t.isUndefined()){
                    return 0;
                } else if(t.isVirtual()){
                    return t._value[0];
                }
                return t._value[T.utility.idxOfShpIdxAndShp(t.shpIdx(i), t.shape())];
            }
            public static double getFrom(T t, int[] idx) {
                t.setIsVirtual(false);
                return t._value[T.utility.idxOfShpIdxAndShp(idx, t.shape())];
            }
            public static void setInto(T t, int i, double value) {
                t.setIsVirtual(false);
                t._value[T.utility.idxOfShpIdxAndShp(t.shpIdx(i), t.shape())] = value;
            }
            public static void setInto(T t, int[] idx, double value) {
                t.setIsVirtual(false);
                t._value[T.utility.idxOfShpIdxAndShp(idx, t.shape())] = value;
            }
            public static void addInto(T t, int i, double value) {
                t.setIsVirtual(false);
                t._value[T.utility.idxOfShpIdxAndShp(t.shpIdx(i), t.shape())] += value;
            }
            public static void addInto(T t, int[] idx, double value) {
                t.setIsVirtual(false);
                t._value[T.utility.idxOfShpIdxAndShp(idx, t._shape)] += value;
            }
            public static T addInto(T t, T source) {
                if(t.isVirtual() && source.isVirtual()){
                    t._value[0] += source._value[0];
                } else {
                    if(t.isVirtual()){
                        t.setIsVirtual(false);
                    }
                    int[] index = new int[t.shape().length];
                    int size = t.size();
                    for (int i = 0; i < size; i++) {
                        addInto(t, index, getFrom(source, index));
                        T.utility.increment(index, t.shape());
                    }
                }
                return source;
            }
            public static void subInto(T t, int i, double value) {
                t.setIsVirtual(false);
                t._value[T.utility.idxOfShpIdxAndShp(t.shpIdx(i), t.shape())] -= value;
            }
            public static void subInto(T t, int[] idx, double value) {
                t.setIsVirtual(false);
                t._value[T.utility.idxOfShpIdxAndShp(idx, t.shape())] -= value;
            }
            public static void subInto(T t, T source) {
                if(t.isVirtual() && source.isVirtual()){
                    t._value[0] -= source._value[0];
                } else {
                    if (t.isVirtual()) {
                        t.setIsVirtual(false);
                    }
                    int[] index = new int[t.shape().length];
                    int size = t.size();
                    for (int i = 0; i < size; i++) {
                        io.subInto(t, index, io.getFrom(source, index));
                        T.utility.increment(index, t.shape());
                    }
                }
            }
            public static void mulInto(T t, int i, double value) {
                t.setIsVirtual(false);
                t._value[T.utility.idxOfShpIdxAndShp(t.shpIdx(i), t.shape())] *= value;
            }
            public static void mulInto(T t, int[] idx, double value) {
                t.setIsVirtual(false);
                t._value[T.utility.idxOfShpIdxAndShp(idx, t._shape)] *= value;
            }

            //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        }

        public static void inject(double[] data, boolean grd, T tensor){
            if(grd) {
                tensor._gradient = data;
            }else{
                tensor._value = data;
            }

        }

        public static T reshaped(T tensor, int[] newForm, boolean newTsr){
            if(newTsr){
                tensor = copyOf(tensor);
            }
            tensor._record(tensor.shape(), tensor.translation());
            tensor._shape = T.utility.shpCheck(T.utility.reshaped(tensor._shape, newForm), tensor);
            tensor._translation = T.utility.retranslated(tensor._translation, tensor._shape, newForm);
            return tensor;
        }

        //OPERATIONS:
        //=========================

        public static T convolution(T tensor1, T tensor2){
            tensor1.setIsVirtual(false);
            tensor2.setIsVirtual(false);
            T newTensor = new T(T.utility.shpOfCon(tensor1.shape(), tensor2.shape()));
            T.utility.tensMul_mxd(
                    newTensor.shape().length,
                    new double[][]{tensor1._value, tensor2._value, newTensor._value}, new int[]{0, 0, 0},
                    T.utility.mxdFromShape(tensor1.shape()),
                    T.utility.mxdFromShape(tensor2.shape()),
                    T.utility.mxdFromShape(newTensor.shape())
            );
            return newTensor;
        }

        public static T convolution_inv(T drain, T source1, T source2, boolean first){
            source1.setIsVirtual(false);
            source2.setIsVirtual(false);
            drain.setIsVirtual(false);
            T.utility.tensMul_inv_mxd(
                    drain.shape().length,
                    new double[][]{source1._value, source2._value, drain._value}, new int[]{0, 0, 0},
                    T.utility.mxdFromShape(source1.shape()),
                    T.utility.mxdFromShape(source2.shape()),
                    T.utility.mxdFromShape(drain.shape()),
                    first
            );
            return (first)?source1:source2;
        }

        public static T multiplication(T tensor1, T tensor2){
            T drn = new T(tensor1.shape());
            int[] index = new int[drn.shape().length];
            int size = drn.size();
            for(int i=0; i<size; i++){
                io.addInto(drn, index, io.getFrom(tensor1, index)* io.getFrom(tensor2, index));
                T.utility.increment(index, drn.shape());
            }
            return drn;
        }

        public static T addition(T tensor1, T tensor2){
            T drn = new T(tensor1.shape());
            int[] index = new int[drn.shape().length];
            int size = drn.size();
            for(int i=0; i<size; i++){
                io.addInto(drn, index, io.getFrom(tensor1, index)+ io.getFrom(tensor2, index));
                T.utility.increment(index, drn.shape());
            }
            return drn;
        }

        public static T newTensor(double value, int[] shape){
            int sze = T.utility.szeOfShp(shape);
            T tensor = new T();
            tensor._value = new double[sze];
            tensor.initialShape(shape);
            for(int i=0; i<sze; i++){
                tensor._value[i] = value;
            }
            return tensor;
        }

        public static T newTensor(double[] value, int[] shape){
            T tensor = new T();
            tensor._value = value;
            tensor.initialShape(shape);
            return tensor;
        }
        public static T newTensor(double[] value, int[] shape, int[] translation){
            T tensor = new T();
            tensor._value = value;
            tensor.initialShape(shape);
            tensor._translation = translation;
            return tensor;
        }
        public static T newTensor(int[] shape, int[] translation){
            T tensor = new T();
            tensor._value = new double[T.utility.szeOfShp(shape)];
            tensor.initialShape(shape);
            tensor._translation = (translation!=null)?translation:tensor._translation;//FUNCTIONS.put()
            return tensor;
        }

        public static T copyOf(T tensor){
            T newTensor = new T();
            newTensor._shape = tensor._shape;
            newTensor._translation = tensor._translation;
            newTensor._value = new double[tensor.size()];
            newTensor._components = null;//tensor._components;
            newTensor._flags = tensor._flags;
            double[] value = tensor._value;
            for (int i = 0; i < value.length; i++) {
                newTensor._value[i] = value[i];
            }
            if(tensor.isOutsourced()){
                newTensor.add(tensor.device());
            }
            return newTensor;
        }
        public static T copyOf(Object[] things){
            for(int i=0; i<things.length; i++){
                if(things[i] instanceof int[]){

                }//TODO: complete
            }
            return new T();
        }
        public static T reshapedCopyOf(T tensor, int[] newForm) {
            T newTensor = new T();
            newTensor._value = tensor._value;
            newTensor._shape = T.utility.reshaped(tensor._shape, newForm);
            newTensor._translation = T.utility.reshaped(tensor._translation, newForm);
            newTensor._components = tensor._components;//Reshaped derivs usw
            return newTensor;
        }
    }
    /**
     *   ======================================================================================================
     *   UTILITY FUNCTIONS:
     *
     * */
    public static class utility {

        public static boolean shareGuestDevice(T[] tsrs){
            boolean onSameGuestDevice = true;
            Device device = null;
            for (int ti = 0; ti < tsrs.length; ti++) {
                device = (tsrs[ti].isOutsourced())?(Device)tsrs[ti].find(Device.class):device;
            }
            if(device!=null) {
                for (int ti = 0; ti < tsrs.length; ti++) {
                    onSameGuestDevice = (!tsrs[ti].isVirtual() && device == tsrs[ti].find(Device.class)) && onSameGuestDevice;
                }
            }else{
                onSameGuestDevice = false;
            }
            return onSameGuestDevice;
        }

        @Contract(pure = true)
        public static void increment_mxd(@NotNull int[] shpIdx, @NotNull int[] shape, int start, int rank) {
            int i = start;
            int end = start + rank;
            while (i >= start && i < end) {
                i = incrementAt(i, shpIdx, shape);
            }
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
        //-----------------------------------------------------------------------
        @Contract(pure = true)
        public static void decrement_onMixed(@NotNull int[] shpIdx, @NotNull int[] shape, int start, int rank) {
            int i = start;
            int end = start + rank;
            while (i >= start && i < end) {
                i = decrementAt(i, shpIdx, shape);
            }
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
        //-----------------------------------------------------------------------
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
        //-----------------------------------------------------------------------
        @Contract(pure = true)
        public static int idxOfShpIdxAndShp(int[] shpIdx, int[] shape) {
            int[] trltn = idxTln(shape);
            int idx = 0;
            for (int i = 0; i < trltn.length; i++) {
                idx += trltn[i] * shpIdx[i];
            }
            return idx;
        }
        @Contract(pure = true)
        public static int[] idxTln(int[] shape) {
            return idxTln(shape, new int[shape.length]);
        }

        @Contract(pure = true)
        public static int[] idxTln(int[] shp, int[] tln) {
            int prod = 1;
            for (int i = 0; i < tln.length; i++) {
                tln[i] = prod;
                prod *= shp[i];
            }
            return tln;
        }

        //-----------------------------------------------------------------------
        @Contract(pure = true)
        public static int[] IdxToShpIdx(int idx, int[] translation) {
            int[] shpIdx = new int[translation.length];
            for (int i = translation.length - 1; i >= 0; i--) {
                int r = idx % translation[i];
                shpIdx[i] = (idx - r) / translation[i];
                idx = r;
            }
            return shpIdx;
        }
        //-----------------------------------------------------------------------
        @Contract(pure = true)
        public static int[] reshaped(int[] shp, @NotNull int[] newForm) {
            int[] newShp = new int[newForm.length];
            for (int i = 0; i < newForm.length; i++) {
                if (newForm[i] < 0) {
                    newShp[i] = Math.abs(newForm[i]);
                } else if (newForm[i] >= 0) {
                    newShp[i] = shp[newForm[i]];
                }
            }
            return newShp;
        }

        @Contract(pure = true)
        public static int[] shpCheck(int[] newShp, T t){
            if(szeOfShp(newShp)!=t.size()){
                throw new IllegalArgumentException("New _shape does not match tensor size!" +
                        " ("+str(newShp)+((szeOfShp(newShp)<t.size()) ?"<":">")+str(t.shape())+")");
            }
            return newShp;
        }

        @Contract(pure = true)
        public static String str(int[] shp){
            String str = "";
            int i=0;
            for(int s : shp){
                str+=s+((i!=shp.length-1)?", ":"");
                i++;
            }
            return "["+str+"]";
        }

        @Contract(pure = true)
        public static int[] retranslated(int[] tln, int[] shp, @NotNull int[] newForm){
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

        //-----------------------------------------------------------------------
        @Contract(pure = true)
        public static double[] randFromShape_mxd(int[] shape, int start, int rank, double[] data, int dataPtr) {
            int size = szeOfShp(shape);
            for (int i = 0; i < size; i++) {
                data[dataPtr + i] = DataHelper.getDoubleOf(i);
            }
            return data;
        }

        @Contract(pure = true)
        public static int[][] reshapedAndToMxd(int[] shape, int[] newShp) {
            return mxdFromShape(reshaped(shape, newShp), reshaped(idxTln(shape), newShp));
        }

        @Contract(pure = true)
        public static int[][] mxdFromShape(int[] shape, int[] trnln) {
            int rank = shape.length;
            int[][] mxd = new int[4][];
            mxd[0] = shape;
            mxd[1] = trnln;
            mxd[2] = new int[rank];
            mxd[3] = new int[]{0};
            return mxd;
        }

        @Contract(pure = true)
        public static int[][] mxdFromShape(int[] shape) {
            int rank = shape.length;
            return mxdFromShape(shape, idxTln(shape, new int[rank]));
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
        public static int[] shpOfCon(int[] frstShp, int[] scndShp) {
            int[] shape = new int[(frstShp.length + scndShp.length) / 2];
            for (int i = 0; i < frstShp.length && i < scndShp.length; i++) {
                shape[i] = Math.abs(frstShp[i] - scndShp[i]) + 1;
            }
            return shape;
        }

        @Contract(pure = true)
        public static void tensMul_mxd
                (
                        int rank,
                        double[][] data,//[0]=>src1, [1]=>src2, [2]=>drn
                        int[] dataPtr,
                        int[][] src1, int[][] src2, int[][] drn
                ) {
            //hdr[0] => dim[]
            //hdr[1] => anchor[]
            //hdr[2] => idx[]
            //hdr[3] => {start}
            int src1End = src1[3][0] + rank;
            int src2End = src2[3][0] + rank;
            int drnEnd = drn[3][0] + rank;
            int drnSze = szeOfShp(drn[0]);
            int i = 0;
            while (i < drnSze) {
                //increment f and drain accordingly:
                int i1 = src1[3][0];
                int i2 = src2[3][0];
                int id = drn[3][0];
                int ri = 0;
                while (ri < rank) {
                    if (src1[0][i1] == src2[0][i2]) {//setting 0
                        src1[2][i1] = drn[2][id];//mtch[mi];
                        src2[2][i2] = drn[2][id];//mtch[mi];
                    } else if (src1[0][i1] > src2[0][i2]) {//setting hdr1 idx to id idx
                        src1[2][i1] = drn[2][id];//mtch[mi];
                        src2[2][i2] = 0;
                    } else if (src1[0][i1] < src2[0][i2]) {//setting hdr2 idx to id idx
                        src1[2][i1] = 0;
                        src2[2][i2] = drn[2][id];//mtch[mi];
                    }
                    i1++;
                    i2++;
                    id++;
                    ri++;
                }
                //----------
                // multiplication:
                double value = 0;
                boolean running = true;
                boolean incrementing = false;
                while (running) {
                    if (i1 == src1End || i2 == src2End || id == drnEnd) {
                        i1 = src1[3][0];
                        i2 = src2[3][0];
                        id = drn[3][0];
                    }
                    if (incrementing == false) {
                        int idx1 = idxOfFrmt_mxd(src1, rank);
                        int idx2 = idxOfFrmt_mxd(src2, rank);
                        System.out.println(
                            "hdr1:" + strInt(src1[2]) + "; " +
                            "hdr2:" + strInt(src2[2]) + "; " +
                            "drn:" + strInt(drn[2]) +
                            " idx1:(" + idx1 + ");" +
                            " idx2:(" + idx2 + ");" +
                            " drn:(" + idxOfFrmt_mxd(drn, rank) + ");" +
                            " val:(" + value + ") += val1:(" + data[0][dataPtr[0] + idx1] + ") x val2:(" + data[1][dataPtr[1] + idx2] + ");");
                        value += data[0][dataPtr[0] + idx1] * data[1][dataPtr[1] + idx2];
                        incrementing = true;
                        i1 = src1[3][0];
                        i2 = src2[3][0];
                        id = drn[3][0];
                    } else {//incrementing:
                        if (src1[2][i1] < src1[0][i1] && src2[2][i2] < src2[0][i2]) {
                            src1[2][i1]++;
                            src2[2][i2]++;
                            if (src1[2][i1] == src1[0][i1] || src2[2][i2] == src2[0][i2]) {
                                if ((i1 == (src1End - 1) || i2 == (src2End - 1))) {
                                    running = false;
                                }
                                if (src1[0][i1] == src2[0][i2]) {//setting 0
                                    src1[2][i1] = drn[2][id];//mtch[mi];
                                    src2[2][i2] = drn[2][id];//mtch[mi];
                                } else if (src1[0][i1] > src2[0][i2]) {//setting hdr1 idx to id idx
                                    src1[2][i1] = drn[2][id];//mtch[mi];
                                    src2[2][i2] = 0;
                                } else if (src1[0][i1] < src2[0][i2]) {//setting hdr2 idx to id idx
                                    src1[2][i1] = 0;
                                    src2[2][i2] = drn[2][id];//mtch[mi];
                                }
                                i1++;
                                i2++;
                                id++;
                            } else {
                                incrementing = false;
                                i1 = src1[3][0];
                                i2 = src2[3][0];
                                id = drn[3][0];
                            }
                        } else {
                            i1++;
                            i2++;
                            id++;
                        }
                    }
                }//setInto _value in drn:
                int idx = idxOfFrmt_mxd(drn, rank);
                data[2][dataPtr[2] + idx] = value;
                System.out.println(idx + " - " + i);
                i++;//increment on drain:
                if (i < drnSze) {
                    increment_mxd(drn[2], drn[0], drn[3][0], rank);
                }
            }
            System.out.println("result:");
            System.out.println(strInt(src1[2]) + "-" + strInt(src1[0]) + "-" + strInt(src1[1]));
            System.out.println(strInt(src2[2]) + "-" + strInt(src2[0]) + "-" + strInt(src2[1]));
            System.out.println(strInt(drn[2]) + "-" + strInt(drn[0]) + "-" + strInt(drn[1]));
        }
        //=========================

        @Contract(pure = true)
        public static void tensMul_inv_mxd
                (
                     int rank,
                     double[][] data,//[0]=>src1, [1]=>src2, [2]=>drn
                     int[] dataPtr,
                     int[][] src1, int[][] src2, int[][] drn,
                     boolean first
                ) {
            if(first){//TODO: ALL OF THIS NEEDS TO BE IMPLEMENTED AS IN TensorKernel!
                for(int i=0; i<data[0].length; i++){
                    data[0][i] = 0;
                }
            }else{
                for(int i=0; i<data[1].length; i++){
                    data[1][i] = 0;
                }
            }
            //hdr[0] => dim[]
            //hdr[1] => anchor[]
            //hdr[2] => idx[]
            //hdr[3] => {start}
            int src1End = src1[3][0] + rank;
            int src2End = src2[3][0] + rank;
            int drnEnd = drn[3][0] + rank;
            int drnSze = szeOfShp(drn[0]);
            int i = 0;
            while (i < drnSze) {
                //increment f and drain accordingly:
                int i1 = src1[3][0];
                int i2 = src2[3][0];
                int id = drn[3][0];
                int ri = 0;
                while (ri < rank) {
                    if (src1[0][i1] == src2[0][i2]) {//setting 0
                        src1[2][i1] = drn[2][id];//mtch[mi];
                        src2[2][i2] = drn[2][id];//mtch[mi];
                    } else if (src1[0][i1] > src2[0][i2]) {//setting hdr1 idx to id idx
                        src1[2][i1] = drn[2][id];//mtch[mi];
                        src2[2][i2] = 0;
                    } else if (src1[0][i1] < src2[0][i2]) {//setting hdr2 idx to id idx
                        src1[2][i1] = 0;
                        src2[2][i2] = drn[2][id];//mtch[mi];
                    }
                    i1++;
                    i2++;
                    id++;
                    ri++;
                }
                //----------
                // multiplication:
                //double _value = 0;
                int idx = idxOfFrmt_mxd(drn, rank);//This has been added too
                boolean running = true;
                boolean incrementing = false;
                while (running) {
                    if (i1 == src1End || i2 == src2End || id == drnEnd) {
                        i1 = src1[3][0];
                        i2 = src2[3][0];
                        id = drn[3][0];
                    }
                    if (incrementing == false) {
                        int idx1 = idxOfFrmt_mxd(src1, rank);
                        int idx2 = idxOfFrmt_mxd(src2, rank);
                        System.out.println(
                                "hdr1:" + strInt(src1[2]) + "; " +
                                        "hdr2:" + strInt(src2[2]) + "; " +
                                        "drn:" + strInt(drn[2]) +
                                        " idx1:(" + idx1 + ");" +
                                        " idx2:(" + idx2 + ");" +
                                        " drn:(" + idxOfFrmt_mxd(drn, rank) + ");" +
                                        //" val:(" + _value + ") += val1:(" + data[0][dataPtr[0] + idx1] + ") x val2:(" + data[1][dataPtr[1] + idx2] +
                                ");");
                        //_value += data[0][dataPtr[0] + idx1] * data[1][dataPtr[1] + idx2];
                        if(first){
                            data[0][dataPtr[0] + idx1] += data[2][dataPtr[2] + idx] * data[1][dataPtr[1] + idx2];
                        } else {
                            data[1][dataPtr[1] + idx2] += data[2][dataPtr[2] + idx] * data[0][dataPtr[0] + idx1];
                        }
                        incrementing = true;
                        i1 = src1[3][0];
                        i2 = src2[3][0];
                        id = drn[3][0];
                    } else {//incrementing:
                        if (src1[2][i1] < src1[0][i1] && src2[2][i2] < src2[0][i2]) {
                            src1[2][i1]++;
                            src2[2][i2]++;
                            if (src1[2][i1] == src1[0][i1] || src2[2][i2] == src2[0][i2]) {
                                if ((i1 == (src1End - 1) || i2 == (src2End - 1))) {
                                    running = false;
                                }
                                if (src1[0][i1] == src2[0][i2]) {//setting 0
                                    src1[2][i1] = drn[2][id];//mtch[mi];
                                    src2[2][i2] = drn[2][id];//mtch[mi];
                                } else if (src1[0][i1] > src2[0][i2]) {//setting hdr1 idx to id idx
                                    src1[2][i1] = drn[2][id];//mtch[mi];
                                    src2[2][i2] = 0;
                                } else if (src1[0][i1] < src2[0][i2]) {//setting hdr2 idx to id idx
                                    src1[2][i1] = 0;
                                    src2[2][i2] = drn[2][id];//mtch[mi];
                                }
                                i1++;
                                i2++;
                                id++;
                            } else {
                                incrementing = false;
                                i1 = src1[3][0];
                                i2 = src2[3][0];
                                id = drn[3][0];
                            }
                        } else {
                            i1++;
                            i2++;
                            id++;
                        }
                    }
                }//setInto _value in drn:
                //int idx = idxOfFrmt_mxd(drn, rank);
                //data[2][dataPtr[2] + idx] = _value;
                System.out.println(idx + " - " + i);
                i++;//increment on drain:
                if (i < drnSze) {
                    increment_mxd(drn[2], drn[0], drn[3][0], rank);
                }
            }
            System.out.println("result:");
            System.out.println(strInt(src1[2]) + "-" + strInt(src1[0]) + "-" + strInt(src1[1]));
            System.out.println(strInt(src2[2]) + "-" + strInt(src2[0]) + "-" + strInt(src2[1]));
            System.out.println(strInt(drn[2]) + "-" + strInt(drn[0]) + "-" + strInt(drn[1]));
        }


        //=========================

        @Contract(pure = true)
        public static int idxOfFrmt_mxd(int[][] mxdFrmt, int rank) {
            int end = mxdFrmt[3][0] + rank;
            int idx = 0;
            for (int i = mxdFrmt[3][0]; i < end; i++) {
                idx += mxdFrmt[1][i] * mxdFrmt[2][i];//anchor[i]*idx[i]
            }
            return idx;
        }

        @Contract(pure = true)
        public static int szeOfShp(int[] shape) {
            int size = 1;
            for (int Di = 0; Di < shape.length; Di++) {
                size *= shape[Di];
            }
            return size;
        }

        @Contract(pure = true)
        public static int[][] setupMxdOfCon(int[][] shape1, int[][] shape2) {
            int[][] match = new int[4][((shape1[0].length + shape2[0].length) / 2)];
            for (int i = 0; i < shape1[0].length && i < shape2[0].length; i++) {
                match[0][i] = Math.abs(shape1[0][i] - shape2[0][i]) + 1;
            }
            match[3] = new int[]{0};
            match[1] = T.utility.idxTln(match[0]);
            match[2] = new int[match[0].length];
            return match;
        }
        //-----------------------------------------------------------------------
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
         * 	int[] selfDim = find(int[].class);
         * 	//[Ii][Ni][0]=>weightDim [1]=>form
         *	int[][] FormData = find(int[][][][].class)[Ii][Ni]
         *	int[] weightDim = FormData[0];
         *	int[] connForm = FormData[1];
         *
         * */


    }


}
