package neureka.core;

import neureka.core.function.TFunctionFactory;
import neureka.core.device.TDevice;
import neureka.core.autograd.TGradientNode;
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
    private static TDevice CPU;

    //STATIC SHARED MEMORY:
    //=========================
    private static HashMap<Long, int[]> SHARED;
    static {
        SHARED = new HashMap<>();//The things we do for memory
        CPU = new TDevice(null);//<= creates CPU-Aparapi-Kernel
    }
    //-----------------------------------------------------------------------

    //DATA FIELDS:
    //=========================
    protected int[] shape = null;
    protected int[] translation = null;
    protected double[] value = null;
    protected double[] gradient = null;
    //-----------------------------------------------------------------------

    public TDevice device(){
        if(this.isOutsourced()){
            return (TDevice) this.find(TDevice.class);
        }
        return CPU;
    }

    public double[] gradient(){
        if(this.rqsGradient()&&this.isOutsourced()&&this.has(TDevice.class)){
            return ((TDevice)this.find(TDevice.class)).valueOf(this, true);
        }
        return gradient;
    }
    public void setGradient(T g){
        this.gradient = g.value;
    }

    public double[] value() {
        if(this.value==null && this.isOutsourced() && this.has(TDevice.class)){
            return ((TDevice)this.find(TDevice.class)).valueOf(this, false);
        }
        return value;
    }
    public void setValue(double[] newValue){
        this.value = newValue;
        if(this.isOutsourced() && newValue!=null){
            ((TDevice)this.find(TDevice.class)).add(this);
        }
    }

    public int[] shape() {
        return shape;
    }

    public int[] translation(){
        return translation;
    }

    public int size() {
        if(this.isEmpty()){
            return 0;
        }
        //this.value is not optimal!
        return (this.isOutsourced())?T.utility.szeOfShp(this.shape()):this.value.length;
    }

    public int[] shpIdx(int idx) {
        return T.utility.IdxToShpIdx(idx, this.translation);
    }

    public boolean isEmpty() {
        if (value == null && !this.isOutsourced()) {
            return true;
        }
        return false;
    }

    public boolean isUndefined(){
        if(this.shape == null){
            return true;
        }
        return false;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //FLAG FIELDS:
    //=========================
    private int flags = 0 + 0 + 0;//Default
    private final static int rqsGradient_MASK = 1;
    private final static int isOutsourced_MASK = 2;
    //-----------------------------------------------------------------------
    public boolean rqsGradient() {
        return ((flags & rqsGradient_MASK) == rqsGradient_MASK) ? true : false;
    }
    public void setRqsGradient(boolean rqsGradient) {
        if (rqsGradient() != rqsGradient) {
            if (rqsGradient) {
                flags += rqsGradient_MASK;
            } else {
                flags -= rqsGradient_MASK;
            }
        }
    }

    public boolean isOutsourced() {
        return ((flags & isOutsourced_MASK) == isOutsourced_MASK) ? true : false;
    }
    public void setIsOutsourced(boolean isOutsourced) {
        if (isOutsourced() != isOutsourced) {
            if (isOutsourced) {
                flags += isOutsourced_MASK;
            } else {
                flags -= isOutsourced_MASK;
            }
        }
        if(isOutsourced){
            this.value = null;
            this.gradient = null;
        }else if(this.has(TDevice.class)){
            TDevice device = (TDevice) this.find(TDevice.class);
            if(device.has(this)){
                device.get(this);
            }
            this.remove(TDevice.class);
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //MODULE I / O :
    //=========================
    private ArrayList<Object> Modules = new ArrayList<Object>();
    //-----------------------------------------------------------------------
    public ArrayList<Object> getModules() {
        return Modules;
    }
    public void setModules(ArrayList<Object> properties) {
        Modules = properties;
    }
    public void add(Object newModule) {
        if (Modules != null) {
            Object oldCompartment = find(newModule.getClass());
            if (oldCompartment != null) {
                Modules.remove(oldCompartment);
                Modules.trimToSize();
            }
            Modules.add((newModule instanceof  int[])?cached((int[])newModule):newModule);
        } else {
            Modules = new ArrayList<>();
            Modules.add(newModule);
        }
    }
    public Object find(Class moduleClass) {
        if (Modules != null) {
            for (int Pi = 0; Pi < Modules.size(); Pi++) {
                if (moduleClass.isInstance(Modules.get(Pi))) {
                    return Modules.get(Pi);
                }
            }
        }
        return null;
    }
    public void remove(Class moduleClass) {
        Object oldCompartment = find(moduleClass);
        if (oldCompartment != null) {
            Modules.remove(oldCompartment);
            Modules.trimToSize();
        }
        if (Modules.size() == 0) {
            Modules = null;
        }
    }
    public boolean has(Class moduleClass) {
        if (find(moduleClass) != null) {
            return true;
        }
        return false;
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
        for(int i=0; i<this.shape.length; i++){
            strShape+=this.shape[i];
            if(i<this.shape.length-1){
                strShape+="x";
            }
        }
        strShape = "["+strShape+"]";
        String strValue = "";
        double[] v = this.value();
        for(int i=0; i<v.length; i++){
            strValue+=v[i];
            if(i<v.length-1){
                strValue+=", ";
            }
        }
        strValue = strShape+":("+strValue+")";
        if(mode=="r"){
            if(this.has(TGradientNode.class)){
                TGradientNode d = (TGradientNode) this.find(TGradientNode.class);
                String[] strDerivatives = {"; "};
                d.forEach((target, derivative)->{
                    strDerivatives[0]+="=>d|[ "+derivative.toString("r")+" ]|:t{ "+target.toString("r")+" }, ";
                });
                strValue += strDerivatives[0];
            }
        }else if(mode == "d"){
            if(this.has(TGradientNode.class)){
                TGradientNode d = (TGradientNode) this.find(TGradientNode.class);
                String[] strDerivatives = {"; "};
                d.forEach((target, derivative)->{
                    strDerivatives[0]+="->d"+derivative.toString()+", ";
                });
                strValue += strDerivatives[0];
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
        value = new double[T.utility.szeOfShp(shape)];
        this.initialShape(shape);
    }
    public T(int[] shape, double value) {
        this.value = new double[T.utility.szeOfShp(shape)];
        this.initialShape(shape);
        for (int i = 0; i < this.value.length; i++) {
            this.value[i] = value;
        }
    }
    public T(T tensor) {
        this.shape = tensor.shape;
        this.translation = tensor.translation;
        this.value = new double[tensor.size()];
        this.Modules = null;//tensor.Modules;
        this.flags = tensor.flags;
        for (int i = 0; i < this.value.length; i++) {
            this.value[i] = tensor.value[i];
        }
    }

    public void initialShape(int[] newShape) {
        int size = T.utility.szeOfShp(newShape);
        if (value == null) {
            value = new double[size];
        }
        if (size != value.length) {
            return;
        }
        this.shape = cached(newShape);
        this.translation = cached(T.utility.idxTln(newShape));
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
        int[] found = SHARED.get(key);
        if (found != null) {
            return found;
        } else {
            SHARED.put(key, data);
            return data;
        }
    }
    //TRACKED COMPUTATION :
    //=========================
    public T(T tensor, String operation) {
        if(tensor==null){return;}
        construct(new T[]{tensor}, operation);
    }
    public T(T[] tensors, String operation) {
        construct(tensors, operation);
    }
    public T(T[] tensors, int[][] translation, String operation) {
        this.internalize(TFunctionFactory.newBuild(operation, true).activate(tensors));
        if(this.has(TGradientNode.class)){
            ((TGradientNode)this.find(TGradientNode.class)).trimTree(null);
        }
    }
    private void construct(T[] tensors, String operation){
        if(tensors==null||tensors.length==0||tensors[0]==null){return;}
        this.internalize(TFunctionFactory.newBuild(operation, true).activate(tensors));
        if(this.has(TGradientNode.class)){
            ((TGradientNode)this.find(TGradientNode.class)).trimTree(null);
        }
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

    private void record(int[] shp, int[] tln){
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

    public void internalize(T tensor) {
        this.value = tensor.value;
        this.shape = tensor.shape;
        this.translation = tensor.translation;
        this.Modules = tensor.Modules;
        this.flags = tensor.flags;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void backward(T error) {
        if(this.rqsGradient()){
            this.setGradient(error);
        }
        if(this.has(TGradientNode.class)){
            ((TGradientNode)this.find(TGradientNode.class)).backward(error);
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public void delete(){
        if(this.isOutsourced()){
            ((TDevice)this.find(TDevice.class)).rmv(this);
        }
        this.flags = -1;
        this.value = null;
        this.shape = null;
        this.translation = null;
        this.Modules = null;
        this.gradient = null;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //ELEMENTARY OPERATIONS:
    //=========================
    public void foreach(Consumer<Integer> action){
        int sze = this.size();
        int[] idx = new int[this.shape().length];
        for(int i=0; i<sze; i++){
            T.utility.increment(idx, this.shape());
            action.accept(T.utility.idxOfShpIdxAndShp(idx, this.shape()));
        }
    }
    public void foreach(BiConsumer<Integer, Integer> action){
        int sze = this.size();
        int[] idx = new int[this.shape().length];
        for(int i=0; i<sze; i++){
            T.utility.increment(idx, this.shape());
            action.accept(i, T.utility.idxOfShpIdxAndShp(idx, this.shape()));
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public double e_get(int i) {
        if(this.isEmpty()||this.isUndefined()){
            return 0;
        }
        return T.utility.idxOfShpIdxAndShp(shpIdx(i), shape);
    }
    public double e_get(int[] idx) {
        return value[T.utility.idxOfShpIdxAndShp(idx, shape)];
    }
    public void e_set(int i, double value) {
        this.value[T.utility.idxOfShpIdxAndShp(shpIdx(i), shape)] = value;
    }
    public void e_set(int[] idx, double value) {
        this.value[T.utility.idxOfShpIdxAndShp(idx, shape)] = value;
    }
    public void e_add(int i, double value) {
        this.value[T.utility.idxOfShpIdxAndShp(shpIdx(i), shape)] += value;
    }
    public void e_add(int[] idx, double value) {
        this.value[T.utility.idxOfShpIdxAndShp(idx, shape)] += value;
    }
    public void e_sub(int i, double value) {
        this.value[T.utility.idxOfShpIdxAndShp(shpIdx(i), shape)] -= value;
    }
    public void e_sub(int[] idx, double value) {
        this.value[T.utility.idxOfShpIdxAndShp(idx, shape)] -= value;
    }
    public void e_mul(int i, double value) {
        this.value[T.utility.idxOfShpIdxAndShp(shpIdx(i), shape)] *= value;
    }
    public void e_mul(int[] idx, double value) {
        this.value[T.utility.idxOfShpIdxAndShp(idx, shape)] *= value;
    }
    public T e_add(T tensor) {
        int[] index = new int[shape.length];
        int size = size();
        for (int i = 0; i < size; i++) {
            e_add(index, tensor.e_get(index));
            T.utility.increment(index, shape);
        }
        return tensor;
    }
    public void e_sub(T tensor) {
        int[] index = new int[shape.length];
        int size = size();
        for (int i = 0; i < size; i++) {
            this.e_sub(index, tensor.e_get(index));
            T.utility.increment(index, shape);
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    /**
     *    ======================================================================================================
     *    FACTORY FUNCTIONS:
     * */
    public static class factory{

        public static void inject(double[] data, boolean grd, T tensor){
            if(grd) {
                tensor.gradient = data;
            }else{
                tensor.value = data;
            }

        }

        public static T reshaped(T tensor, int[] newForm, boolean newTsr){
            if(newTsr){
                tensor = copyOf(tensor);
            }
            tensor.record(tensor.shape(), tensor.translation());
            tensor.shape = T.utility.reshaped(tensor.shape, newForm);
            tensor.translation = T.utility.retranslated(tensor.translation, tensor.shape, newForm);
            return tensor;
        }


        //OPERATIONS:
        //=========================
        public static T convolution(T tensor1, T tensor2){
            T newTensor = new T(T.utility.shpOfTensMul(tensor1.shape(), tensor2.shape()));
            T.utility.tensMul_mxd(
                    newTensor.shape().length,
                    new double[][]{tensor1.value(), tensor2.value(), newTensor.value()}, new int[]{0, 0, 0},
                    T.utility.mxdFromShape(tensor1.shape()),
                    T.utility.mxdFromShape(tensor2.shape()),
                    T.utility.mxdFromShape(newTensor.shape())
            );
            return newTensor;
        }

        public static T multiplication(T tensor1, T tensor2){
            T drn = new T(tensor1.shape());
            int[] index = new int[drn.shape().length];
            int size = drn.size();
            for(int i=0; i<size; i++){
                drn.e_add(index, tensor1.e_get(index)*tensor2.e_get(index));
                T.utility.increment(index, drn.shape());
            }
            return drn;
        }

        public static T addition(T tensor1, T tensor2){
            T drn = new T(tensor1.shape());
            int[] index = new int[drn.shape().length];
            int size = drn.size();
            for(int i=0; i<size; i++){
                drn.e_add(index, tensor1.e_get(index)+tensor2.e_get(index));
                T.utility.increment(index, drn.shape());
            }
            return drn;
        }

        public static T newTensor(double value, int[] shape){
            int sze = T.utility.szeOfShp(shape);
            T tensor = new T();
            tensor.value = new double[sze];
            tensor.initialShape(shape);
            for(int i=0; i<sze; i++){
                tensor.value[i] = value;
            }
            return tensor;
        }

        public static T newTensor(double[] value, int[] shape){
            T tensor = new T();
            tensor.value = value;
            tensor.initialShape(shape);
            return tensor;
        }
        public static T newTensor(double[] value, int[] shape, int[] translation){
            T tensor = new T();
            tensor.value = value;
            tensor.initialShape(shape);
            tensor.translation = translation;
            return tensor;
        }
        public static T newTensor(int[] shape, int[] translation){
            T tensor = new T();
            tensor.value = new double[T.utility.szeOfShp(shape)];
            tensor.initialShape(shape);
            tensor.translation = (translation!=null)?translation:tensor.translation;//SHARED.put()
            return tensor;
        }

        public static T copyOf(T tensor){
            T newTensor = new T();
            newTensor.shape = tensor.shape;
            newTensor.translation = tensor.translation;
            newTensor.value = new double[tensor.size()];
            newTensor.Modules = null;//tensor.Modules;
            newTensor.flags = tensor.flags;
            double[] value = tensor.value();
            for (int i = 0; i < value.length; i++) {
                newTensor.value[i] = value[i];
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
            newTensor.value = tensor.value;
            newTensor.shape = T.utility.reshaped(tensor.shape, newForm);
            newTensor.translation = T.utility.reshaped(tensor.translation, newForm);
            newTensor.Modules = tensor.Modules;//Reshaped derivs usw
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
            TDevice device = null;
            for (int ti = 0; ti < tsrs.length; ti++) {
                device = (tsrs[ti].isOutsourced())?(TDevice)tsrs[ti].find(TDevice.class):device;
            }
            if(device!=null) {
                for (int ti = 0; ti < tsrs.length; ti++) {
                    onSameGuestDevice = (device == tsrs[ti].find(TDevice.class)) && onSameGuestDevice;
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
        public static int[] shpOfTensMul(int[] frstShp, int[] scndShp) {
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
                }//e_set value in drn:
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
        public static int[][] resultMxdOf(int[][] shape1, int[][] shape2) {
            int[][] match = new int[4][(int) ((shape1[0].length + shape2[0].length) / 2)];
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
