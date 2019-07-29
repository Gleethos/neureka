package neureka.core;

import neureka.core.module.calc.FunctionFactory;
import neureka.core.module.calc.GraphBuilder;
import neureka.core.module.calc.TDevice;
import neureka.core.module.calc.gcomp.GradientNode;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.function.Consumer;

public class T {

    // DEFAULT DEVICE (HOST CPU)
    //=========================
    private static TDevice CPU;

    //STATIC SHARED MEMORY:
    //=========================
    private static HashMap<Long, int[]> SHARED;

    static {//The things we do for memory:
        SHARED = new HashMap<>();
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
        return value.length;
    }

    public int[] shpIdx(int idx) {
        return T.utility.IdxToShpIdx(idx, translation);
    }

    public boolean isEmpty() {
        if (value == null) {
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
    public void addModule(Object newModule) {
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
        for(int i=0; i<this.value.length; i++){
            strValue+=this.value[i];
            if(i<this.value.length-1){
                strValue+=", ";
            }
        }
        strValue = strShape+":("+strValue+")";
        if(mode=="r"){
            if(this.has(GradientNode.class)){
                GradientNode d = (GradientNode) this.find(GradientNode.class);
                String[] strDerivatives = {"; "};
                d.forEach((target, derivative)->{
                    strDerivatives[0]+="=>d|[ "+derivative.toString("r")+" ]|:t{ "+target.toString("r")+" }, ";
                });
                strValue += strDerivatives[0];
            }
        }else if(mode == "d"){
            if(this.has(GradientNode.class)){
                GradientNode d = (GradientNode) this.find(GradientNode.class);
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
        value = new double[T.utility.sizeOfShape_mxd(shape, 0, shape.length)];
        this.initialShape(shape);
    }
    public T(int[] shape, double value) {
        this.value = new double[T.utility.sizeOfShape_mxd(shape, 0, shape.length)];
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
        int size = T.utility.sizeOfShape_mxd(newShape, 0, newShape.length);
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

    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //MODIFICATION :
    //=========================
    public void reshape(int[] newForm) {
        this.shape = T.utility.reshaped(this.shape, newForm);
        this.translation = T.utility.reshaped(this.translation, newForm);
    }

    public void internalize(T tensor) {
        this.value = tensor.value;
        this.shape = tensor.shape;
        this.translation = tensor.translation;
        this.Modules = tensor.Modules;
        this.flags = tensor.flags;
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    //TRACKED COMPUTATION :
    //=========================
    public T f(T tensor, String operation) {
        if(tensor==null){return this;}
        return this.f(new T[]{tensor}, operation);
    }
    public T f(T[] tensors, String operation) {
        if(tensors==null||tensors.length==0||tensors[0]==null){return this;}
        this.internalize(FunctionFactory.newBuild(operation).activate(tensors));
        if(this.has(GradientNode.class)){
            ((GradientNode)this.find(GradientNode.class)).trimTree(null);
        }
        return this;
    }
    public T f(T[] tensors, int[][] translation, String operation) {
        this.internalize(FunctionFactory.newBuild(operation).activate(tensors));
        if(this.has(GradientNode.class)){
            ((GradientNode)this.find(GradientNode.class)).trimTree(null);
        }
        return this;
    }
    //-----------------------------------------------------------------------
    public void backward(T error) {
        if(this.rqsGradient()){
            this.setGradient(error);
        }
        if(this.has(GradientNode.class)){
            ((GradientNode)this.find(GradientNode.class)).backward(error);
        }
        //operation.backward(gradient, this);
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

    public double e_get(int i) {
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
            int sze = T.utility.sizeOfShape_mxd(shape, 0, shape.length);
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
            tensor.value = new double[T.utility.sizeOfShape_mxd(shape, 0, shape.length)];
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
            for (int i = 0; i < newTensor.value.length; i++) {
                newTensor.value[i] = tensor.value[i];
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
        public static int[] reshaped(int[] shape, @NotNull int[] newForm) {
            int[] newShp = new int[newForm.length];
            for (int i = 0; i < newForm.length; i++) {
                if (newForm[i] < 0) {
                    newShp[i] = Math.abs(newForm[i]);//dim[Math.abs(newForm[Di])-1]*-1;
                } else if (newForm[i] >= 0) {
                    newShp[i] = shape[newForm[i]];
                }
            }
            return newShp;
        }

        //-----------------------------------------------------------------------
        @Contract(pure = true)
        public static double[] randFromShape_mxd(int[] shape, int start, int rank, double[] data, int dataPtr) {
            int size = sizeOfShape_mxd(shape, start, rank);
            for (int i = 0; i < size; i++) {
                data[dataPtr + i] = Utility.getDoubleOf(i);
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
            int drnSze = sizeOfShape_mxd(drn[0], drn[3][0], rank);
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
        public static int sizeOfShape_mxd(int[] shape, int start, int rank) {
            int size = 1;
            int end = start + rank;
            for (int Di = 0; Di < end; Di++) {
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
