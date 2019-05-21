package neureka.main.core.base.data;

import neureka.main.core.NVUtility;
import org.jetbrains.annotations.Contract;
import org.jetbrains.annotations.NotNull;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Random;

public class NTensor {

    private static HashMap<Long, int[]> shared;

    static {//The thing we do for memory:
        shared = new HashMap<Long, int[]>();
    }

    private long id = new Random().nextLong();
    private double[] value = null;
    private int[] shape = null;
    private int[] translation = null;

    //-------------------------------------------------------------
    private int Flags = 0 + 2;//Default
    private final static int rqsGradient_MASK = 1;
    private final static int carriesDerivatives_MASK = 2;
    private final static int carriesBackwardRoutes_MASK = 4;
    private final static int srcDrained_MASK = 0;
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public boolean srcDrained() {
        return ((Flags & srcDrained_MASK) == srcDrained_MASK) ? true : false;
    }
    public void setSrcDrained(boolean srcDrained) {
        if (srcDrained() != srcDrained) {
            if (srcDrained) {
                Flags += srcDrained_MASK;
            } else {
                Flags -= srcDrained_MASK;
            }
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public boolean carriesBackwardRoutes() {
        return ((Flags & carriesBackwardRoutes_MASK) == carriesBackwardRoutes_MASK) ? true : false;
    }
    public void setCarriesBackwardRoutes(boolean caryBackwardRoutes) {
        if (carriesBackwardRoutes() != caryBackwardRoutes) {
            if (caryBackwardRoutes) {
                Flags += carriesBackwardRoutes_MASK;
            } else {
                Flags -= carriesBackwardRoutes_MASK;
            }
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public boolean carriesDerivatives() {
        return ((Flags & carriesDerivatives_MASK) == carriesDerivatives_MASK) ? true : false;
    }
    public void setCarriesDerivatives(boolean caryDerivatives) {
        if (carriesDerivatives() != caryDerivatives) {
            if (caryDerivatives) {
                Flags += carriesDerivatives_MASK;
            } else {
                Flags -= carriesDerivatives_MASK;
            }
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public boolean rqsGradient() {
        return ((Flags & rqsGradient_MASK) == rqsGradient_MASK) ? true : false;
    }
    public void setRqsGradient(boolean rqsGradient) {
        if (rqsGradient() != rqsGradient) {
            if (rqsGradient) {
                Flags += rqsGradient_MASK;
            } else {
                Flags -= rqsGradient_MASK;
            }
        }
        if (rqsGradient) {
            if (!this.hasModule(NDerivatives.class)) {
                NDerivatives d = new NDerivatives();
                d.put(this, new NTensor(this.shape, 1));
                this.addModule(d);
            }
        }
    }
    //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    public String toString(){
        if(this.isEmpty()){
            return "empty";
        }
        String strShape = "";
        for(int i=0; i<this.shape.length; i++){
            strShape+=this.shape[i];
            if(i<this.shape.length-1){
                strShape+="X";
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
        if(this.hasModule(NDerivatives.class)){
            NDerivatives d = (NDerivatives) this.findModule(NDerivatives.class);
            String[] strDerivatives = {"; "};
            d.forEach((target, derivative)->{
                strDerivatives[0]+="->d"+derivative.toString()+", ";
            });
            strValue += strDerivatives[0];
        }
        return strValue;
    }
    //MODUL I / O :
    //=========================
    private ArrayList<Object> Modules = new ArrayList<Object>();

    public ArrayList<Object> getModules() {
        return Modules;
    }

    public void setModules(ArrayList<Object> properties) {
        Modules = properties;
    }

    public void addModule(Object newModule) {
        if (Modules != null) {
            Object oldCompartment = findModule(newModule.getClass());
            if (oldCompartment != null) {
                Modules.remove(oldCompartment);
                Modules.trimToSize();
            }
            Modules.add(newModule);
        } else {
            Modules = new ArrayList<Object>();
            Modules.add(newModule);
        }
    }

    public Object findModule(Class moduleClass) {
        if (Modules != null) {
            for (int Pi = 0; Pi < Modules.size(); Pi++) {
                if (moduleClass.isInstance(Modules.get(Pi))) {
                    return Modules.get(Pi);
                }
            }
        }
        return null;
    }

    public void removeModule(Class moduleClass) {
        Object oldCompartment = findModule(moduleClass);
        if (oldCompartment != null) {
            Modules.remove(oldCompartment);
            Modules.trimToSize();
        }
        if (Modules.size() == 0) {
            Modules = null;
        }
    }

    public boolean hasModule(Class moduleClass) {
        if (findModule(moduleClass) != null) {
            return true;
        }
        return false;
    }

    //================================================================================================
    NTensor(int[] shape) {
        value = new double[NTensor.utility.sizeOfShape_mxd(shape, 0, shape.length)];
        this.initialShape(shape);
    }

    public NTensor(int[] shape, double value) {
        this.value = new double[NTensor.utility.sizeOfShape_mxd(shape, 0, shape.length)];
        this.initialShape(shape);
        for (int i = 0; i < this.value.length; i++) {
            this.value[i] = value;
        }
    }

    public NTensor(NTensor tensor) {
        this.shape = tensor.shape;
        this.translation = tensor.translation;
        this.value = new double[tensor.size()];
        this.Modules = null;//tensor.Modules;
        this.Flags = tensor.Flags;
        for (int i = 0; i < this.value.length; i++) {
            this.value[i] = tensor.value[i];
        }
    }


    public NTensor() {
    }

    public void initialShape(int[] newShape) {
        int size = NTensor.utility.sizeOfShape_mxd(newShape, 0, newShape.length);
        if (value == null) {
            value = new double[size];
        }
        if (size != value.length) {
            return;
        }
        translation = NTensor.utility.idxTrltn(newShape);
        long key = 0;
        for (int i = 0; i < newShape.length; i++) {
            key *= 10;
            key += Math.abs(newShape[i]);
        }
        int[] found = shared.get(key);
        if (found != null) {
            this.shape = found;
        } else {
            shared.put(key, newShape);
            this.shape = newShape;
        }
    }

    public long id() {
        return id;
    }

    public double[] value() {
        return value;
    }

    public int[] shape() {
        return shape;
    }

    public int size() {
        return value.length;//NTensor.utility.sizeOfShape_mxd(shape, 0, shape.length);
    }

    public int[] shpIdx(int idx) {
        return NTensor.utility.IdxToShpIdx(idx, translation);
    }

    public boolean isEmpty() {
        if (value == null) {
            return true;
        }
        return false;
    }

    public void reshape(int[] newForm) {
        this.shape = NTensor.utility.reshaped(this.shape, newForm);
        this.translation = NTensor.utility.reshaped(this.translation, newForm);
    }

    public void copy(NTensor tensor) {
        this.value = tensor.value;
        this.shape = tensor.shape;
        this.translation = tensor.translation;
        this.Modules = tensor.Modules;
        this.Flags = tensor.Flags;
    }

    //================================================================================================
    public NTensor of(NTensor tensor, String operation) {
        this.addModule(new NOperation(tensor, operation));
        return this;
    }

    public NTensor of(NTensor[] tensors, String operation) {
        this.addModule(new NOperation(tensors, operation));
        return this;
    }

    public NTensor of(NTensor[] tensors, int[][] translation, String operation) {
        this.addModule(new NOperation(tensors, translation, operation));
        return this;
    }

    public void forward() {
        NOperation operation = (NOperation) this.findModule(NOperation.class);
        if (operation == null) {
            return;
        }
        operation.forEachSource((tensor) -> {
            tensor.forward();
        });
        operation.forward(this);
    }

    // Element wise operations:
    //--------------------------
    public double e_get(int idx) {
        return NTensor.utility.idxOfShpIdxAndShp(shpIdx(idx), shape);
    }

    public double e_get(int[] index) {
        return value[NTensor.utility.idxOfShpIdxAndShp(index, shape)];
    }

    public void e_set(int idx, double value) {
        this.value[NTensor.utility.idxOfShpIdxAndShp(shpIdx(idx), shape)] = value;
    }

    public void e_set(int[] index, double value) {
        this.value[NTensor.utility.idxOfShpIdxAndShp(index, shape)] = value;
    }

    public void e_add(int idx, double value) {
        this.value[NTensor.utility.idxOfShpIdxAndShp(shpIdx(idx), shape)] += value;
    }

    public void e_add(int[] index, double value) {
        this.value[NTensor.utility.idxOfShpIdxAndShp(index, shape)] += value;
    }

    public void e_sub(int idx, double value) {
        this.value[NTensor.utility.idxOfShpIdxAndShp(shpIdx(idx), shape)] -= value;
    }

    public void e_sub(int[] index, double value) {
        this.value[NTensor.utility.idxOfShpIdxAndShp(index, shape)] -= value;
    }

    public void e_mul(int idx, double value) {
        this.value[NTensor.utility.idxOfShpIdxAndShp(shpIdx(idx), shape)] *= value;
    }

    public void e_mul(int[] index, double value) {
        this.value[NTensor.utility.idxOfShpIdxAndShp(index, shape)] *= value;
    }

    public void e_add(NTensor tensor) {
        int[] index = new int[shape.length];
        int size = size();
        for (int i = 0; i < size; i++) {
            e_add(index, tensor.e_get(index));
            NTensor.utility.increment(index, shape);
        }
    }

    public void e_sub(NTensor tensor) {
        int[] index = new int[shape.length];
        int size = size();
        for (int i = 0; i < size; i++) {
            this.e_sub(index, tensor.e_get(index));
            NTensor.utility.increment(index, shape);
        }
    }

    public static class factory{
        public static NTensor newTensor(double[] value, int[] shape){
            NTensor tensor = new NTensor();
            tensor.value = value;
            tensor.initialShape(shape);
            return tensor;
        }
        public static NTensor copyOf(NTensor tensor){
            NTensor newTensor = new NTensor();
            newTensor.shape = tensor.shape;
            newTensor.translation = tensor.translation;
            newTensor.value = new double[tensor.size()];
            newTensor.Modules = null;//tensor.Modules;
            newTensor.Flags = tensor.Flags;
            for (int i = 0; i < newTensor.value.length; i++) {
                newTensor.value[i] = tensor.value[i];
            }
            return newTensor;
        }
        public static NTensor copyOf(Object[] things){

            for(int i=0; i<things.length; i++){
                if(things[i] instanceof int[]){

                }

            }
            return new NTensor();
        }
        public static  NTensor reshapedCopyOf(NTensor tensor, int[] newForm) {
            NTensor newTensor = new NTensor();
            newTensor.value = tensor.value;
            newTensor.shape = NTensor.utility.reshaped(tensor.shape, newForm);
            newTensor.translation = NTensor.utility.reshaped(tensor.translation, newForm);
            newTensor.Modules = tensor.Modules;
            return newTensor;
        }
    }

    public static class utility {
        @Contract(pure = true)
        public static void increment_mxd(@NotNull int[] dimIndex, @NotNull int[] dim, int start, int rank) {
            int Di = start;
            int end = start + rank;
            while (Di >= start && Di < end) {
                Di = incrementAt(Di, dimIndex, dim);
            }
        }

        @Contract(pure = true)
        public static void increment(@NotNull int[] dimIndex, @NotNull int[] dim) {
            int Di = 0;
            while (Di >= 0 && Di < dim.length) {//fixed
                Di = incrementAt(Di, dimIndex, dim);
            }
        }

        @Contract(pure = true)
        public static int incrementAt(int Di, @NotNull int[] dimIndex, @NotNull int[] dim) {
            if (dimIndex[Di] < (dim[Di])) {//fixed
                dimIndex[Di]++;
                if (dimIndex[Di] == (dim[Di])) {
                    dimIndex[Di] = 0;
                    Di++;
                } else {
                    Di = -1;
                }
            } else {
                Di++;
            }
            return Di;
        }

        //-----------------------------------------------------------------------
        @Contract(pure = true)
        public static void decrement_onMixed(@NotNull int[] dimIndex, @NotNull int[] dim, int start, int rank) {
            int Di = start;
            int end = start + rank;
            while (Di >= start && Di < end) {
                Di = decrementAt(Di, dimIndex, dim);
            }
        }

        @Contract(pure = true)
        public static void decrement(@NotNull int[] dimIndex, @NotNull int[] dim) {
            int Di = 0;
            while (Di >= 0 && Di < dim.length) {
                Di = decrementAt(Di, dimIndex, dim);
            }
        }

        @Contract(pure = true)
        public static int decrementAt(int Di, @NotNull int[] dimIndex, @NotNull int[] dim) {
            if (dimIndex[Di] == 0) {
                Di++;
            } else {
                dimIndex[Di]--;
                Di--;
                while (dimIndex[Di] == 0) {
                    dimIndex[Di] = dim[Di] - 1;
                    Di--;
                }
                Di = -1;
            }
            return Di;
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
        public static int idxOfShpIdxAndShp(int[] Index, int[] shape) {
            int[] trltn = idxTrltn(shape);
            int idx = 0;
            for (int i = 0; i < trltn.length; i++) {
                idx += trltn[i] * Index[i];
            }
            return idx;
        }

        @Contract(pure = true)
        public static int[] idxTrltn(int[] dim) {
            int[] idxAnchor = new int[dim.length];
            int prod = 1;
            for (int Di = 0; Di < dim.length; Di++) {
                idxAnchor[Di] = prod;
                prod *= dim[Di];
            }
            return idxAnchor;
        }

        @Contract(pure = true)
        public static int[] idxTrnln_Mxd(int[] mxdShp, int[] trnln, int start, int end) {
            int prod = 1;
            for (int Di = start; Di < end; Di++) {
                trnln[Di] = prod;
                prod *= mxdShp[Di];
            }
            return trnln;
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

        @Contract(pure = true)
        public static int[] IdxToShpIdx_mxd(int idx, int[] anchor, int[] shpIdx, int start, int rank) {
            for (int i = (start + rank) - 1; i >= 0; i--) {
                int r = idx % anchor[start + i];
                shpIdx[start + i] = (idx - r) / anchor[start + i];
                idx = r;
            }
            return shpIdx;
        }

        //-----------------------------------------------------------------------
        @Contract(pure = true)
        public static int[] reshaped(int[] shape, @NotNull int[] newForm) {
            int[] newDim = new int[newForm.length];
            for (int Di = 0; Di < newForm.length; Di++) {
                if (newForm[Di] < 0) {
                    newDim[Di] = Math.abs(newForm[Di]);//dim[Math.abs(newForm[Di])-1]*-1;
                } else if (newForm[Di] >= 0) {
                    newDim[Di] = shape[newForm[Di]];
                }
            }
            return newDim;
        }

        @Contract(pure = true)
        public static int[] reshaped_mxd(int[] shape, int start, int rank, int[] newDim, @NotNull int[] newForm) {
            //int[] newDim = new int[newForm.length];
            for (int Di = 0; Di < newForm.length; Di++) {//!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                if (newForm[Di] < 0) {
                    newDim[Di] = Math.abs(newForm[Di]);//dim[Math.abs(newForm[Di])-1]*-1;
                } else if (newForm[Di] >= 0) {
                    newDim[Di] = shape[newForm[Di]];
                }
            }
            return newDim;
        }

        //-----------------------------------------------------------------------
        @Contract(pure = true)
        public static double[] randFromShape_mxd(int[] shape, int start, int rank, double[] data, int dataPtr) {
            int size = sizeOfShape_mxd(shape, start, rank);
            for (int i = 0; i < size; i++) {
                data[dataPtr + i] = NVUtility.getDoubleOf(i);
            }
            return data;
        }

        @Contract(pure = true)
        public static int[][] reshapedAndToMxd(int[] shape, int[] newShp) {
            return mxdFromShape(reshaped(shape, newShp), reshaped(idxTrltn(shape), newShp));
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
            return mxdFromShape(shape, idxTrnln_Mxd(shape, new int[rank], 0, rank));
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
            int[] shape = new int[(frstShp.length + scndShp.length + 1) / 2];
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
                        int[][] hdr1, int[][] hdr2, int[][] drn
                ) {
            //hdr[0] => dim[]
            //hdr[1] => anchor[]
            //hdr[2] => idx[]
            //hdr[3] => {start}
            int hdr1End = hdr1[3][0] + rank;
            int hdr2End = hdr2[3][0] + rank;
            int drnEnd = drn[3][0] + rank;
            int drnSze = sizeOfShape_mxd(drn[0], drn[3][0], rank);
            int i = 0;
            while (i < drnSze) {
                //increment of and drain accordingly:
                int i1 = hdr1[3][0];
                int i2 = hdr2[3][0];
                int id = drn[3][0];
                int ri = 0;
                while (ri < rank) {
                    if (hdr1[0][i1] == hdr2[0][i2]) {//setting 0
                        hdr1[2][i1] = drn[2][id];//mtch[mi];
                        hdr2[2][i2] = drn[2][id];//mtch[mi];
                    } else if (hdr1[0][i1] > hdr2[0][i2]) {//setting hdr1 idx to id idx
                        hdr1[2][i1] = drn[2][id];//mtch[mi];
                        hdr2[2][i2] = 0;
                    } else if (hdr1[0][i1] < hdr2[0][i2]) {//setting hdr2 idx to id idx
                        hdr1[2][i1] = 0;
                        hdr2[2][i2] = drn[2][id];//mtch[mi];
                    }
                    i1++;
                    i2++;
                    id++;
                    ri++;
                }
                //----------
                // multiply:
                double value = 0;
                boolean running = true;
                boolean incrementing = false;
                while (running) {
                    if (i1 == hdr1End || i2 == hdr2End || id == drnEnd) {
                        i1 = hdr1[3][0];
                        i2 = hdr2[3][0];
                        id = drn[3][0];
                    }
                    if (incrementing == false) {
                        int idx1 = idxOfFrmt_mxd(hdr1, rank);
                        int idx2 = idxOfFrmt_mxd(hdr2, rank);
                        System.out.println(
                                "hdr1:" + strInt(hdr1[2]) +
                                        "; hdr2:" + strInt(hdr2[2]) +
                                        "; drn:" + strInt(drn[2]) +
                                        " idx1:(" + idx1 + ");" +
                                        " idx2:(" + idx2 + ");" +
                                        " drn:(" + idxOfFrmt_mxd(drn, rank) + ");" +
                                        " val:(" + value + ") += val1:(" + data[0][dataPtr[0] + idx1] + ") x val2:(" + data[1][dataPtr[1] + idx2] + ");");
                        value += data[0][dataPtr[0] + idx1] * data[1][dataPtr[1] + idx2];
                        incrementing = true;
                        i1 = hdr1[3][0];
                        i2 = hdr2[3][0];
                        id = drn[3][0];
                    } else {//incrementing:
                        if (hdr1[2][i1] < hdr1[0][i1] && hdr2[2][i2] < hdr2[0][i2]) {
                            hdr1[2][i1]++;
                            hdr2[2][i2]++;
                            if (hdr1[2][i1] == hdr1[0][i1] || hdr2[2][i2] == hdr2[0][i2]) {
                                if ((i1 == (hdr1End - 1) || i2 == (hdr2End - 1))) {
                                    running = false;
                                }
                                if (hdr1[0][i1] == hdr2[0][i2]) {//setting 0
                                    hdr1[2][i1] = drn[2][id];//mtch[mi];
                                    hdr2[2][i2] = drn[2][id];//mtch[mi];
                                } else if (hdr1[0][i1] > hdr2[0][i2]) {//setting hdr1 idx to id idx
                                    hdr1[2][i1] = drn[2][id];//mtch[mi];
                                    hdr2[2][i2] = 0;
                                } else if (hdr1[0][i1] < hdr2[0][i2]) {//setting hdr2 idx to id idx
                                    hdr1[2][i1] = 0;
                                    hdr2[2][i2] = drn[2][id];//mtch[mi];
                                }
                                i1++;
                                i2++;
                                id++;
                            } else {
                                incrementing = false;
                                i1 = hdr1[3][0];
                                i2 = hdr2[3][0];
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
            System.out.println("wow");
            System.out.println(strInt(hdr1[2]) + "-" + strInt(hdr1[0]) + "-" + strInt(hdr1[1]));
            System.out.println(strInt(hdr2[2]) + "-" + strInt(hdr2[0]) + "-" + strInt(hdr2[1]));
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
            match[1] = NTensor.utility.idxTrltn(match[0]);
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
         * 	int[] selfDim = findModule(int[].class);
         * 	//[Ii][Ni][0]=>weightDim [1]=>form
         *	int[][] FormData = findModule(int[][][][].class)[Ii][Ni]
         *	int[] weightDim = FormData[0];
         *	int[] connForm = FormData[1];
         *
         * */


    }


}
