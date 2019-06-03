package neureka.main.core.base.data;

import neureka.main.core.modul.calc.NVFHead;
import neureka.main.core.modul.calc.NVFunction;

import java.util.function.Consumer;

import com.aparapi.Kernel;

public class TOperation {

    NVFunction Function = null;
    T[] Source = null;
    boolean doTensMul = false;

    int referenced = 0;
    int mode = 0;
    public boolean usesAD(){return (mode !=0);}
    public boolean usesForwardAD(){ return (mode >0); }
    public boolean usesReverseAD(){ return (mode <0); }
    /**
     *  modes:    |
     *  ----------+----------------------------------+-
     *  mode == 0 | no Auto-Differentiation          |
     *  ----------+----------------------------------+-
     *  mode > 0  | forward Auto-Differentiation     |
     *  ----------+----------------------------------+-
     *  mode < 0  | backward Auto-Differentiation    |
     *  ----------+----------------------------------+-
     * */

    /**
     *   CONSTRUCTOR:
     * */
    TOperation(T drain, T[] source, int[][] translation, String operation){
        T[] translated = new T[source.length];
        for(int i=0; i<translated.length&&i<translation.length; i++){
            translated[i] = T.factory.reshapedCopyOf(source[i], translation[i]);//source[i].reshaped(translation[i]);
        }
    }
    TOperation(T drain, T[] source, String operation){
        construct(drain, source, operation);
    }
    private void construct(T drain, T[] source, String operation){
        Source = source;
        doTensMul = false;//tensmul:
        String replacement = "I[0]";
        if(operation.contains("tensmul")){
            if(operation.contains("tensmul(Ij)")){
                operation = operation.replace("tensmul(Ij)", replacement);
                doTensMul = true;
            }else {
                operation = operation.replace("tensmul", replacement);
                doTensMul = true;
            }
        }
        if(operation.contains("tensDotMul")){
            if(operation.contains("tensDotMul(Ij)")){
                operation = operation.replace("tensDotMul(Ij)", replacement);
                doTensMul = true;
            }else {
                operation = operation.replace("tensDotMul", replacement);
                doTensMul = true;
            }
        }
        if(operation.contains("tm")){
            if(operation.contains("tm(Ij)")){
                operation = operation.replace("tm(Ij)", replacement);
                doTensMul = true;
            }else{
                operation = operation.replace("tm", replacement);
                doTensMul = true;
            }
        }
        if(doTensMul){
            operation = operation
                .replace("Ij", "I[0]")
                .replace("Ii", "I[0]")
                .replace("I[i]", "I[0]")
                .replace("I[j]", "I[0]");
        }
        Function = new NVFHead();
        Function = Function.newBuild(operation);
        /**
         *   Evaluating validity:
         * */
        if(!this.doTensMul){
            if(!utility.isValid(this.Source, this.doTensMul)){
                this.Function = null;
                this.Source = null;
                return;
            }
        }else if(this.doTensMul && !utility.isValid(this.Source, this.doTensMul)){//Shape fitting:
            int largest = 0;
            for(int i=0; i<this.Source.length; i++){
                largest = (this.Source[i].shape().length>largest)?this.Source[i].shape().length:largest;
            }
            int[][] newShapes = new int[this.Source.length][largest];
            for(int i=0; i<this.Source.length; i++){
                for(int ii=0; ii<largest; ii++){
                    newShapes[i][ii] = (ii<this.Source[i].shape().length)?ii:-1;
                }
                this.Source[i].reshape(newShapes[i]);
            }
        }
        /**
         *  Increment reference counter
         *              &
         *  Evaluate auto-grad mode:
         * */
        this.mode = 0;
        int[] srcModes = new int[this.Source.length];
        int m = 0;
        for(int Ii = 0; Ii< this.Source.length; Ii++){
            if(Source[Ii].hasModule(TOperation.class)){
                TOperation junctor = (TOperation)Source[Ii].findModule(TOperation.class);
                junctor.referenced++;
                srcModes[Ii] = junctor.mode;
            }else if(Source[Ii].rqsGradient()){
                srcModes[Ii] = 1;
            }
            m += (srcModes[Ii]!=0)?1:0;
        }
        if(m==1){
            for(int Ii = 0; Ii< this.Source.length; Ii++){
                mode += (srcModes[Ii]<0)?1:srcModes[Ii];
            }
        }else if(m>1){
            mode = -m;
        }
        /**
         *  forward:
         * */
        this.forward(drain);
    }

    private void forward(T drain){
        if(Source==null || (Function==null && !doTensMul)){ return; }
        //boolean calculateDerivatives = (this.mode!=0);
        if(doTensMul){// ((((((A B)C)D)E)G)F)
            T temp =null;
            for(int i = 0; i< Source.length; i++){
                T result = Source[i];
                T second = Source[i];
                T first = temp;
                if(i>0){
                    result = T.factory.tensDotMul(first, second);
                    if(this.usesAD())
                    {//--------------------------------------------------------------------------------------
                        if(this.usesForwardAD()){
                            RelativeGradients gFirst = (RelativeGradients) first.findModule(RelativeGradients.class);
                            RelativeGradients gSecond = (RelativeGradients) second.findModule(RelativeGradients.class);
                            if(gFirst!=null||gSecond!=null){
                                result.addModule(new RelativeGradients());
                            }
                            RelativeGradients gResult = (RelativeGradients) result.findModule(RelativeGradients.class);
                            if(gFirst!=null && gSecond==null){
                                gFirst.forEach((T src, T derivative)-> {
                                    gResult.put(src, T.factory.tensDotMul(derivative, second));
                                });
                            }
                            if(gSecond!=null && gFirst==null){
                                gSecond.forEach((T src, T derivative)-> {
                                    gResult.put(src, T.factory.tensDotMul(derivative, first));
                                });
                            }else if(gSecond!=null && gFirst!=null){
                                gFirst.forEach((src, derivative)-> {
                                    gResult.put(src, T.factory.tensDotMul(derivative, second));
                                });
                                gSecond.forEach((src, derivative)-> {
                                    if(gResult.has(src)){
                                        gResult.get(src).e_add(T.factory.tensDotMul(derivative, first));
                                    }else{
                                        gResult.put(src, T.factory.tensDotMul(derivative, first));
                                    }
                                });
                            }
                        }else{
                            result.addModule(new RelativeGradients());
                            RelativeGradients gResult = (RelativeGradients) result.findModule(RelativeGradients.class);
                            if(first.hasModule(RelativeGradients.class)){
                                gResult.put(first, T.factory.tensDotMul(new T(first.shape(), 1), second));
                            }
                            if(second.hasModule(RelativeGradients.class)){
                                gResult.put(second, T.factory.tensDotMul(new T(second.shape(), 1), first));
                            }
                        }
                    }//--------------------------------------------------------------------------------------
                }
                temp = result;
            }
            if(Function!=null){
                this.activationOn(new T[]{temp}, drain);
            }else{
                drain.internalize(temp);
            }
        }else{
            if(Function!=null){
                this.activationOn(Source, drain);
            }
        }
        //--------------------------------------------------------------------------------------
        if(this.usesAD() && this.usesForwardAD()){
            RelativeGradients selfGradients = (RelativeGradients) drain.findModule(RelativeGradients.class);
            if(drain.rqsGradient()){
                if(selfGradients==null){
                    selfGradients = new RelativeGradients();
                    drain.addModule(selfGradients);
                }
                selfGradients.put(drain, new T(drain.shape(), 1));
            }
        }
        //--------------------------------------------------------------------------------------
    }

    private void activationOn(T[] source, T drain){
        if(drain.isEmpty()){
            drain.initialShape(source[0].shape());
        }
        for(int[] idx = {0}; idx[0]<drain.value().length; idx[0]++){
            //--------------------------------------------------------------------------------------
            //Thread t = new Thread(() -> {
                double[] input = new double[source.length];
                elementary(source, drain, idx[0], input, false);
                if(this.usesAD()){
                    elementary(source, drain, idx[0], input, true);
                }
           // });
            //--------------------------------------------------------------------------------------
        }// Idx loop closed!
    }
    private void elementary(T[] source, T drain, int idx, double[] input, boolean derive){
        if(derive==false) {//-------------------------------------------------------------
            for (int Ii = 0; Ii < input.length; Ii++) {
                input[Ii] = source[Ii].e_get(source[Ii].shpIdx(idx));
            }
            drain.value()[idx] = Function.activate(input);
        }else{//----------------------------------------------------------------
            if(!drain.hasModule(RelativeGradients.class)){
                drain.addModule(new RelativeGradients());
            }
            RelativeGradients drainDeriv = (RelativeGradients) drain.findModule(RelativeGradients.class);
            double[] d_input = new double[source.length];
            for(int Ii = 0; Ii<source.length; Ii++){
                RelativeGradients relDeriv = (RelativeGradients) source[Ii].findModule(RelativeGradients.class);
                if(relDeriv!=null){
                    d_input[Ii] = Function.derive(input, Ii);
                }
            }
            for(int Ii = 0; Ii< source.length; Ii++){
                RelativeGradients srcDerivatives =
                    (this.usesReverseAD() && (source[Ii].rqsGradient()||(source[Ii].hasModule(TOperation.class)&&((TOperation)source[Ii].findModule(TOperation.class)).usesAD())))
                        ?drainDeriv
                        :(RelativeGradients) source[Ii].findModule(RelativeGradients.class);
                if(srcDerivatives!=null){
                    if(d_input[Ii]!=0){
                        int[] idx_enc = {idx};
                        double[] d_input_enc = {d_input[Ii]};
                        srcDerivatives.forEach(
                            (target, derivative)->{
                                T found = drainDeriv.get(target);
                                if(found==null){
                                    found = T.factory.copyOf(derivative);
                                    drainDeriv.put(target, found);
                                }
                                found.e_mul(idx_enc[0], d_input_enc[0]);
                            }
                        );
                    }
                }
            }
        }
    }

    public void backward(T error, T drain){
        if(!this.usesAD()){
            return;
        }
        RelativeGradients drnGradients = (RelativeGradients) drain.findModule(RelativeGradients.class);
        drnGradients.forEach((target, g)->{
            target.backward(T.factory.tensMul(error, g));
        });

    }

    private void exec() {
        Kernel kernel = new Kernel(){
            @Override public void run(){
                int i= getGlobalId();//result[i]=intA[i]+inB[i];
            }
        };
        //Range range = Range.create(result.length);
        //kernel.execute(range);
    }

    /**
     *   tensor3 = new T().of(new TOperation(new T[]{tensor1, tensor2}, "tensDotMul"));
     *   tensor4 = new TOperation(tensor3, "sum(tanh[Ij])").out();
     *
     *
     * */

    private  static class utility{

        private static boolean isValid(T[] source, boolean doingTensMul){
            T current = null;
            T last = null;
            for(int i=0; i<source.length; i++){
                current = source[i];
                if(i>0){
                    if(current.shape().length!=last.shape().length){
                        return false;
                    }
                    if(!doingTensMul){
                        for(int j=0; i<current.shape().length; j++){
                            if(current.shape()[j]!=last.shape()[j]){
                                return false;
                            }
                        }
                    }
                }
                last = current;
            }
            return true;
        }




    }

}



