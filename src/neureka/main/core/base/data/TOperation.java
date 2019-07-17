package neureka.main.core.base.data;

import neureka.main.core.modul.calc.FunctionConstructor;

import java.util.function.Consumer;

public class TOperation {

    neureka.main.core.modul.calc.Function Function = null;
    T[] Source = null;
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
    public TOperation(T drain, T[] source, int[][] translation, String operation){
        T[] translated = new T[source.length];
        for(int i=0; i<translated.length&&i<translation.length; i++){
            translated[i] = T.factory.reshapedCopyOf(source[i], translation[i]);//source[i].reshaped(translation[i]);
        }
    }
    public TOperation(T drain, T[] source, String operation, boolean forward, boolean tipReached){//, boolean derive
        this.Source = source;
        construct(operation, source, tipReached);
        prepare(drain, source, true);
        if(forward){
            forward(drain);
        }
        if(this.Function.isFlat()&&tipReached){//derive &&
            performDifferentiation(drain);
        }
    }
    public TOperation(T drain, T[] source, int f_id, boolean forward, boolean tipReached){//, boolean derive
        construct(f_id, source, tipReached);
        prepare(drain, source, forward);
        if(forward){
            forward(drain);
        }
        if(this.Function.isFlat()&&tipReached){
            performDifferentiation(drain);
        }
    }

    public TOperation(T drain, T[] source, String operation, boolean forward){//, boolean derive
        construct(operation, source, false);
        prepare(drain, source, true);
        if(forward){
            forward(drain);
        }
        if(this.Function.isFlat()){//derive &&
            performDifferentiation(drain);
        }
    }
    private void construct(int f_id, T[] source, boolean tipReached){
        this.Source = source;
        this.Function = new FunctionConstructor().newBuild(f_id, source.length, tipReached);
    }
    private void construct(String operation, T[] source, boolean tipReached) {
        this.Source = source;
        String replacement = "I[0]";
        for(int i=0; i<(source.length-1); i++){
            replacement+="xI["+(i+1)+"]";
        }
        if (operation.contains("tensmul")) {
            if (operation.contains("tensmul(Ij)")) {
                operation = operation.replace("tensmul(Ij)", replacement);
            } else {
                operation = operation.replace("tensmul", replacement);
            }
        }
        if (operation.contains("convolution")) {
            if (operation.contains("convolution(Ij)")) {
                operation = operation.replace("convolution(Ij)", replacement);
            } else {
                operation = operation.replace("convolution", replacement);
            }
        }
        if (operation.contains("tm")) {
            if (operation.contains("tm(Ij)")) {
                operation = operation.replace("tm(Ij)", replacement);
            } else {
                operation = operation.replace("tm", replacement);
            }
        }
        Function = new FunctionConstructor().newBuild(operation, tipReached);
        validate(operation);
    }
    private void validate(String operation){
        /**
         *   Evaluating validity:
         * */
        if(!operation.contains("x")){
            if(!utility.isValid(this.Source, false)){
                this.Function = null;
                this.Source = null;
                return;
            }
        }else if(operation.contains("x") && !utility.isValid(this.Source, true)){//Shape fitting:
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
    }

    private void prepare(T drain, T[] source, boolean forward){
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
    }

    private void forward(T drain){
        if(Source==null || (Function==null)){ return; }
        //}else{
            if(Function!=null){
                drain.internalize(Function.activate(Source));
                //this.activationOn(Source, drain);
            }
        //}

    }

    private void performDifferentiation(T drain){
        //--------------------------------------------------------------------------------------
        if(this.usesAD() && Function.isFlat()){
            if(!drain.hasModule(RelativeGradients.class)){//&& (drain.rqsGradients()||src.rqsGradients())
                RelativeGradients rg = new RelativeGradients();
                drain.addModule(rg);
            }
            RelativeGradients selfGradients = (RelativeGradients) drain.findModule(RelativeGradients.class);
            /**
             *  Preparing for backpropagation:
             * */
            if(this.usesForwardAD()){
                if(drain.rqsGradient()){
                    selfGradients.put(drain, new T(drain.shape(), 1));
                }
                int[] i = {0};
                this.foreach((src)->{
                    RelativeGradients src_gradients = (RelativeGradients) src.findModule(RelativeGradients.class);
                    if(src_gradients!=null){
                        T d = Function.derive(this.Source, i[0]);
                        if(
                            drain.rqsGradient()
                            ||
                            drain.hasModule(TOperation.class)
                            && ((TOperation)drain.findModule(TOperation.class)).mode>0
                        ){
                            src_gradients.forEach(
                            (t, g)->{
                                /**
                                 *  Chain rule for every gradient with respect to leaves:
                                 * */
                                if(selfGradients.has(t)){
                                    T sg = selfGradients.get(t);
                                    if(Function.toString().contains("x")){
                                        selfGradients.put(t, T.factory.addition(sg,T.factory.convolution(d, g)));
                                    }else{
                                        selfGradients.put(t, T.factory.addition(sg,T.factory.multiplication(d, g)));
                                    }
                                }else{
                                    if(Function.toString().contains("x")){
                                        selfGradients.put(t, T.factory.convolution(d, g));
                                    }else{
                                        selfGradients.put(t, T.factory.multiplication(d, g));
                                    }
                                }
                                //TODO: flag within sr
                                // c tsrs that grant that the tensor has been created by function constructor!
                            });
                        }else{
                            selfGradients.put(src, d);
                        }

                    }
                    i[0]++;
                });

            }else if(this.usesReverseAD()){
                int[] i = {0};
                this.foreach((src)->{
                    RelativeGradients gSrc = (RelativeGradients) src.findModule(RelativeGradients.class);
                    if(gSrc!=null){
                        T d = Function.derive(this.Source, i[0]);
                        selfGradients.put(src, d);// Add gradients with respect to every source tensor!
                    }
                    i[0]++;
                });
            }
        }
        //--------------------------------------------------------------------------------------
    }

    private void foreach(Consumer<T> action){
        for(int i=0; i<this.Source.length; i++){
            action.accept(this.Source[i]);
        }
    }

    public void backward(T error, T drain){
        if(!this.usesAD()){
            return;
        }
        if(drain.rqsGradient()){
            drain.setGradient(
                    (drain.gradient()==null)
                            ?error
                            :T.factory.addition(error, T.factory.newTensor(drain.value(), drain.shape()))
            );
        }
        RelativeGradients drnGradients = (RelativeGradients) drain.findModule(RelativeGradients.class);
        if(drnGradients!=null){
            drnGradients.forEach((target, g)->{
             //   target.backward(T.factory.multiplication(error, g));
            });
        }
    }

    /**
     *   tensor3 = new T().of(new TOperation(new T[]{tensor1, tensor2}, "convolution"));
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
                        for(int j=0; j<current.shape().length; j++){
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



