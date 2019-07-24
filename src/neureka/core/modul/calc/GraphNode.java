package neureka.core.modul.calc;

import neureka.core.T;
import neureka.core.modul.calc.fcomp.Function;
import neureka.core.modul.calc.gcomp.RelativeGradients;

import java.util.function.Consumer;

public class GraphNode {

    Function Fcn = null;
    T[] Src = null;
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
    public GraphNode(T drain, T[] src, int[][] translation, String operation){
        T[] translated = new T[src.length];
        for(int i=0; i<translated.length&&i<translation.length; i++){
            translated[i] = T.factory.reshapedCopyOf(src[i], translation[i]);//src[i].reshaped(translation[i]);
        }
    }
    public GraphNode(T drain, T[] src, String operation, boolean forward, boolean derive){//, boolean derive
        construct(operation, src);
        if(this.Fcn.isFlat()){
            configure(drain, src, true);
        }
        if(forward){
            forward(drain);
        }
        if(this.Fcn.isFlat()&&derive){//derive &&
            performDifferentiation(drain);
        }
    }
    public GraphNode(T drain, T[] src, int f_id, boolean forward, boolean derive){//, boolean derive
        construct(f_id, src);
        configure(drain, src, forward);
        if(forward){
            forward(drain);
        }
        if(this.Fcn.isFlat()&&derive){
            performDifferentiation(drain);
        }
    }

    public GraphNode(T drain, T[] src, String operation, boolean forward){//, boolean derive
        construct(operation, src);
        configure(drain, src, true);
        if(forward){
            forward(drain);
        }
        if(this.Fcn.isFlat()){//derive &&
            performDifferentiation(drain);
        }
    }
    private void construct(int f_id, T[] source){
        this.Src = source;
        this.Fcn = new FunctionFactory().newBuild(f_id, source.length);//, tipReached);
    }
    private void construct(String operation, T[] source) {
        this.Src = source;
        Fcn = new FunctionFactory().newBuild(operation);
        validate(operation);
    }

    private void validate(String operation){
        /**
         *   Evaluating validity:
         * */
        if(!operation.contains("x")){
            if(!utility.isValid(this.Src, false)){
                this.Fcn = null;
                this.Src = null;
                return;
            }
        }else if(operation.contains("x") && !utility.isValid(this.Src, true)){//Shape fitting:
            int largest = 0;
            for(int i = 0; i<this.Src.length; i++){
                largest = (this.Src[i].shape().length>largest)?this.Src[i].shape().length:largest;
            }
            int[][] newShapes = new int[this.Src.length][largest];
            for(int i = 0; i<this.Src.length; i++){
                for(int ii=0; ii<largest; ii++){
                    newShapes[i][ii] = (ii<this.Src[i].shape().length)?ii:-1;
                }
                this.Src[i].reshape(newShapes[i]);
            }
        }
    }

    private void configure(T drain, T[] source, boolean forward){
        /**
         *  Increment reference counter
         *              &
         *  Evaluate auto-grad mode:
         * */
        this.mode = 0;
        int[] srcModes = new int[this.Src.length];
        int m = 0;
        for(int Ii = 0; Ii< this.Src.length; Ii++){
            if(Src[Ii].has(GraphNode.class)){
                GraphNode calc = (GraphNode) Src[Ii].find(GraphNode.class);
                calc.referenced++;
                srcModes[Ii] = calc.mode;
            }else if(Src[Ii].rqsGradient()){
                srcModes[Ii] = 1;
            }
            m += (srcModes[Ii]!=0)?1:0;
        }
        if(m==1 && (this.Fcn.id()!=18)){
            for(int Ii = 0; Ii< this.Src.length; Ii++){
                mode += (srcModes[Ii]<0)?1:srcModes[Ii];
            }
        }else{
            mode = -m;
        }
    }

    private void forward(T drain){
        if(Src ==null || (Fcn ==null)){
            return;
        }
        if(Fcn !=null){
            drain.internalize(Fcn.activate(Src));
        }
    }

    private void performDifferentiation(T drain)
    {//--------------------------------------------------------------------------------------
        if(this.usesAD() && Fcn.isFlat()){
            if(!drain.has(RelativeGradients.class)){
                RelativeGradients rg = new RelativeGradients();
                drain.addModule(rg);
            }
            RelativeGradients drain_gradients = (RelativeGradients) drain.find(RelativeGradients.class);
            /**
             *  Preparing for backpropagation:
             * */
            if(this.usesForwardAD()){
                int[] i = {0};
                this.foreach((src)->{
                    if(src.has(GraphNode.class) && ((GraphNode) src.find(GraphNode.class)).Fcn.id()==18){
                        T d = Fcn.derive(this.Src, i[0]);
                        drain_gradients.put(src, d);// Sources created by x-mul are revers-mode cases!
                    }else{
                        RelativeGradients src_gradients = (RelativeGradients) src.find(RelativeGradients.class);
                        if(src_gradients!=null){
                            T d = Fcn.derive(this.Src, i[0]);
                            src_gradients.forEach(
                                (t, g)->{
                                    /**
                                     *  Chain rule for every gradient with respect to leaves:
                                     * */
                                    if(drain_gradients.has(t)){
                                        T sg = drain_gradients.get(t);
                                        drain_gradients.put(t, T.factory.addition(sg,T.factory.multiplication(d, g)));
                                    }else{
                                        drain_gradients.put(t, T.factory.multiplication(d, g));
                                    }
                                    //TODO: flag within src tsrs that grant that the tensor has been created by function constructor!
                                });
                        }
                        i[0]++;
                    }
                });
            }else if(this.usesReverseAD()){
                int[] i = {0};
                this.foreach((src)->{
                    if(src.has(RelativeGradients.class) || src.rqsGradient()){
                        T d = Fcn.derive(this.Src, i[0]);
                        drain_gradients.put(src, d);// Add gradients with respect to every source tensor!
                    }
                    i[0]++;
                });
            }
        }
    }//--------------------------------------------------------------------------------------

    private void foreach(Consumer<T> action){
        for(int i = 0; i<this.Src.length; i++){
            action.accept(this.Src[i]);
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
        RelativeGradients drnGradients = (RelativeGradients) drain.find(RelativeGradients.class);
        if(drnGradients!=null){
            drnGradients.forEach((target, g)->{
             //   target.backward(T.factory.multiplication(error, g));
            });
        }
    }

    /**
     *   tensor3 = new T().of(new GraphNode(new T[]{tensor1, tensor2}, "convolution"));
     *   tensor4 = new GraphNode(tensor3, "sum(tanh[Ij])").out();
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



