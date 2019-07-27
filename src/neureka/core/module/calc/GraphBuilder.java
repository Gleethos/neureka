package neureka.core.module.calc;

import neureka.core.T;
import neureka.core.module.calc.fcomp.Function;
import neureka.core.module.calc.gcomp.GradientNode;

import java.util.function.Consumer;

public class GraphBuilder {

    //Function Fcn = null;
    //T[] Src = null;
    //int referenced = 0;
    //int mode = 0;
    private static boolean usesAD(int mode){return (mode !=0);}
    private static boolean usesForwardAD(int mode){ return (mode >0); }
    private static boolean usesReverseAD(int mode){ return (mode <0); }
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
    public GraphBuilder(T drain, T[] src, int[][] translation, String operation){
        T[] translated = new T[src.length];
        for(int i=0; i<translated.length&&i<translation.length; i++){
            translated[i] = T.factory.reshapedCopyOf(src[i], translation[i]);//src[i].reshaped(translation[i]);
        }
    }
    public GraphBuilder(T drain, T[] src, String operation, boolean forward, boolean derive){//, boolean derive
        Function function = construct(operation, src);
        //if(function.isFlat()){
            int mode = configure(src, function);
        //}
        if(function.isFlat()&&derive){//derive &&
            performDifferentiation(drain, function, src, mode);
        }
    }
    public static void connect(T drain, T[] src, int f_id, boolean derive){//, boolean derive
        Function function = construct(f_id, src);
        int mode = configure(src, function);
        if(function.isFlat()&&derive){
            performDifferentiation(drain, function, src, mode);
        }
    }

    private static Function construct(int f_id, T[] source){
        Function function = new FunctionFactory().newBuild(f_id, source.length);//, tipReached);
        return function;
    }
    private static Function construct(String operation, T[] source) {
        Function function = new FunctionFactory().newBuild(operation);
        validate(operation, function, source);
        return function;
    }

    private static void validate(String operation, Function function, T[] source){
        /**
         *   Evaluating validity:
         * */
        if(!operation.contains("x")){
            if(!utility.isValid(source, false)){
                //this.Fcn = null;
                //this.Src = null;
                return;
            }
        }else if(operation.contains("x") && !utility.isValid(source, true)){//Shape fitting:
            int largest = 0;
            for(int i = 0; i<source.length; i++){
                largest = (source[i].shape().length>largest)?source[i].shape().length:largest;
            }
            int[][] newShapes = new int[source.length][largest];
            for(int i = 0; i<source.length; i++){
                for(int ii=0; ii<largest; ii++){
                    newShapes[i][ii] = (ii<source[i].shape().length)?ii:-1;
                }
                source[i].reshape(newShapes[i]);
            }
        }
    }

    private static int configure(T[] source, Function function){
        /**
         *  Increment reference counter
         *              &
         *  Evaluate auto-grad mode:
         * */
        int mode = 0;
        int[] srcModes = new int[source.length];
        int m = 0;
        for(int Ii = 0; Ii< source.length; Ii++){
            if(source[Ii].has(GraphBuilder.class)){//TODO: Fuck graph nodes! Relative GRadients are the new nodes! They contain mode !!!!!!
                GraphBuilder node = (GraphBuilder) source[Ii].find(GraphBuilder.class);//That's better! this should retun rg, with mode or not!(null)
                //node.referenced++;
                //srcModes[Ii] = node.mode;
            }else if(source[Ii].has(GradientNode.class)){
                GradientNode node = (GradientNode) source[Ii].find(GradientNode.class);//That's better! this should retun rg, with mode or not!(null)
                //node.referenced++;
                srcModes[Ii] = node.mode();
            }else if(source[Ii].rqsGradient()){
                srcModes[Ii] = 1;
            }
            m += (srcModes[Ii]!=0)?1:0;
        }
        if(m==1 && (function.id()!=18)){
            for(int Ii = 0; Ii< source.length; Ii++){
                mode += (srcModes[Ii]<0)?1:srcModes[Ii];
            }
        }else{
            mode = -m;
        }
        return mode;
    }

    private static void performDifferentiation(T drain, Function function, T[] source, int m)
    {//--------------------------------------------------------------------------------------
        if(usesAD(m) && function.isFlat()){
            foreach(source, (src)->{
                
            });
            if(!drain.has(GradientNode.class)){
                GradientNode rg = new GradientNode(m, function);
                drain.addModule(rg);
            }
            GradientNode drain_gradients = (GradientNode) drain.find(GradientNode.class);
            /**
             *  Preparing for back      propagation:
             * */
            if(usesForwardAD(m)){
                int[] i = {0};
                foreach(source, (src)->{
                    if(src.has(GradientNode.class) && ((GradientNode) src.find(GradientNode.class)).function().id()==18){
                        T d = function.derive(source, i[0]);
                        drain_gradients.put(src, d);// Sources created by x-mul are revers-mode cases!
                    }else{
                        GradientNode src_gradients = (GradientNode) src.find(GradientNode.class);
                                if(src_gradients!=null){
                            T d = function.derive(source, i[0]);
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
            }else if(usesReverseAD(m)){
                int[] i = {0};
                foreach(source, (src)->{
                    if(src.has(GradientNode.class) || src.rqsGradient()){
                        T d = function.derive(source, i[0]);
                        drain_gradients.put(src, d);// Add gradients with respect to every source tensor!
                    }
                    i[0]++;
                });
            }
        }
    }//--------------------------------------------------------------------------------------

    private static void foreach(T[] source, Consumer<T> action){
        for(int i = 0; i<source.length; i++){
            action.accept(source[i]);
        }
    }

    public static void backward(T error, T drain){
        if(!usesAD(0)){
            return;
        }
        if(drain.rqsGradient()){
            drain.setGradient(
                    (drain.gradient()==null)
                            ?error
                            :T.factory.addition(error, T.factory.newTensor(drain.value(), drain.shape()))
            );
        }
        GradientNode drnGradients = (GradientNode) drain.find(GradientNode.class);
        if(drnGradients!=null){
            drnGradients.forEach((target, g)->{
             //   target.backward(T.factory.multiplication(error, g));
            });
        }
    }

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



