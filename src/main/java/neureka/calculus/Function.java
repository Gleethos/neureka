
package neureka.calculus;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.calculus.environment.OperationType;
import neureka.calculus.factory.assembly.FunctionBuilder;
import neureka.calculus.environment.Cache;

import java.util.function.Supplier;

public interface Function
{
    //Global context and cache:
    Cache CACHE = Cache.instance();

    Function IDY = create("I[0]<-Ii[1]");

    Function X = create("I[0]xI[1]");
    Function PLUS = create("(I[0]+I[1])");
    Function PLUS_ASSIGN = create("I[0]<-(I[0]+I[1])");
    Function MINUS = create("(I[0]-I[1])");
    Function MINUS_ASSIGN = create("I[0]<-(I[0]-I[1])");
    Function DIV = create("(I[0]/I[1])");
    Function DIV_ASSIGN = create("I[0]<-(I[0]/I[1])");
    Function POW = create("(I[0]^I[1])");
    Function POW_ASSIGN = create("I[0]<-(I[0]^I[1])");
    Function MUL = create("I[0]*I[1]");
    Function MUL_ASSIGN = create("I[0]<-(I[0]*I[1])");
    Function MOD = create("(I[0]%I[1])");
    Function MOD_ASSIGN = create("I[0]<-(I[0]%I[1])");
    Function NEG = create("(-1*I[0])");

    class Detached{
        public static  Function IDY = create("I[0]<-I[1]", false);

        public static Function X = create("I[0]xI[1]", false);
        public static Function PLUS = create("(I[0]+I[1])", false);
        public static Function PLUS_ASSIGN = create("I[0]<-(I[0]+I[1])", false);
        public static Function MINUS = create("(I[0]-I[1])", false);
        public static Function MINUS_ASSIGN = create("I[0]<-(I[0]-I[1])", false);
        public static Function DIV = create("(I[0]/I[1])", false);
        public static Function DIV_ASSIGN = create("I[0]<-(I[0]/I[1])", false);
        public static Function POW = create("(I[0]^I[1])", false);
        public static Function POW_ASSIGN = create("I[0]<-(I[0]^I[1])", false);
        public static Function MUL = create("I[0]*I[1]", false);
        public static Function MUL_ASSIGN = create("I[0]<-(I[0]*I[1])", false);
        public static Function MOD = create("(I[0]%I[1])", false);
        public static Function MOD_ASSIGN = create("I[0]<-(I[0]%I[1])", false);
        public static Function NEG = create("(-1*I[0])", false);
    }

    static Function create(String expression){
        return create(expression, true);
    }

    static Function create(String expression, boolean doAD){
        return FunctionBuilder.build(expression, doAD);
    }

    class Setup
    {
        public static Tsr commit(Tsr[] tensors, String operation, boolean doAD) {
            return commit(null, tensors, FunctionBuilder.build(operation, doAD));
        }

        public static Tsr commit(Tsr drain, Tsr[] tensors, String operation, boolean doAD) {
            return commit(drain, tensors, FunctionBuilder.build(operation, doAD));
        }

        public static Tsr commit(Tsr[] inputs, Function function) {
            return commit(null, inputs, function);
        }

        public static Tsr commit(Tsr drain, Tsr[] inputs, Function function) {
            return commit(drain, inputs, function, null);
        }

        public static Tsr commit(Tsr drain, Tsr[] inputs, Function function, Supplier<Tsr> activation){

            Tsr.makeFit(inputs);// reshaping if needed

            GraphLock newLock = new GraphLock(function, inputs);
            for (Tsr t : inputs) {
                if(t.has(GraphNode.class)) ((GraphNode)t.find(GraphNode.class)).obtainLocking(newLock);
                else new GraphNode(function, newLock, ()-> t);
            }
            Tsr result = null;
            if(activation==null) result = function.activate(inputs);
            else result = activation.get();

            Function.CACHE.free(newLock);
            boolean resultIsUnique = true;
            if(drain!=null){
                for(Tsr t : inputs){
                    Tsr g = (Tsr)t.find(Tsr.class);
                    if (t == result || (g!=null && g==result)) {
                        resultIsUnique = false;
                        break;
                    }
                }
            }
            if(resultIsUnique) return result;
            else return null;
        }

    }

    //------------------------------------------------------------------------------------------------------------------
    Function newBuild(String expression);

    boolean doesAD();//Note: only branch nodes can 'do Auto-Differentiation'

    boolean isFlat();

    int id();

    OperationType type();

    boolean dependsOn(int index);
    //------------------------------------------------------------------------------------------------------------------
    double activate(double input);

    double activate(double[] inputs, int j);// Iteration over input via j !

    double activate(double[] inputs);

    double derive(double[] inputs, int index, int j);

    double derive(double[] inputs, int index);

    //------------------------------------------------------------------------------------------------------------------
    Tsr activate(Tsr input);

    Tsr activate(Tsr[] inputs, int j);// Iteration over input via j !

    Tsr activate(Tsr[] inputs);

    Tsr derive(Tsr[] inputs, int index, int j);

    Tsr derive(Tsr[] inputs, int index);

    //---
    String toString();

    ADAgent getADAgent(Tsr[] inputs, int i, boolean forward);

    //------------------------------------------------------------------------------------------------------------------

}

 