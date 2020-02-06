
package neureka.calculus;

import neureka.Tsr;
import neureka.autograd.ADAgent;
import neureka.calculus.environment.Types;
import neureka.autograd.GraphLock;
import neureka.autograd.GraphNode;
import neureka.calculus.factory.assembly.FunctionBuilder;
import neureka.calculus.environment.Cache;

import java.util.function.Supplier;

public interface Function
{
    //Global context and cache:
    Cache CACHE = new Cache();
    Types TYPES = new Types();

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

            GraphLock newLock = new GraphLock(function, inputs);
            for (Tsr t : inputs) {
                if(t.has(GraphNode.class)){
                    ((GraphNode)t.find(GraphNode.class)).obtainLocking(newLock);
                } else {
                    new GraphNode(function, newLock, ()->t);
                }
            }
            Tsr result = null;
            if(activation==null){
                result = function.activate(inputs);
            } else {
                result = activation.get();
            }
            Function.CACHE.free(newLock);
            boolean resultIsUnique = true;
            if(drain!=null){
                for(Tsr t : inputs){
                    Tsr g = (Tsr)t.find(Tsr.class);
                    if(t == result || (g!=null && g==result)){
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

    String type();

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

 