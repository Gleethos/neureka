
package neureka.function;

import neureka.Tsr;
import neureka.function.environment.Types;
import neureka.function.factory.autograd.GraphLock;
import neureka.function.factory.autograd.GraphNode;
import neureka.function.factory.assembly.FunctionBuilder;
import neureka.function.environment.Cache;

public interface Function
{
    Cache CACHE = new Cache();
    Types TYPES = new Types();
    //------------------------------------------------------------------------------------------------------------------

    class setup
    {
        public static Tsr commit(Tsr[] tensors, String operation, boolean doAD) {
            return commit(tensors, FunctionBuilder.build(operation, doAD));
        }

        public static Tsr commit(Tsr[] inputs, Function function) {
            GraphLock newLock = new GraphLock(function, inputs);
            for (Tsr t : inputs) {
                if(t.has(GraphNode.class)){
                    ((GraphNode)t.find(GraphNode.class)).optainLocking(newLock);
                } else {
                    t.add(new GraphNode(t, null, null, newLock));
                }
            }
            Tsr result = (function.activate(inputs));
            if (result.has(GraphNode.class)) {
                ((GraphNode) result.find(GraphNode.class)).redundantGradientCleanup();
            }
            Function.CACHE.free(newLock);
            //for (Tsr t : inputs) {
            //    t.setGradientIsTargeted(false);
            //}
            return result;
        }
    }

    //------------------------------------------------------------------------------------------------------------------
    Function newBuild(String expression);

    boolean isFlat();

    int id();

    String type();

    //------------------------------------------------------------------------------------------------------------------
    double activate(double[] inputs, int j);// Iteration over input via j !

    double activate(double[] inputs);

    double derive(double[] inputs, int index, int j);

    double derive(double[] inputs, int index);

    //------------------------------------------------------------------------------------------------------------------
    Tsr activate(Tsr[] inputs, int j);// Iteration over input via j !

    Tsr activate(Tsr[] inputs);

    Tsr derive(Tsr[] inputs, int index, int j);

    Tsr derive(Tsr[] inputs, int index);

    //------------------------------------------------------------------------------------------------------------------
    String toString();
}

 